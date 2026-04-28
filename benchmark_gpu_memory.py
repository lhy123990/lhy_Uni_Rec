"""GPU memory benchmark and activation-checkpoint recommendation for PCVRHyFormer.

This script is intentionally separate from ``benchmark_gpu_epoch.py`` so the
existing efficiency benchmark keeps measuring epoch speed without extra memory
probe logic.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import random
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dataset import NUM_TIME_BUCKETS, PCVRParquetDataset
from model import ActivationCheckpointConfig, ModelInput, PCVRHyFormer
from train import build_feature_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU memory benchmark for PCVRHyFormer activation checkpointing"
    )
    parser.add_argument("--data_dir", default="data_sample_1000")
    parser.add_argument("--schema_path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_sizes", default="128,256,512,1024")
    parser.add_argument("--seq_max_lens", default="seq_a:256,seq_b:256,seq_c:512,seq_d:512")
    parser.add_argument("--memory_budget_gib", type=float, default=0.0)
    parser.add_argument("--recommend_max_units", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--check_equivalence", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_seq_max_lens(spec: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not spec:
        return out
    for pair in spec.split(","):
        k, v = pair.split(":")
        out[k.strip()] = int(v.strip())
    return out


def parse_int_list(spec: str) -> List[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def load_cpu_batches(
    data_dir: str,
    schema_path: str,
    batch_size: int,
    seq_max_lens: Dict[str, int],
) -> Tuple[PCVRParquetDataset, List[Dict[str, Any]]]:
    dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=None,
        clip_vocab=True,
        is_training=True,
    )
    return dataset, list(iter(dataset))


def build_model(dataset: PCVRParquetDataset, device: torch.device) -> PCVRHyFormer:
    user_ns_groups = [[i] for i in range(len(dataset.user_int_schema.entries))]
    item_ns_groups = [[i] for i in range(len(dataset.item_int_schema.entries))]
    model = PCVRHyFormer(
        user_int_feature_specs=build_feature_specs(
            dataset.user_int_schema, dataset.user_int_vocab_sizes
        ),
        item_int_feature_specs=build_feature_specs(
            dataset.item_int_schema, dataset.item_int_vocab_sizes
        ),
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        d_model=64,
        emb_dim=64,
        num_queries=2,
        num_hyformer_blocks=2,
        num_heads=4,
        seq_encoder_type="transformer",
        hidden_mult=4,
        dropout_rate=0.01,
        seq_top_k=50,
        seq_causal=False,
        action_num=1,
        num_time_buckets=NUM_TIME_BUCKETS,
        rank_mixer_mode="full",
        use_rope=False,
        rope_base=10000.0,
        emb_skip_threshold=1000000,
        seq_id_threshold=10000,
        ns_tokenizer_type="rankmixer",
        user_ns_tokens=5,
        item_ns_tokens=2,
        sparse_embeddings=False,
    )
    return model.to(device).train()


def batch_to_model_input(batch: Dict[str, Any], device: torch.device) -> Tuple[ModelInput, torch.Tensor]:
    def move(t: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        if dtype is not None:
            t = t.to(dtype=dtype)
        return t.to(device, non_blocking=False)

    seq_domains = batch["_seq_domains"]
    inputs = ModelInput(
        user_int_feats=move(batch["user_int_feats"]),
        item_int_feats=move(batch["item_int_feats"]),
        user_dense_feats=move(batch["user_dense_feats"], torch.float32),
        item_dense_feats=move(batch["item_dense_feats"], torch.float32),
        seq_data={d: move(batch[d]) for d in seq_domains},
        seq_lens={d: move(batch[f"{d}_len"]) for d in seq_domains},
        seq_time_buckets={d: move(batch[f"{d}_time_bucket"]) for d in seq_domains},
    )
    return inputs, move(batch["label"], torch.float32)


@contextmanager
def saved_tensor_counter(model: PCVRHyFormer, enabled: bool = True):
    stats: Dict[str, int] = {}
    if not enabled:
        yield stats
        return

    def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
        unit = model.current_activation_unit or "__outside_units__"
        if tensor.is_cuda:
            stats[unit] = stats.get(unit, 0) + tensor.numel() * tensor.element_size()
        return tensor

    def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        yield stats


def make_optimizers(model: PCVRHyFormer) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer | None]:
    dense_optimizer = torch.optim.AdamW(model.get_dense_params(), lr=1e-4, betas=(0.9, 0.98))
    sparse_params = model.get_sparse_params()
    sparse_optimizer = torch.optim.Adagrad(sparse_params, lr=0.05, weight_decay=0.0)
    return dense_optimizer, sparse_optimizer


def train_step(
    model: PCVRHyFormer,
    dense_optimizer: torch.optim.Optimizer,
    sparse_optimizer: torch.optim.Optimizer | None,
    inputs: ModelInput,
    label: torch.Tensor,
    count_saved_tensors: bool,
) -> Tuple[float, Dict[str, int]]:
    dense_optimizer.zero_grad(set_to_none=True)
    if sparse_optimizer is not None:
        sparse_optimizer.zero_grad(set_to_none=True)

    with saved_tensor_counter(model, enabled=count_saved_tensors) as saved_bytes:
        logits = model(inputs).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, label)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.get_dense_params() if p.grad is not None and not p.grad.is_sparse],
        max_norm=1.0,
        foreach=True,
    )
    dense_optimizer.step()
    if sparse_optimizer is not None:
        sparse_optimizer.step()
    return float(loss.detach().cpu()), saved_bytes


def measure_one(
    *,
    dataset: PCVRParquetDataset,
    batch: Dict[str, Any],
    device: torch.device,
    seed: int,
    checkpoint_units: Tuple[str, ...] = (),
    warmup_steps: int = 1,
    count_saved_tensors: bool = True,
) -> Dict[str, Any]:
    set_seed(seed)
    model = build_model(dataset, device)
    mode = "custom" if checkpoint_units else "none"
    model.configure_activation_checkpointing(
        ActivationCheckpointConfig(mode=mode, units=checkpoint_units)
    )
    dense_optimizer, sparse_optimizer = make_optimizers(model)
    inputs, label = batch_to_model_input(batch, device)

    try:
        for _ in range(warmup_steps):
            train_step(
                model, dense_optimizer, sparse_optimizer,
                inputs, label, count_saved_tensors=False,
            )
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start_event.record()
        loss, saved_bytes = train_step(
            model, dense_optimizer, sparse_optimizer,
            inputs, label, count_saved_tensors=count_saved_tensors,
        )
        end_event.record()
        torch.cuda.synchronize(device)
        wall_sec = time.perf_counter() - wall_start
        gpu_sec = start_event.elapsed_time(end_event) / 1000.0
        result = {
            "ok": True,
            "oom": False,
            "loss": loss,
            "wall_sec": wall_sec,
            "gpu_sec": gpu_sec,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "saved_activation_bytes_by_unit": saved_bytes,
        }
    except torch.cuda.OutOfMemoryError as exc:
        result = {
            "ok": False,
            "oom": True,
            "error": str(exc).split("\n")[0],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "saved_activation_bytes_by_unit": {},
        }
    finally:
        del model, dense_optimizer, sparse_optimizer, inputs, label
        torch.cuda.empty_cache()
        gc.collect()
    return result


def bytes_to_mib(value: int | float) -> float:
    return float(value) / (1024.0 ** 2)


def bytes_to_gib(value: int | float) -> float:
    return float(value) / (1024.0 ** 3)


def pick_recommendation(
    payload: Dict[str, Any],
    memory_budget_gib: float,
    max_units: int,
) -> Dict[str, Any]:
    device_total = payload["device_total_bytes"]
    budget_bytes = int(memory_budget_gib * 1024 ** 3) if memory_budget_gib > 0 else int(device_total * 0.90)
    baseline_rows = [r for r in payload["batch_results"] if r["baseline"]["ok"]]
    if not baseline_rows:
        return {
            "mode": "custom",
            "units": [],
            "reason": "All probed baseline batch sizes OOM; reduce batch size first, then rerun the memory probe.",
            "budget_gib": bytes_to_gib(budget_bytes),
        }

    target = max(baseline_rows, key=lambda r: r["batch_size"])
    larger_oom = any(
        r["batch_size"] > target["batch_size"] and r["baseline"]["oom"]
        for r in payload["batch_results"]
    )
    over_budget = target["baseline"]["peak_reserved_bytes"] > budget_bytes
    saved = target["baseline"]["saved_activation_bytes_by_unit"]
    unit_items = [
        (unit, value) for unit, value in saved.items()
        if unit != "__outside_units__" and value > 0
    ]
    unit_items.sort(key=lambda item: item[1], reverse=True)
    units = [unit for unit, _ in unit_items[:max_units]]

    if not units:
        reason = "No checkpointable activation-heavy units were observed in the probe."
        mode = "none"
    elif over_budget:
        reason = (
            f"Batch size {target['batch_size']} exceeds the memory budget; "
            "checkpoint the largest saved-activation units first."
        )
        mode = "custom"
    elif larger_oom:
        reason = (
            f"Batch size {target['batch_size']} is the largest probed baseline "
            "that completed, but a larger probed batch OOMed; checkpoint the "
            "largest saved-activation units before retrying the larger batch."
        )
        mode = "custom"
    else:
        reason = (
            f"Batch size {target['batch_size']} fits the memory budget; "
            "checkpointing is optional unless you raise batch size further."
        )
        mode = "none"
        units = []

    return {
        "mode": mode,
        "units": units,
        "reason": reason,
        "budget_gib": bytes_to_gib(budget_bytes),
        "target_batch_size": target["batch_size"],
        "top_units": [
            {"unit": unit, "saved_activation_mib": bytes_to_mib(value)}
            for unit, value in unit_items[:max_units]
        ],
    }


def run_equivalence(
    dataset: PCVRParquetDataset,
    batch: Dict[str, Any],
    device: torch.device,
    seed: int,
    units: Tuple[str, ...],
) -> Dict[str, Any]:
    set_seed(seed)
    base = build_model(dataset, device)
    set_seed(seed + 1)
    ckpt = build_model(dataset, device)
    ckpt.load_state_dict(base.state_dict())
    ckpt.configure_activation_checkpointing(
        ActivationCheckpointConfig(mode="custom" if units else "none", units=units)
    )
    inputs, label = batch_to_model_input(batch, device)

    base.zero_grad(set_to_none=True)
    ckpt.zero_grad(set_to_none=True)
    set_seed(seed + 2)
    logits_base = base(inputs).squeeze(-1)
    loss_base = F.binary_cross_entropy_with_logits(logits_base, label)
    loss_base.backward()
    set_seed(seed + 2)
    logits_ckpt = ckpt(inputs).squeeze(-1)
    loss_ckpt = F.binary_cross_entropy_with_logits(logits_ckpt, label)
    loss_ckpt.backward()
    torch.cuda.synchronize(device)

    logits_diff = (logits_base.detach() - logits_ckpt.detach()).abs()
    grad_max_abs = 0.0
    compared = 0
    for (name_a, p_a), (name_b, p_b) in zip(base.named_parameters(), ckpt.named_parameters()):
        if name_a != name_b:
            raise RuntimeError(f"Parameter order mismatch: {name_a} vs {name_b}")
        if p_a.grad is None and p_b.grad is None:
            continue
        if p_a.grad is None or p_b.grad is None:
            raise RuntimeError(f"Gradient presence mismatch for {name_a}")
        if p_a.grad.is_sparse or p_b.grad.is_sparse:
            continue
        grad_max_abs = max(
            grad_max_abs,
            float((p_a.grad.detach() - p_b.grad.detach()).abs().max().cpu()),
        )
        compared += 1

    result = {
        "units": list(units),
        "loss_abs_diff": abs(float(loss_base.detach().cpu()) - float(loss_ckpt.detach().cpu())),
        "logits_max_abs_diff": float(logits_diff.max().cpu()),
        "logits_mean_abs_diff": float(logits_diff.mean().cpu()),
        "grad_max_abs_diff": grad_max_abs,
        "compared_dense_grad_tensors": compared,
        "pass_logits": float(logits_diff.max().cpu()) <= 1e-5,
        "pass_grads": grad_max_abs <= 1e-4,
    }
    del base, ckpt, inputs, label
    torch.cuda.empty_cache()
    gc.collect()
    return result


def format_report(payload: Dict[str, Any]) -> str:
    rec = payload["recommendation"]
    train_args = f"--activation_checkpoint_mode {rec['mode']}"
    if rec["units"]:
        train_args += " --activation_checkpoint_units " + ",".join(rec["units"])
    lines = [
        "Checkpoint recommendation:",
        f"  {train_args}",
        f"  reason: {rec['reason']}",
        f"  budget={rec['budget_gib']:.2f} GiB",
        f"Device: {payload['device']} ({payload['device_name']}), total={bytes_to_gib(payload['device_total_bytes']):.2f} GiB",
        f"Data: {payload['data_dir']}",
        "",
        "Batch memory results:",
    ]
    for row in payload["batch_results"]:
        base = row["baseline"]
        if base["oom"]:
            lines.append(f"  batch={row['batch_size']}: baseline OOM ({base.get('error', 'OOM')})")
            continue
        lines.append(
            f"  batch={row['batch_size']}: "
            f"peak_alloc={bytes_to_gib(base['peak_allocated_bytes']):.3f} GiB, "
            f"peak_reserved={bytes_to_gib(base['peak_reserved_bytes']):.3f} GiB, "
            f"gpu={base['gpu_sec']:.4f}s, loss={base['loss']:.6f}"
        )
        top = sorted(
            base["saved_activation_bytes_by_unit"].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:payload["recommend_max_units"]]
        for unit, value in top:
            lines.append(f"    {unit}: saved_activation={bytes_to_mib(value):.2f} MiB")
        if row.get("recommended"):
            recommended = row["recommended"]
            lines.append(
                f"    recommended profile: peak_reserved={bytes_to_gib(recommended['peak_reserved_bytes']):.3f} GiB, "
                f"gpu={recommended['gpu_sec']:.4f}s"
            )
    if payload.get("equivalence"):
        eq = payload["equivalence"]
        lines.extend([
            "",
            "Checkpoint equivalence:",
            f"  logits_max_abs_diff={eq['logits_max_abs_diff']:.10e}, pass={eq['pass_logits']}",
            f"  grad_max_abs_diff={eq['grad_max_abs_diff']:.10e}, pass={eq['pass_grads']}",
        ])
    return "\n".join(lines)


def write_reports(payload: Dict[str, Any], log_dir: str) -> Tuple[str, str]:
    os.makedirs(log_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(log_dir, f"benchmark_gpu_memory_{stamp}.txt")
    json_path = os.path.join(log_dir, f"benchmark_gpu_memory_{stamp}.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(format_report(payload) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return txt_path, json_path


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    set_seed(args.seed)

    data_dir = os.path.abspath(args.data_dir)
    schema_path = os.path.abspath(args.schema_path or os.path.join(data_dir, "schema.json"))
    batch_sizes = parse_int_list(args.batch_sizes)
    seq_max_lens = parse_seq_max_lens(args.seq_max_lens)

    batch_results = []
    datasets: Dict[int, PCVRParquetDataset] = {}
    batches: Dict[int, Dict[str, Any]] = {}
    for batch_size in batch_sizes:
        dataset, cpu_batches = load_cpu_batches(data_dir, schema_path, batch_size, seq_max_lens)
        if not cpu_batches:
            raise RuntimeError(f"No CPU batches available for batch_size={batch_size}")
        datasets[batch_size] = dataset
        batches[batch_size] = cpu_batches[0]
        baseline = measure_one(
            dataset=dataset,
            batch=cpu_batches[0],
            device=device,
            seed=args.seed,
            checkpoint_units=(),
            warmup_steps=args.warmup_steps,
            count_saved_tensors=True,
        )
        batch_results.append({"batch_size": batch_size, "baseline": baseline})

    props = torch.cuda.get_device_properties(device)
    payload = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_total_bytes": int(props.total_memory),
        "data_dir": data_dir,
        "schema_path": schema_path,
        "batch_sizes": batch_sizes,
        "recommend_max_units": args.recommend_max_units,
        "batch_results": batch_results,
    }
    recommendation = pick_recommendation(
        payload,
        memory_budget_gib=args.memory_budget_gib,
        max_units=args.recommend_max_units,
    )
    payload["recommendation"] = recommendation

    units = tuple(recommendation["units"])
    target_batch_size = recommendation.get("target_batch_size")
    if units and target_batch_size in datasets:
        recommended = measure_one(
            dataset=datasets[target_batch_size],
            batch=batches[target_batch_size],
            device=device,
            seed=args.seed,
            checkpoint_units=units,
            warmup_steps=args.warmup_steps,
            count_saved_tensors=False,
        )
        for row in batch_results:
            if row["batch_size"] == target_batch_size:
                row["recommended"] = recommended
                break

    if args.check_equivalence and target_batch_size in datasets:
        payload["equivalence"] = run_equivalence(
            datasets[target_batch_size],
            batches[target_batch_size],
            device,
            args.seed,
            units,
        )

    txt_path, json_path = write_reports(payload, os.path.abspath(args.log_dir))
    if args.json:
        print(json.dumps(payload, indent=2))
        print(f"report_txt={txt_path}", file=os.sys.stderr)
        print(f"report_json={json_path}", file=os.sys.stderr)
        return
    print(format_report(payload))
    print(f"Report written: {txt_path}")
    print(f"JSON written: {json_path}")


if __name__ == "__main__":
    main()
