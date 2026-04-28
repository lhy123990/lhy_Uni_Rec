"""Online GPU memory benchmark for PCVRHyFormer.

This variant is meant for managed/online experiment platforms where report
files cannot be retrieved. It reads the same core paths from environment
variables as ``train.py`` and emits every result through ``logging.info``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List

import torch

from benchmark_gpu_memory import (
    bytes_to_gib,
    bytes_to_mib,
    measure_one,
    parse_int_list,
    parse_seq_max_lens,
    pick_recommendation,
    run_equivalence,
    set_seed,
)
from dataset import PCVRParquetDataset


def env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Online GPU memory benchmark that logs via logging.info"
    )
    parser.add_argument(
        "--data_dir",
        default=env_value("TRAIN_DATA_PATH", default="data_sample_1000"),
        help="Data directory. Defaults to env TRAIN_DATA_PATH.",
    )
    parser.add_argument(
        "--schema_path",
        default=env_value("TRAIN_SCHEMA_PATH"),
        help="Schema path. Defaults to env TRAIN_SCHEMA_PATH or <data_dir>/schema.json.",
    )
    parser.add_argument(
        "--device",
        default=env_value("TRAIN_DEVICE", "CUDA_DEVICE", default="cuda:0"),
        help="Torch device. Defaults to env TRAIN_DEVICE/CUDA_DEVICE or cuda:0.",
    )
    parser.add_argument(
        "--batch_sizes",
        default=env_value(
            "MEMORY_BENCH_BATCH_SIZES",
            "TRAIN_BATCH_SIZE",
            default="256",
        ),
        help="Comma-separated batch sizes. Defaults to MEMORY_BENCH_BATCH_SIZES, "
             "then TRAIN_BATCH_SIZE, then 256.",
    )
    parser.add_argument(
        "--memory_budget_gib",
        type=float,
        default=float(env_value("MEMORY_BENCH_BUDGET_GIB", default="20")),
        help="Memory budget in GiB. Defaults to env MEMORY_BENCH_BUDGET_GIB or 20.",
    )
    parser.add_argument(
        "--seq_max_lens",
        default=env_value(
            "MEMORY_BENCH_SEQ_MAX_LENS",
            "TRAIN_SEQ_MAX_LENS",
            default="seq_a:256,seq_b:256,seq_c:512,seq_d:512",
        ),
        help="Per-domain sequence max lens.",
    )
    parser.add_argument(
        "--recommend_max_units",
        type=int,
        default=int(env_value("MEMORY_BENCH_RECOMMEND_MAX_UNITS", default="4")),
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=int(env_value("MEMORY_BENCH_WARMUP_STEPS", default="1")),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(env_value("TRAIN_SEED", "MEMORY_BENCH_SEED", default="42")),
    )
    parser.add_argument(
        "--check_equivalence",
        action="store_true",
        default=env_value("MEMORY_BENCH_CHECK_EQUIVALENCE", default="0") == "1",
    )
    parser.add_argument(
        "--log_json",
        action="store_true",
        default=env_value("MEMORY_BENCH_LOG_JSON", default="0") == "1",
        help="Also log the full JSON payload as one logging.info message.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )


def load_probe_batch(
    data_dir: str,
    schema_path: str,
    batch_size: int,
    seq_max_lens: Dict[str, int],
) -> tuple[PCVRParquetDataset, Dict[str, Any]]:
    """Load only one CPU batch.

    The offline benchmark can materialize all batches because it normally runs
    on ``data_sample_1000``. Online datasets are much larger, so collecting the
    full iterable into a list can exhaust host memory before the GPU probe even
    starts.
    """
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
    try:
        first_batch = next(iter(dataset))
    except StopIteration as exc:
        raise RuntimeError(f"No CPU batches available for batch_size={batch_size}") from exc
    return dataset, first_batch


def log_recommendation(payload: Dict[str, Any]) -> None:
    rec = payload["recommendation"]
    train_args = f"--activation_checkpoint_mode {rec['mode']}"
    if rec["units"]:
        train_args += " --activation_checkpoint_units " + ",".join(rec["units"])
    logging.info("Checkpoint recommendation: %s", train_args)
    logging.info("Recommendation reason: %s", rec["reason"])
    logging.info("Memory budget: %.2f GiB", rec["budget_gib"])
    if rec.get("top_units"):
        logging.info("Top checkpoint candidates:")
        for item in rec["top_units"]:
            logging.info(
                "  %s saved_activation=%.2f MiB",
                item["unit"],
                item["saved_activation_mib"],
            )


def log_batch_results(payload: Dict[str, Any]) -> None:
    logging.info(
        "Device: %s (%s), total=%.2f GiB",
        payload["device"],
        payload["device_name"],
        bytes_to_gib(payload["device_total_bytes"]),
    )
    logging.info("Data: %s", payload["data_dir"])
    logging.info("Schema: %s", payload["schema_path"])
    for row in payload["batch_results"]:
        batch_size = row["batch_size"]
        baseline = row["baseline"]
        if baseline["oom"]:
            logging.info(
                "batch=%s baseline OOM peak_alloc=%.3f GiB peak_reserved=%.3f GiB error=%s",
                batch_size,
                bytes_to_gib(baseline.get("peak_allocated_bytes", 0)),
                bytes_to_gib(baseline.get("peak_reserved_bytes", 0)),
                baseline.get("error", "OOM"),
            )
            continue
        logging.info(
            "batch=%s baseline peak_alloc=%.3f GiB peak_reserved=%.3f GiB "
            "gpu=%.4fs wall=%.4fs loss=%.6f",
            batch_size,
            bytes_to_gib(baseline["peak_allocated_bytes"]),
            bytes_to_gib(baseline["peak_reserved_bytes"]),
            baseline["gpu_sec"],
            baseline["wall_sec"],
            baseline["loss"],
        )
        top_saved = sorted(
            baseline["saved_activation_bytes_by_unit"].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:payload["recommend_max_units"]]
        for unit, value in top_saved:
            logging.info(
                "batch=%s unit=%s saved_activation=%.2f MiB",
                batch_size,
                unit,
                bytes_to_mib(value),
            )
        if "recommended" in row:
            recommended = row["recommended"]
            if recommended["oom"]:
                logging.info(
                    "batch=%s recommended profile OOM error=%s",
                    batch_size,
                    recommended.get("error", "OOM"),
                )
            else:
                logging.info(
                    "batch=%s recommended profile peak_alloc=%.3f GiB "
                    "peak_reserved=%.3f GiB gpu=%.4fs wall=%.4fs",
                    batch_size,
                    bytes_to_gib(recommended["peak_allocated_bytes"]),
                    bytes_to_gib(recommended["peak_reserved_bytes"]),
                    recommended["gpu_sec"],
                    recommended["wall_sec"],
                )


def log_equivalence(payload: Dict[str, Any]) -> None:
    eq = payload.get("equivalence")
    if not eq:
        return
    logging.info(
        "Checkpoint equivalence units=%s logits_max_abs_diff=%.10e pass_logits=%s "
        "grad_max_abs_diff=%.10e pass_grads=%s compared_dense_grad_tensors=%s",
        ",".join(eq["units"]) if eq["units"] else "<none>",
        eq["logits_max_abs_diff"],
        eq["pass_logits"],
        eq["grad_max_abs_diff"],
        eq["pass_grads"],
        eq["compared_dense_grad_tensors"],
    )


def main() -> None:
    setup_logging()
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_seed(args.seed)

    data_dir = os.path.abspath(args.data_dir)
    schema_path = os.path.abspath(args.schema_path or os.path.join(data_dir, "schema.json"))
    batch_sizes = parse_int_list(args.batch_sizes)
    seq_max_lens = parse_seq_max_lens(args.seq_max_lens)
    logging.info(
        "Online memory benchmark started: data_dir=%s schema_path=%s "
        "device=%s batch_sizes=%s memory_budget_gib=%.2f seq_max_lens=%s",
        data_dir,
        schema_path,
        device,
        batch_sizes,
        args.memory_budget_gib,
        args.seq_max_lens,
    )

    batch_results: List[Dict[str, Any]] = []
    datasets: Dict[int, PCVRParquetDataset] = {}
    batches: Dict[int, Dict[str, Any]] = {}
    for batch_size in batch_sizes:
        logging.info("Loading data for batch_size=%s", batch_size)
        dataset, cpu_batch = load_probe_batch(data_dir, schema_path, batch_size, seq_max_lens)
        datasets[batch_size] = dataset
        batches[batch_size] = cpu_batch
        logging.info("Running baseline memory probe for batch_size=%s", batch_size)
        baseline = measure_one(
            dataset=dataset,
            batch=cpu_batch,
            device=device,
            seed=args.seed,
            checkpoint_units=(),
            warmup_steps=args.warmup_steps,
            count_saved_tensors=True,
        )
        batch_results.append({"batch_size": batch_size, "baseline": baseline})

    props = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
    payload = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "device_total_bytes": int(props.total_memory) if props is not None else 0,
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
        logging.info(
            "Running recommended checkpoint profile for batch_size=%s units=%s",
            target_batch_size,
            ",".join(units),
        )
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
        logging.info("Running checkpoint equivalence check")
        payload["equivalence"] = run_equivalence(
            datasets[target_batch_size],
            batches[target_batch_size],
            device,
            args.seed,
            units,
        )

    log_recommendation(payload)
    log_batch_results(payload)
    log_equivalence(payload)
    if args.log_json:
        logging.info("Full memory benchmark JSON: %s", json.dumps(payload, sort_keys=True))
    logging.info("Online memory benchmark finished")


if __name__ == "__main__":
    main()
