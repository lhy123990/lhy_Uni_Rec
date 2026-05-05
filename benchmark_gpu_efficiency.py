"""Benchmark one training epoch on the PCVRHyFormer sample data.

The script compares the original training-step behavior against the optimized
path introduced for gradient clipping. It intentionally avoids checkpointing,
TensorBoard, and training logs so the reported time focuses on one train epoch;
validation AUC/logloss are measured after each epoch to catch behavior changes.

Default model/data arguments mirror ``run.sh`` plus ``train.py`` defaults:
rankmixer tokenizer, 5 user NS tokens, 2 item NS tokens, 2 query tokens,
``emb_skip_threshold=1000000``, and the standard sequence max lengths.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import random
import sys
import time
import types
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from dataset import NUM_TIME_BUCKETS, PCVRParquetDataset
from model import (
    ActivationCheckpointConfig,
    ModelInput,
    PCVRHyFormer,
    apply_rope_to_tensor,
)
from train import build_feature_specs
from utils import sigmoid_focal_loss


@dataclass
class EpochResult:
    variant: str
    repeat: int
    epoch: int
    steps: int
    rows: int
    loss: float
    epoch_wall_sec: float
    epoch_gpu_sec: float
    h2d_wall_sec: float
    val_auc: float
    val_logloss: float
    val_wall_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU epoch benchmark for PCVRHyFormer")
    parser.add_argument("--data_dir", default="data_sample_1000")
    parser.add_argument("--schema_path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--include_sparse_embedding_grads", action="store_true")
    parser.add_argument(
        "--disable_dense_clip",
        action="store_true",
        help="Run only the optimized training path with gradient clipping disabled.",
    )
    parser.add_argument(
        "--attention_chunk_sizes",
        default="256",
        help="Comma-separated chunk sizes for experimental local self-attention "
             "variants. Use an empty string to disable chunk variants.",
    )
    parser.add_argument("--seq_max_lens", default="seq_a:256,seq_b:256,seq_c:512,seq_d:512")
    parser.add_argument("--log_dir", default="logs", help="Directory for benchmark reports")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    parser.add_argument(
        "--amp_dtype",
        choices=["bf16", "fp16", "fp32"],
        default="fp32",
        help="Autocast dtype for train/eval benchmark; fp32 disables autocast.",
    )
    parser.add_argument("--loss_type", choices=["bce", "focal"], default="bce")
    parser.add_argument("--focal_alpha", type=float, default=0.1)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument(
        "--repeat_batch_to_size",
        type=int,
        default=0,
        help="Synthetic benchmark helper: tile each training batch along dim 0 "
             "until it reaches this many rows. Validation batches are kept unchanged.",
    )
    parser.add_argument(
        "--variant",
        choices=[
            "baseline_all_param_clip",
            "optimized_dense_clip",
            "optimized_no_clip",
            "sparse_embedding_grads",
        ],
        default=None,
        help="Run only one training variant; default keeps the historical comparison set.",
    )
    parser.add_argument(
        "--activation_checkpoint_mode",
        choices=["none", "all_blocks", "all_seq_encoders", "custom"],
        default="none",
    )
    parser.add_argument("--activation_checkpoint_units", default="")
    return parser.parse_args()


def parse_mask_equivalence_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare old expanded attention mask vs new broadcast mask"
    )
    parser.add_argument("--data_dir", default="data_sample_1000")
    parser.add_argument("--schema_path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_max_lens", default="seq_a:256,seq_b:256,seq_c:512,seq_d:512")
    parser.add_argument("--log_dir", default="logs", help="Directory for equivalence reports")
    parser.add_argument("--logits_atol", type=float, default=1e-5)
    parser.add_argument("--grads_atol", type=float, default=1e-4)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    return parser.parse_args(argv)


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
    if not spec:
        return []
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def resolve_amp_dtype(device: torch.device, amp_dtype: str) -> torch.dtype | None:
    if device.type != "cuda" or amp_dtype == "fp32":
        return None
    if amp_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 AMP requested, but torch.cuda.is_bf16_supported() is False")
        return torch.bfloat16
    return torch.float16


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def repeat_batch_rows(batch: Dict[str, Any], target_rows: int) -> Dict[str, Any]:
    """Tile tensor rows in a CPU batch to synthesize a larger benchmark batch."""
    if target_rows <= 0:
        return batch

    labels = batch.get("label")
    if not isinstance(labels, torch.Tensor) or labels.dim() == 0:
        return batch

    rows = int(labels.shape[0])
    if rows <= 0 or rows >= target_rows:
        return batch

    idx = torch.arange(target_rows, dtype=torch.long) % rows
    repeated: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0 and int(value.shape[0]) == rows:
            repeated[key] = value.index_select(0, idx.to(value.device))
        else:
            repeated[key] = value
    return repeated


def load_train_valid_batches(
    data_dir: str,
    schema_path: str,
    batch_size: int,
    seq_max_lens: Dict[str, int],
    valid_ratio: float,
) -> Tuple[PCVRParquetDataset, List[Dict[str, Any]], List[Dict[str, Any]]]:
    probe = PCVRParquetDataset(
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
    total_rgs = len(probe._rg_list)
    if total_rgs <= 1:
        n_valid_rgs = 1
        n_train_rgs = 1
    else:
        n_valid_rgs = max(1, int(total_rgs * valid_ratio))
        n_train_rgs = max(1, total_rgs - n_valid_rgs)
        if n_train_rgs + n_valid_rgs > total_rgs:
            n_train_rgs = total_rgs - n_valid_rgs

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(0, n_train_rgs),
        clip_vocab=True,
        is_training=True,
    )
    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(n_train_rgs, total_rgs) if total_rgs > 1 else (0, 1),
        clip_vocab=True,
        is_training=True,
    )
    return train_dataset, list(iter(train_dataset)), list(iter(valid_dataset))


def build_model(
    dataset: PCVRParquetDataset,
    sparse_embeddings: bool,
    device: torch.device,
    attention_chunk_size: int = 0,
    activation_checkpoint_mode: str = "none",
    activation_checkpoint_units: Tuple[str, ...] = (),
) -> PCVRHyFormer:
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
        sparse_embeddings=sparse_embeddings,
    )
    model.configure_activation_checkpointing(
        ActivationCheckpointConfig(
            mode=activation_checkpoint_mode,
            units=activation_checkpoint_units,
        )
    )
    if attention_chunk_size > 0:
        patch_chunked_transformer_attention(model, attention_chunk_size)
    return model.to(device).train()


def patch_chunked_transformer_attention(model: PCVRHyFormer, chunk_size: int) -> None:
    """Use local chunked self-attention inside TransformerEncoder modules.

    This is an experimental benchmark-only shape change. It keeps all tokens,
    but prevents attention across chunk boundaries, so quality must be judged
    with validation metrics rather than treated as a drop-in equivalent.
    """

    def make_forward(encoder):
        def chunked_forward(
            self,
            x: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
            rope_cos: torch.Tensor | None = None,
            rope_sin: torch.Tensor | None = None,
        ) -> Tuple[torch.Tensor, torch.Tensor | None]:
            B, L, _ = x.shape
            if L <= chunk_size:
                return self._benchmark_original_forward(
                    x,
                    key_padding_mask=key_padding_mask,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                )

            residual = x
            x_norm = self.norm1(x)
            chunks = []
            for start in range(0, L, chunk_size):
                end = min(start + chunk_size, L)
                x_chunk = x_norm[:, start:end, :]
                mask_chunk = (
                    key_padding_mask[:, start:end]
                    if key_padding_mask is not None else None
                )
                cos_chunk = rope_cos[:, start:end, :] if rope_cos is not None else None
                sin_chunk = rope_sin[:, start:end, :] if rope_sin is not None else None
                out_chunk, _ = self.self_attn(
                    query=x_chunk,
                    key=x_chunk,
                    value=x_chunk,
                    key_padding_mask=mask_chunk,
                    rope_cos=cos_chunk,
                    rope_sin=sin_chunk,
                )
                chunks.append(out_chunk)
            x = residual + torch.cat(chunks, dim=1)

            residual = x
            x = self.norm2(x)
            x = self.ffn(x)
            x = residual + x
            return x, key_padding_mask

        return types.MethodType(chunked_forward, encoder)

    for block in model.blocks:
        for encoder in block.seq_encoders:
            if encoder.__class__.__name__ == "TransformerEncoder":
                encoder._benchmark_original_forward = encoder.forward
                encoder.forward = make_forward(encoder)


def patch_old_expanded_attention_mask(model: PCVRHyFormer) -> None:
    """Restore the pre-optimization explicit mask expansion for comparison."""

    def make_forward(attn):
        def old_forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
            attn_mask: torch.Tensor | None = None,
            rope_cos: torch.Tensor | None = None,
            rope_sin: torch.Tensor | None = None,
            q_rope_cos: torch.Tensor | None = None,
            q_rope_sin: torch.Tensor | None = None,
            need_weights: bool = False,
        ) -> tuple:
            del need_weights
            B, Lq, _ = query.shape
            Lk = key.shape[1]

            Q = self.W_q(query)
            K = self.W_k(key)
            V = self.W_v(value)

            Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

            if rope_cos is not None and rope_sin is not None:
                K = apply_rope_to_tensor(K, rope_cos, rope_sin)
                if self.rope_on_q:
                    q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                    q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                    Q = apply_rope_to_tensor(Q, q_cos, q_sin)

            sdpa_attn_mask = None
            if key_padding_mask is not None:
                sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)
                sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

            if attn_mask is not None:
                bool_attn = (attn_mask == 0)
                bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(
                    B, self.num_heads, Lq, Lk
                )
                if sdpa_attn_mask is not None:
                    sdpa_attn_mask = sdpa_attn_mask & bool_attn
                else:
                    sdpa_attn_mask = bool_attn

            dropout_p = self.dropout if self.training else 0.0
            out = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=sdpa_attn_mask,
                dropout_p=dropout_p,
            )
            out = torch.nan_to_num(out, nan=0.0)
            out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
            G = self.W_g(query)
            out = out * torch.sigmoid(G)
            out = self.W_o(out)
            return out, None

        return types.MethodType(old_forward, attn)

    for module in model.modules():
        if module.__class__.__name__ == "RoPEMultiheadAttention":
            module.forward = make_forward(module)


def batch_to_model_input(batch: Dict[str, Any], device: torch.device) -> Tuple[ModelInput, torch.Tensor, float]:
    seq_domains = batch["_seq_domains"]
    h2d_start = time.perf_counter()

    def move(t: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        if dtype is not None:
            t = t.to(dtype=dtype)
        return t.to(device, non_blocking=False)

    seq_time_buckets: Dict[str, torch.Tensor] = {}
    seq_time_cont_feats: Dict[str, torch.Tensor] = {}
    seq_abs_time_feats: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        B = batch[domain].shape[0]
        L = batch[domain].shape[2]
        seq_time_buckets[domain] = move(
            batch.get(
                f"{domain}_time_bucket",
                torch.zeros(B, 4, L, dtype=torch.long),
            )
        )
        seq_time_cont_feats[domain] = move(
            batch.get(
                f"{domain}_time_cont",
                torch.zeros(B, 4, L, dtype=torch.float32),
            ),
            torch.float32,
        )
        seq_abs_time_feats[domain] = move(
            batch.get(
                f"{domain}_abs_time_feats",
                torch.zeros(B, 3, L, dtype=torch.long),
            )
        )

    inputs = ModelInput(
        user_int_feats=move(batch["user_int_feats"]),
        item_int_feats=move(batch["item_int_feats"]),
        user_dense_feats=move(batch["user_dense_feats"], torch.float32),
        item_dense_feats=move(batch["item_dense_feats"], torch.float32),
        seq_data={d: move(batch[d]) for d in seq_domains},
        seq_lens={d: move(batch[f"{d}_len"]) for d in seq_domains},
        seq_time_buckets=seq_time_buckets,
        seq_time_cont_feats=seq_time_cont_feats,
        seq_abs_time_feats=seq_abs_time_feats,
    )
    label = move(batch["label"], torch.float32)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    h2d_wall = time.perf_counter() - h2d_start
    return inputs, label, h2d_wall


def iter_device_batches(
    cpu_batches: Iterable[Dict[str, Any]],
    device: torch.device,
) -> Iterable[Tuple[ModelInput, torch.Tensor, int, float]]:
    for batch in cpu_batches:
        inputs, label, h2d_wall = batch_to_model_input(batch, device)
        yield inputs, label, int(label.numel()), h2d_wall


def evaluate(
    *,
    model: PCVRHyFormer,
    cpu_batches: List[Dict[str, Any]],
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> Tuple[float, float, float]:
    was_training = model.training
    model.eval()
    logits_list = []
    labels_list = []
    wall_start = time.perf_counter()

    with torch.no_grad():
        for inputs, label, _, _ in iter_device_batches(cpu_batches, device):
            with autocast_context(device, amp_dtype):
                logits, _ = model.predict(inputs)
            logits_list.append(logits.squeeze(-1).float().detach().cpu())
            labels_list.append(label.detach().cpu())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_sec = time.perf_counter() - wall_start
    if was_training:
        model.train()

    if not logits_list:
        return 0.0, float("inf"), wall_sec

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0).long()
    valid_mask = ~torch.isnan(logits)
    logits = logits[valid_mask]
    labels = labels[valid_mask]
    if logits.numel() == 0:
        return 0.0, float("inf"), wall_sec

    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()
    auc = 0.0 if len(np.unique(labels_np)) < 2 else float(roc_auc_score(labels_np, probs))
    logloss = F.binary_cross_entropy_with_logits(logits, labels.float()).item()
    return auc, logloss, wall_sec


def run_epoch(
    *,
    variant: str,
    repeat: int,
    epoch: int,
    model: PCVRHyFormer,
    dense_optimizer: torch.optim.Optimizer,
    sparse_optimizer: torch.optim.Optimizer | None,
    cpu_batches: List[Dict[str, Any]],
    valid_batches: List[Dict[str, Any]],
    device: torch.device,
    dense_params: List[torch.nn.Parameter],
    amp_dtype: torch.dtype | None,
    loss_type: str,
    focal_alpha: float,
    focal_gamma: float,
) -> EpochResult:
    steps = 0
    rows = 0
    h2d_wall = 0.0
    last_loss = 0.0

    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start_event.record()
    wall_start = time.perf_counter()

    for inputs, label, batch_rows, batch_h2d_wall in iter_device_batches(cpu_batches, device):
        if variant == "baseline_all_param_clip":
            dense_optimizer.zero_grad()
            if sparse_optimizer is not None:
                sparse_optimizer.zero_grad()
        else:
            dense_optimizer.zero_grad(set_to_none=True)
            if sparse_optimizer is not None:
                sparse_optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp_dtype):
            logits = model(inputs).squeeze(-1)
            if loss_type == "focal":
                loss = sigmoid_focal_loss(
                    logits,
                    label,
                    alpha=focal_alpha,
                    gamma=focal_gamma,
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, label)
        loss.backward()

        if variant == "baseline_all_param_clip":
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, foreach=False
            )
        elif variant != "optimized_no_clip":
            torch.nn.utils.clip_grad_norm_(
                [p for p in dense_params if p.grad is not None and not p.grad.is_sparse],
                max_norm=1.0,
                foreach=True,
            )

        dense_optimizer.step()
        if sparse_optimizer is not None:
            sparse_optimizer.step()

        steps += 1
        rows += batch_rows
        h2d_wall += batch_h2d_wall
        last_loss = float(loss.detach().cpu())

    if device.type == "cuda":
        end_event.record()
        torch.cuda.synchronize(device)
        gpu_sec = start_event.elapsed_time(end_event) / 1000.0
    else:
        gpu_sec = float("nan")
    wall_sec = time.perf_counter() - wall_start
    val_auc, val_logloss, val_wall_sec = evaluate(
        model=model,
        cpu_batches=valid_batches,
        device=device,
        amp_dtype=amp_dtype,
    )

    return EpochResult(
        variant=variant,
        repeat=repeat,
        epoch=epoch,
        steps=steps,
        rows=rows,
        loss=last_loss,
        epoch_wall_sec=wall_sec,
        epoch_gpu_sec=gpu_sec,
        h2d_wall_sec=h2d_wall,
        val_auc=val_auc,
        val_logloss=val_logloss,
        val_wall_sec=val_wall_sec,
    )


def run_variant(
    *,
    variant: str,
    dataset: PCVRParquetDataset,
    cpu_batches: List[Dict[str, Any]],
    valid_batches: List[Dict[str, Any]],
    device: torch.device,
    repeats: int,
    warmup_epochs: int,
    epochs: int,
    seed: int,
    attention_chunk_size: int = 0,
    amp_dtype: torch.dtype | None = None,
    loss_type: str = "bce",
    focal_alpha: float = 0.1,
    focal_gamma: float = 2.0,
    activation_checkpoint_mode: str = "none",
    activation_checkpoint_units: Tuple[str, ...] = (),
) -> List[EpochResult]:
    sparse_embeddings = variant == "sparse_embedding_grads"
    set_seed(seed)
    model = build_model(
        dataset,
        sparse_embeddings=sparse_embeddings,
        device=device,
        attention_chunk_size=attention_chunk_size,
        activation_checkpoint_mode=activation_checkpoint_mode,
        activation_checkpoint_units=activation_checkpoint_units,
    )
    sparse_params = model.get_sparse_params()
    dense_params = model.get_dense_params()
    sparse_optimizer = torch.optim.Adagrad(sparse_params, lr=0.05, weight_decay=0.0)
    dense_optimizer = torch.optim.AdamW(dense_params, lr=1e-4, betas=(0.9, 0.98))

    for i in range(warmup_epochs):
        run_epoch(
            variant=variant,
            repeat=-(i + 1),
            epoch=0,
            model=model,
            dense_optimizer=dense_optimizer,
            sparse_optimizer=sparse_optimizer,
            cpu_batches=cpu_batches,
            valid_batches=valid_batches,
            device=device,
            dense_params=dense_params,
            amp_dtype=amp_dtype,
            loss_type=loss_type,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )

    results = []
    for repeat in range(repeats):
        for epoch in range(1, epochs + 1):
            results.append(
                run_epoch(
                    variant=variant,
                    repeat=repeat,
                    epoch=epoch,
                    model=model,
                    dense_optimizer=dense_optimizer,
                    sparse_optimizer=sparse_optimizer,
                    cpu_batches=cpu_batches,
                    valid_batches=valid_batches,
                    device=device,
                    dense_params=dense_params,
                    amp_dtype=amp_dtype,
                    loss_type=loss_type,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                )
            )

    del model, dense_optimizer, sparse_optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return results


def summarize(results: List[EpochResult]) -> Dict[str, Any]:
    repeats = sorted({r.repeat for r in results})
    epochs = max(r.epoch for r in results)
    repeat_wall = np.array([
        sum(r.epoch_wall_sec for r in results if r.repeat == repeat)
        for repeat in repeats
    ], dtype=np.float64)
    repeat_gpu = np.array([
        sum(r.epoch_gpu_sec for r in results if r.repeat == repeat)
        for repeat in repeats
    ], dtype=np.float64)
    repeat_h2d = np.array([
        sum(r.h2d_wall_sec for r in results if r.repeat == repeat)
        for repeat in repeats
    ], dtype=np.float64)
    final_results = [
        max((r for r in results if r.repeat == repeat), key=lambda r: r.epoch)
        for repeat in repeats
    ]
    auc = np.array([r.val_auc for r in final_results], dtype=np.float64)
    logloss = np.array([r.val_logloss for r in final_results], dtype=np.float64)
    val_wall = np.array([
        sum(r.val_wall_sec for r in results if r.repeat == repeat)
        for repeat in repeats
    ], dtype=np.float64)
    return {
        "variant": results[0].variant,
        "repeats": len(repeats),
        "epochs": epochs,
        "steps": results[0].steps,
        "rows": results[0].rows,
        "total_steps": results[0].steps * epochs,
        "total_rows": results[0].rows * epochs,
        "loss_last": final_results[-1].loss,
        "total_wall_sec_mean": float(repeat_wall.mean()),
        "total_wall_sec_std": float(repeat_wall.std(ddof=0)),
        "total_gpu_sec_mean": float(repeat_gpu.mean()),
        "total_gpu_sec_std": float(repeat_gpu.std(ddof=0)),
        "epoch_wall_sec_mean": float((repeat_wall / epochs).mean()),
        "epoch_wall_sec_std": float((repeat_wall / epochs).std(ddof=0)),
        "epoch_gpu_sec_mean": float((repeat_gpu / epochs).mean()),
        "epoch_gpu_sec_std": float((repeat_gpu / epochs).std(ddof=0)),
        "h2d_wall_sec_mean": float(repeat_h2d.mean()),
        "val_auc_mean": float(auc.mean()),
        "val_auc_std": float(auc.std(ddof=0)),
        "val_auc_last": float(final_results[-1].val_auc),
        "val_logloss_mean": float(logloss.mean()),
        "val_logloss_last": float(final_results[-1].val_logloss),
        "val_wall_sec_mean": float(val_wall.mean()),
    }


def format_report(payload: Dict[str, Any]) -> str:
    lines = [
        f"Device: {payload['device']} ({payload['device_name']})",
        f"Data: {payload['data_dir']} | "
        f"train_batches={payload['num_batches']} train_rows={payload['num_rows']} | "
        f"valid_batches={payload['num_valid_batches']} valid_rows={payload['num_valid_rows']}",
        f"Synthetic repeat: repeat_batch_to_size={payload.get('repeat_batch_to_size', 0)} "
        f"source_train_rows={payload.get('source_num_rows', payload['num_rows'])}",
        f"Precision/Loss: amp_dtype={payload.get('amp_dtype', 'fp32')} "
        f"loss_type={payload.get('loss_type', 'bce')}",
        f"Activation checkpoint: mode={payload.get('activation_checkpoint_mode', 'none')} "
        f"units={payload.get('activation_checkpoint_units', [])}",
    ]
    for summary in payload["summaries"]:
        chunk = int(summary.get("attention_chunk_size", 0))
        chunk_text = f", chunk={chunk}" if chunk > 0 else ""
        lines.append(
            f"{summary['variant']}: "
            f"total_wall={summary['total_wall_sec_mean']:.6f}s "
            f"+/- {summary['total_wall_sec_std']:.6f}s, "
            f"total_gpu={summary['total_gpu_sec_mean']:.6f}s "
            f"+/- {summary['total_gpu_sec_std']:.6f}s, "
            f"per_epoch_wall={summary['epoch_wall_sec_mean']:.6f}s, "
            f"per_epoch_gpu={summary['epoch_gpu_sec_mean']:.6f}s, "
            f"h2d_total={summary['h2d_wall_sec_mean']:.6f}s, "
            f"val_auc={summary['val_auc_mean']:.6f} "
            f"+/- {summary['val_auc_std']:.6f}, "
            f"val_logloss={summary['val_logloss_mean']:.6f}, "
            f"epochs={summary['epochs']}, "
            f"steps_per_epoch={summary['steps']}, rows_per_epoch={summary['rows']}"
            f"{chunk_text}"
        )
        if "wall_speedup_vs_baseline" in summary:
            baseline_auc = payload["summaries"][0]["val_auc_mean"]
            lines.append(
                f"  vs baseline: wall_speedup={summary['wall_speedup_vs_baseline']:.3f}x "
                f"({summary['wall_reduction_pct_vs_baseline']:.2f}% less), "
                f"gpu_speedup={summary['gpu_speedup_vs_baseline']:.3f}x "
                f"({summary['gpu_reduction_pct_vs_baseline']:.2f}% less), "
                f"auc_delta={summary['val_auc_mean'] - baseline_auc:+.6f}"
            )
    return "\n".join(lines)


def write_reports(payload: Dict[str, Any], log_dir: str) -> Tuple[str, str]:
    os.makedirs(log_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(log_dir, f"benchmark_gpu_epoch_{stamp}.txt")
    json_path = os.path.join(log_dir, f"benchmark_gpu_epoch_{stamp}.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(format_report(payload) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return txt_path, json_path


def format_mask_equivalence_report(payload: Dict[str, Any]) -> str:
    lines = [
        f"Device: {payload['device']} ({payload['device_name']})",
        f"Data: {payload['data_dir']} | batch_rows={payload['batch_rows']}",
        f"loss_old={payload['loss_old']:.10f}",
        f"loss_new={payload['loss_new']:.10f}",
        f"loss_abs_diff={payload['loss_abs_diff']:.10e}",
        f"logits_max_abs_diff={payload['logits_max_abs_diff']:.10e}",
        f"logits_mean_abs_diff={payload['logits_mean_abs_diff']:.10e}",
        f"grad_max_abs_diff={payload['grad_max_abs_diff']:.10e}",
        f"grad_mean_abs_diff={payload['grad_mean_abs_diff']:.10e}",
        f"grad_max_rel_diff={payload['grad_max_rel_diff']:.10e}",
        f"compared_grad_tensors={payload['compared_grad_tensors']}",
        f"pass_logits={payload['pass_logits']}",
        f"pass_grads={payload['pass_grads']}",
    ]
    if payload["top_grad_diffs"]:
        lines.append("Top gradient diffs:")
        for item in payload["top_grad_diffs"]:
            lines.append(
                f"  {item['name']}: max_abs={item['max_abs_diff']:.10e}, "
                f"mean_abs={item['mean_abs_diff']:.10e}, "
                f"max_rel={item['max_rel_diff']:.10e}"
            )
    return "\n".join(lines)


def write_mask_equivalence_report(payload: Dict[str, Any], log_dir: str) -> Tuple[str, str]:
    os.makedirs(log_dir, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(log_dir, f"mask_equivalence_{stamp}.txt")
    json_path = os.path.join(log_dir, f"mask_equivalence_{stamp}.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(format_mask_equivalence_report(payload) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return txt_path, json_path


def grad_tensor(grad: torch.Tensor | None) -> torch.Tensor | None:
    if grad is None:
        return None
    if grad.is_sparse:
        return grad.coalesce().to_dense()
    return grad


def main_mask_equivalence(argv: List[str]) -> None:
    args = parse_mask_equivalence_args(argv)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False")

    set_seed(args.seed)
    data_dir = os.path.abspath(args.data_dir)
    schema_path = os.path.abspath(args.schema_path or os.path.join(data_dir, "schema.json"))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dataset, cpu_batches, _ = load_train_valid_batches(
        data_dir=data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        seq_max_lens=parse_seq_max_lens(args.seq_max_lens),
        valid_ratio=0.1,
    )
    if not cpu_batches:
        raise RuntimeError("No training batch available for mask equivalence check")

    set_seed(args.seed)
    old_model = build_model(dataset, sparse_embeddings=False, device=device)
    set_seed(args.seed + 1)
    new_model = build_model(dataset, sparse_embeddings=False, device=device)
    new_model.load_state_dict(old_model.state_dict())
    patch_old_expanded_attention_mask(old_model)
    old_model.eval()
    new_model.eval()

    inputs, label, _ = batch_to_model_input(cpu_batches[0], device)

    old_model.zero_grad(set_to_none=True)
    new_model.zero_grad(set_to_none=True)
    logits_old = old_model(inputs).squeeze(-1)
    logits_new = new_model(inputs).squeeze(-1)
    loss_old = F.binary_cross_entropy_with_logits(logits_old, label)
    loss_new = F.binary_cross_entropy_with_logits(logits_new, label)
    loss_old.backward()
    loss_new.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    logits_abs = (logits_old.detach() - logits_new.detach()).abs()
    top_grad_diffs = []
    grad_abs_sum = 0.0
    grad_numel = 0
    grad_max_abs = 0.0
    grad_max_rel = 0.0
    compared = 0
    for (name_old, p_old), (name_new, p_new) in zip(
        old_model.named_parameters(),
        new_model.named_parameters(),
    ):
        if name_old != name_new:
            raise RuntimeError(f"Parameter order mismatch: {name_old} vs {name_new}")
        g_old = grad_tensor(p_old.grad)
        g_new = grad_tensor(p_new.grad)
        if g_old is None and g_new is None:
            continue
        if g_old is None or g_new is None:
            raise RuntimeError(f"Gradient presence mismatch for {name_old}")
        diff = (g_old.detach() - g_new.detach()).abs()
        scale = torch.maximum(g_old.detach().abs(), g_new.detach().abs()).clamp_min(1e-12)
        rel = diff / scale
        max_abs = float(diff.max().cpu()) if diff.numel() else 0.0
        mean_abs = float(diff.mean().cpu()) if diff.numel() else 0.0
        max_rel = float(rel.max().cpu()) if rel.numel() else 0.0
        grad_max_abs = max(grad_max_abs, max_abs)
        grad_max_rel = max(grad_max_rel, max_rel)
        grad_abs_sum += float(diff.sum().cpu())
        grad_numel += diff.numel()
        compared += 1
        top_grad_diffs.append({
            "name": name_old,
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "max_rel_diff": max_rel,
        })

    top_grad_diffs.sort(key=lambda x: x["max_abs_diff"], reverse=True)
    grad_mean_abs = grad_abs_sum / max(grad_numel, 1)
    payload = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "data_dir": data_dir,
        "batch_rows": int(label.numel()),
        "loss_old": float(loss_old.detach().cpu()),
        "loss_new": float(loss_new.detach().cpu()),
        "loss_abs_diff": abs(float(loss_old.detach().cpu()) - float(loss_new.detach().cpu())),
        "logits_max_abs_diff": float(logits_abs.max().cpu()),
        "logits_mean_abs_diff": float(logits_abs.mean().cpu()),
        "grad_max_abs_diff": grad_max_abs,
        "grad_mean_abs_diff": grad_mean_abs,
        "grad_max_rel_diff": grad_max_rel,
        "compared_grad_tensors": compared,
        "logits_atol": args.logits_atol,
        "grads_atol": args.grads_atol,
        "pass_logits": float(logits_abs.max().cpu()) <= args.logits_atol,
        "pass_grads": grad_max_abs <= args.grads_atol,
        "top_grad_diffs": top_grad_diffs[:10],
    }

    txt_path, json_path = write_mask_equivalence_report(payload, os.path.abspath(args.log_dir))
    if args.json:
        print(json.dumps(payload, indent=2))
        print(f"report_txt={txt_path}", file=sys.stderr)
        print(f"report_json={json_path}", file=sys.stderr)
        return
    print(format_mask_equivalence_report(payload))
    print(f"Report written: {txt_path}")
    print(f"JSON written: {json_path}")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False")

    set_seed(args.seed)
    data_dir = os.path.abspath(args.data_dir)
    schema_path = os.path.abspath(args.schema_path or os.path.join(data_dir, "schema.json"))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    amp_dtype = resolve_amp_dtype(device, args.amp_dtype)
    activation_checkpoint_units = tuple(
        unit.strip()
        for unit in args.activation_checkpoint_units.split(",")
        if unit.strip()
    )

    dataset, cpu_batches, valid_batches = load_train_valid_batches(
        data_dir=data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        seq_max_lens=parse_seq_max_lens(args.seq_max_lens),
        valid_ratio=args.valid_ratio,
    )
    source_num_rows = sum(int(b["label"].numel()) for b in cpu_batches)
    if args.repeat_batch_to_size > 0:
        cpu_batches = [
            repeat_batch_rows(batch, args.repeat_batch_to_size)
            for batch in cpu_batches
        ]

    if args.variant is not None:
        variant_specs = [(args.variant, 0)]
    elif args.disable_dense_clip:
        variant_specs: List[Tuple[str, int]] = [("optimized_no_clip", 0)]
    else:
        variant_specs = [
            ("baseline_all_param_clip", 0),
            ("optimized_dense_clip", 0),
        ]
        for chunk_size in parse_int_list(args.attention_chunk_sizes):
            variant_specs.append((f"optimized_chunk{chunk_size}", chunk_size))
        if args.include_sparse_embedding_grads:
            variant_specs.append(("sparse_embedding_grads", 0))

    all_results: Dict[str, List[EpochResult]] = {}
    summaries = []
    for variant, attention_chunk_size in variant_specs:
        results = run_variant(
            variant=variant,
            dataset=dataset,
            cpu_batches=cpu_batches,
            valid_batches=valid_batches,
            device=device,
            repeats=args.repeats,
            warmup_epochs=args.warmup_epochs,
            epochs=args.epochs,
            seed=args.seed,
            attention_chunk_size=attention_chunk_size,
            amp_dtype=amp_dtype,
            loss_type=args.loss_type,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            activation_checkpoint_mode=args.activation_checkpoint_mode,
            activation_checkpoint_units=activation_checkpoint_units,
        )
        all_results[variant] = results
        summary = summarize(results)
        summary["attention_chunk_size"] = attention_chunk_size
        summaries.append(summary)

    baseline = summaries[0]
    for summary in summaries[1:]:
        summary["wall_speedup_vs_baseline"] = (
            baseline["epoch_wall_sec_mean"] / summary["epoch_wall_sec_mean"]
        )
        summary["wall_reduction_pct_vs_baseline"] = (
            1.0 - summary["epoch_wall_sec_mean"] / baseline["epoch_wall_sec_mean"]
        ) * 100.0
        summary["gpu_speedup_vs_baseline"] = (
            baseline["epoch_gpu_sec_mean"] / summary["epoch_gpu_sec_mean"]
        )
        summary["gpu_reduction_pct_vs_baseline"] = (
            1.0 - summary["epoch_gpu_sec_mean"] / baseline["epoch_gpu_sec_mean"]
        ) * 100.0
        summary["total_wall_speedup_vs_baseline"] = (
            baseline["total_wall_sec_mean"] / summary["total_wall_sec_mean"]
        )
        summary["total_wall_reduction_pct_vs_baseline"] = (
            1.0 - summary["total_wall_sec_mean"] / baseline["total_wall_sec_mean"]
        ) * 100.0
        summary["total_gpu_speedup_vs_baseline"] = (
            baseline["total_gpu_sec_mean"] / summary["total_gpu_sec_mean"]
        )
        summary["total_gpu_reduction_pct_vs_baseline"] = (
            1.0 - summary["total_gpu_sec_mean"] / baseline["total_gpu_sec_mean"]
        ) * 100.0

    payload = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "data_dir": data_dir,
        "num_batches": len(cpu_batches),
        "num_rows": sum(int(b["label"].numel()) for b in cpu_batches),
        "source_num_rows": source_num_rows,
        "repeat_batch_to_size": args.repeat_batch_to_size,
        "num_valid_batches": len(valid_batches),
        "num_valid_rows": sum(int(b["label"].numel()) for b in valid_batches),
        "attention_chunk_sizes": parse_int_list(args.attention_chunk_sizes),
        "amp_dtype": args.amp_dtype,
        "loss_type": args.loss_type,
        "focal_alpha": args.focal_alpha,
        "focal_gamma": args.focal_gamma,
        "activation_checkpoint_mode": args.activation_checkpoint_mode,
        "activation_checkpoint_units": activation_checkpoint_units,
        "summaries": summaries,
    }

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
    if len(sys.argv) > 1 and sys.argv[1] == "mask_equivalence":
        main_mask_equivalence(sys.argv[2:])
    else:
        main()
