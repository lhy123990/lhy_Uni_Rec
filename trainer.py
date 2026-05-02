"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import glob
import shutil
import logging
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        amp_dtype: str = 'auto',
        disable_amp: bool = False,
        precision_log_every_n_steps: int = 100,
        topk_best: int = 3,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path
        self.grad_clip_params: Optional[list[nn.Parameter]] = None

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            self.grad_clip_params = dense_params
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98)
            )
        else:
            self.sparse_optimizer = None
            self.grad_clip_params = list(model.parameters())
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.device_type: str = torch.device(device).type
        self.precision_log_every_n_steps: int = max(
            0, int(precision_log_every_n_steps)
        )
        self.amp_enabled, self.amp_dtype = self._resolve_amp_config(
            amp_dtype=amp_dtype,
            disable_amp=disable_amp,
        )
        scaler_enabled = self.amp_enabled and self.amp_dtype == torch.float16
        try:
            self.scaler = torch.amp.GradScaler('cuda', enabled=scaler_enabled)
        except TypeError:  # pragma: no cover - compatibility with older torch
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        if sparse_weight_decay != 0.0 and getattr(model, 'sparse_embeddings', False):
            raise ValueError(
                "sparse_weight_decay must be 0.0 when sparse embedding "
                "gradients are enabled; torch.optim.Adagrad does not support "
                "weight decay for sparse gradients."
            )

        self.topk_best: int = int(topk_best)
        if self.topk_best < 1:
            raise ValueError(f"topk_best must be >= 1, got {self.topk_best}")
        self._topk_ckpts: list[tuple[float, int, str]] = []
        self._load_existing_topk_checkpoints()

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}")
        logging.info(
            "AMP config: enabled=%s, requested=%s, resolved=%s, "
            "grad_scaler=%s, precision_log_every_n_steps=%d",
            self.amp_enabled,
            amp_dtype,
            self._amp_dtype_name(self.amp_dtype),
            self.scaler.is_enabled(),
            self.precision_log_every_n_steps,
        )
        logging.info("Checkpoint retention: keep top-%d checkpoints by val AUC", self.topk_best)

    @staticmethod
    def _amp_dtype_name(dtype: Optional[torch.dtype]) -> str:
        if dtype is torch.bfloat16:
            return 'bf16'
        if dtype is torch.float16:
            return 'fp16'
        return 'fp32'

    def _resolve_amp_config(
        self,
        amp_dtype: str,
        disable_amp: bool,
    ) -> Tuple[bool, Optional[torch.dtype]]:
        """Resolve user AMP flags into an autocast dtype.

        CUDA bf16 is preferred for H20-class online training. On devices where
        bf16 is unavailable, auto mode falls back to fp16 and enables
        GradScaler for training.
        """
        amp_dtype = amp_dtype.lower()
        if disable_amp or amp_dtype == 'fp32' or self.device_type != 'cuda':
            return False, None

        if amp_dtype == 'auto':
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return True, torch.bfloat16
            return True, torch.float16

        if amp_dtype == 'bf16':
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return True, torch.bfloat16
            logging.warning(
                "Requested bf16 AMP, but CUDA bf16 is not reported as "
                "supported. Falling back to fp16 AMP."
            )
            return True, torch.float16

        if amp_dtype == 'fp16':
            return True, torch.float16

        raise ValueError(
            "amp_dtype must be one of auto, bf16, fp16, or fp32; "
            f"got {amp_dtype!r}"
        )

    def _autocast_context(self):
        if not self.amp_enabled or self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.amp_dtype)

    @staticmethod
    def _tensor_mean_var(tensor: torch.Tensor) -> Tuple[float, float]:
        if tensor.numel() == 0:
            return 0.0, 0.0
        t = tensor.detach().float()
        mean = float(t.mean().item())
        var = float(t.var(unbiased=False).item()) if t.numel() > 1 else 0.0
        return mean, var

    @staticmethod
    def _running_mean_var(
        total_sum: float,
        total_sumsq: float,
        total_count: int,
    ) -> Tuple[float, float]:
        if total_count <= 0:
            return 0.0, 0.0
        mean = total_sum / total_count
        var = max(total_sumsq / total_count - mean * mean, 0.0)
        return float(mean), float(var)

    def _should_log_precision(
        self,
        global_step: Optional[int],
        force: bool = False,
    ) -> bool:
        if self.writer is None:
            return False
        if force:
            return True
        if self.precision_log_every_n_steps <= 0 or global_step is None:
            return False
        return global_step <= 1 or global_step % self.precision_log_every_n_steps == 0

    def _write_tensor_stats(
        self,
        name: str,
        tensor: torch.Tensor,
        global_step: int,
    ) -> None:
        if self.writer is None:
            return
        mean, var = self._tensor_mean_var(tensor)
        self.writer.add_scalar(f'Precision/{name}_mean', mean, global_step)
        self.writer.add_scalar(f'Precision/{name}_var', var, global_step)

    def _write_running_stats(
        self,
        name: str,
        total_sum: float,
        total_sumsq: float,
        total_count: int,
        global_step: int,
    ) -> None:
        if self.writer is None:
            return
        mean, var = self._running_mean_var(total_sum, total_sumsq, total_count)
        self.writer.add_scalar(f'Precision/{name}_mean', mean, global_step)
        self.writer.add_scalar(f'Precision/{name}_var', var, global_step)

    def _clip_dense_grad_norm(self, max_norm: float) -> torch.Tensor:
        """Clip only dense gradients.

        Embedding tables are optimized by the sparse optimizer group and can
        optionally produce sparse gradients. Keeping clipping on the dense
        optimizer group avoids the slow all-embedding norm pass while still
        covering the Transformer, FFNs, projections, and classifier.
        """
        dense_grad_params = [
            p for p in (self.grad_clip_params or [])
            if p.grad is not None and not p.grad.is_sparse
        ]
        if not dense_grad_params:
            return torch.tensor(0.0, device=self.device)
        return torch.nn.utils.clip_grad_norm_(
            dense_grad_params, max_norm=max_norm, foreach=True
        )

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """Build a checkpoint sub-directory name such as
        ``global_step2500.layer=2.head=4.hidden=64[.best_model]``.
        """
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists up to three files, all overwritten on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                # Override the stored path to a filename relative to ckpt_dir;
                # infer.py already falls back to `<ckpt_dir>/<basename>` when
                # the recorded path is not absolute, which keeps the ckpt
                # portable across hosts.
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir.

        Args:
            global_step: current global step used to name the directory.
            is_best: whether this is a new-best checkpoint.
            skip_model_file: if True, skip writing ``model.pt`` (because the
                caller, e.g. EarlyStopping, has already persisted it to the
                same path). Sidecar files are still (re)written.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _format_auc_for_dir(self, val_auc: float) -> str:
        return f"{val_auc:.6f}"

    def _build_topk_dir_name(self, global_step: int, val_auc: float) -> str:
        base = self._build_step_dir_name(global_step, is_best=False)
        return f"{base}.topk_auc={self._format_auc_for_dir(val_auc)}"

    def _load_existing_topk_checkpoints(self) -> None:
        pattern = os.path.join(self.save_dir, "global_step*.topk_auc=*")
        found: list[tuple[float, int, str]] = []
        for ckpt_dir in glob.glob(pattern):
            name = os.path.basename(ckpt_dir)
            try:
                step_str = name.split("global_step", 1)[1].split(".", 1)[0]
                step = int(step_str)
                auc_part = [p for p in name.split(".") if p.startswith("topk_auc=")][0]
                auc = float(auc_part.split("=", 1)[1])
            except Exception:
                continue
            if not os.path.exists(os.path.join(ckpt_dir, "model.pt")):
                continue
            found.append((auc, step, ckpt_dir))

        found.sort(key=lambda x: (-x[0], -x[1]))
        self._topk_ckpts = found[: self.topk_best]
        for auc, step, ckpt_dir in found[self.topk_best :]:
            try:
                shutil.rmtree(ckpt_dir)
                logging.info(
                    "Pruned extra topk checkpoint: step=%d auc=%.6f dir=%s",
                    step, auc, ckpt_dir,
                )
            except Exception as e:
                logging.warning("Failed to prune %s: %s", ckpt_dir, e)

    def _maybe_save_topk_checkpoint(self, global_step: int, val_auc: float) -> None:
        if not np.isfinite(val_auc):
            return

        for _, step, _ in self._topk_ckpts:
            if step == global_step:
                return

        qualifies = len(self._topk_ckpts) < self.topk_best
        if not qualifies:
            worst_auc = min(a for a, _, _ in self._topk_ckpts)
            qualifies = val_auc > worst_auc
        if not qualifies:
            return

        ckpt_dir = self._save_step_checkpoint(
            global_step,
            is_best=False,
            skip_model_file=False,
            dir_name=self._build_topk_dir_name(global_step, val_auc),
        )
        self._topk_ckpts.append((val_auc, global_step, ckpt_dir))
        self._topk_ckpts.sort(key=lambda x: (-x[0], -x[1]))

        while len(self._topk_ckpts) > self.topk_best:
            auc, step, stale_dir = self._topk_ckpts.pop(-1)
            try:
                shutil.rmtree(stale_dir)
                logging.info(
                    "Removed stale (outside top-%d) ckpt: step=%d auc=%.6f dir=%s",
                    self.topk_best, step, auc, stale_dir,
                )
            except Exception as e:
                logging.warning("Failed to remove stale checkpoint dir %s: %s", stale_dir, e)

        logging.info(
            "Topk checkpoints now: %s",
            ", ".join([f"(auc={a:.6f}, step={s})" for a, s, _ in self._topk_ckpts]),
        )

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Persist validation state and keep the top-K AUC checkpoints."""
        self.early_stopping(val_auc, self.model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
        })
        self._maybe_save_topk_checkpoint(total_step, val_auc)

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, triggers EarlyStopping and the periodic sparse
        re-initialization strategy.
        """
        print("Start training (PCVRHyFormer)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True)
            loss_sum = 0.0

            for step, batch in train_pbar:
                next_step = total_step + 1
                loss = self._train_step(batch, global_step=next_step)
                total_step = next_step
                loss_sum += loss

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)

                train_pbar.set_postfix({"loss": f"{loss:.4f}"})

                # Step-level validation (only when eval_every_n_steps > 0).
                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    val_auc, val_logloss = self.evaluate(
                        epoch=epoch, global_step=total_step)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info(f"Step {total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                    self._handle_validation_result(total_step, val_auc, val_logloss)

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return

            logging.info(f"Epoch {epoch}, Average Loss: {loss_sum / len(self.train_loader)}")

            val_auc, val_logloss = self.evaluate(
                epoch=epoch, global_step=total_step)
            self.model.train()
            torch.cuda.empty_cache()

            logging.info(f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

            self._handle_validation_result(total_step, val_auc, val_logloss)

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # After the configured epoch, reinitialize high-cardinality sparse
            # params (Embeddings) as a form of cold restart to reduce overfit.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                # Snapshot Adagrad state per parameter via data_ptr, so state
                # of low-cardinality embeddings can be preserved across rebuild.
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
        )

    def _train_step(self, batch: Dict[str, Any], global_step: int) -> float:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        self.dense_optimizer.zero_grad(set_to_none=True)
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad(set_to_none=True)

        model_input = self._make_model_input(device_batch)
        with self._autocast_context():
            logits = self.model(model_input)  # (B, 1)
            logits = logits.squeeze(-1)  # (B,)

            if self.loss_type == 'focal':
                loss = sigmoid_focal_loss(
                    logits, label, alpha=self.focal_alpha,
                    gamma=self.focal_gamma)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, label)

        log_precision = self._should_log_precision(global_step)
        if log_precision:
            self._write_tensor_stats('train_logits', logits, global_step)

        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.scaler.unscale_(self.sparse_optimizer)
            self._clip_dense_grad_norm(max_norm=1.0)
            self.scaler.step(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.scaler.step(self.sparse_optimizer)
            self.scaler.update()
            if log_precision and self.writer:
                self.writer.add_scalar(
                    'Precision/loss_scaler',
                    float(self.scaler.get_scale()),
                    global_step,
                )
        else:
            loss.backward()
            self._clip_dense_grad_norm(max_norm=1.0)
            self.dense_optimizer.step()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.step()

        return loss.item()

    def evaluate(
        self,
        epoch: Optional[int] = None,
        global_step: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Run validation over ``self.valid_loader`` and return ``(AUC, logloss)``.

        NaN predictions (which can arise from exploding gradients) are filtered
        out before computing both metrics.
        """
        print("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        all_logits_list = []
        all_labels_list = []
        embedding_sum = 0.0
        embedding_sumsq = 0.0
        embedding_count = 0

        with torch.no_grad():
            for step, batch in pbar:
                logits, labels, embeddings = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())
                embeddings_f = embeddings.detach().float()
                embedding_sum += float(embeddings_f.sum().item())
                embedding_sumsq += float((embeddings_f * embeddings_f).sum().item())
                embedding_count += int(embeddings_f.numel())

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()
        log_step = int(global_step if global_step is not None else epoch)
        if self._should_log_precision(log_step, force=True):
            self._write_tensor_stats('valid_logits', all_logits, log_step)
            self._write_tensor_stats('predict_logits', all_logits, log_step)
            self._write_running_stats(
                'predict_embedding',
                embedding_sum,
                embedding_sumsq,
                embedding_count,
                log_step,
            )

        # Binary AUC via sklearn.
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        # Filter NaN predictions (may appear if gradients explode).
        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        # Binary logloss (same NaN filtering).
        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        return auc, logloss

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a single validation step and return logits, labels, embeddings."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        with self._autocast_context():
            logits, embeddings = self.model.predict(model_input)  # (B, 1), (B, D)
        logits = logits.squeeze(-1)  # (B,)

        return logits.float(), label, embeddings
