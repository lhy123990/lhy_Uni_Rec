# PCVRHyFormer 训练优化记录

本文记录当前围绕 `model.py`、`trainer.py`、`train.py` 和
`benchmark_gpu_epoch.py` 做过的 GPU profiler、优化尝试与结论，供后续继续优化。

测试数据使用 `data_sample_1000/`，模型参数以 `run.sh` 的默认配置为基准：

- `ns_tokenizer_type=rankmixer`
- `user_ns_tokens=5`
- `item_ns_tokens=2`
- `num_queries=2`
- `emb_skip_threshold=1000000`
- `seq_max_lens=seq_a:256,seq_b:256,seq_c:512,seq_d:512`

推荐 benchmark 入口：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n taac python benchmark_gpu_epoch.py \
  --data_dir data_sample_1000 \
  --device cuda:0 \
  --epochs 5 \
  --repeats 3 \
  --warmup_epochs 1 \
  --log_dir logs
```

报告会写入 `logs/`，包括文本版和 JSON 版。

## 1. 初始瓶颈

最初的 GPU profiler 显示主要热点如下：

- attention backward 是最大 CUDA 热点，`aten::_efficient_attention_backward`
  约为 `136 ms / 3 steps`。
- attention forward 已经走 PyTorch efficient SDPA/CUTLASS 路径，
  `aten::_efficient_attention_forward` 约为 `49 ms / 3 steps`。
- 原始全参数 `clip_grad_norm_(foreach=False)` 明显偏慢，约为
  `67 ms / 3 steps`。
- dense projection、FFN、QKV/O/gate projection 对应的 `addmm/mm` 占比也较高。
- embedding backward 和 Adagrad 也可见：
  `embedding_dense_backward` 约为 `33 ms / 3 steps`，
  `Adagrad.step` 约为 `29 ms / 3 steps`。

`data_sample_1000/` 上的序列长度分布：

| Domain | 配置长度 | 平均长度 | 平均 padding | 满长比例 |
| --- | ---: | ---: | ---: | ---: |
| `seq_a` | 256 | 219 | 14% | 72% |
| `seq_b` | 256 | 198 | 23% | 63% |
| `seq_c` | 512 | 322 | 37% | 27% |
| `seq_d` | 512 | 419 | 18% | 72% |

每个 batch 的 max length 都等于配置上限，因此简单按 batch 最大真实长度裁剪，
在这个样本集上基本没有收益。

## 2. 已应用到训练代码的优化

### 2.1 Broadcast SDPA Mask

已应用到 `model.py` 的 `RoPEMultiheadAttention.forward`。

旧逻辑会把 padding mask 显式 expand 成 `[B, H, Lq, Lk]`：

```python
sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)
sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)
```

当前逻辑保留可 broadcast 的小形状 `[B, 1, 1, Lk]`：

```python
sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)
```

`attn_mask` 也从显式 expand 改为 `[1, 1, Lq, Lk]`：

```python
bool_attn = (attn_mask == 0)
bool_attn = bool_attn.unsqueeze(0).unsqueeze(0)
```

这个修改的目标是减少显式传给 SDPA 的大 mask 形状，同时保持 attention 语义不变。

等价性验证子命令：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n taac python benchmark_gpu_epoch.py mask_equivalence \
  --data_dir data_sample_1000 \
  --device cuda:0 \
  --batch_size 256 \
  --log_dir logs
```

代表性结果：

```text
loss_abs_diff=0.0000000000e+00
logits_max_abs_diff=0.0000000000e+00
logits_mean_abs_diff=0.0000000000e+00
grad_max_abs_diff=2.9802322388e-08
grad_mean_abs_diff=8.6551264237e-13
pass_logits=True
pass_grads=True
```

结论：old expanded mask 和 new broadcast mask 在 logits 上完全一致，
gradient 只有 fp32 噪声级差异。因此该修改可以作为语义等价优化保留。

### 2.2 Dense-Only Gradient Clipping

已应用到 `trainer.py`。

旧训练步对所有参数做全局裁剪：

```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)
```

当前训练器缓存 dense 参数组，只对非 embedding dense 参数裁剪，并使用 `foreach=True`：

```python
self._clip_dense_grad_norm(max_norm=1.0)
```

这样可以跳过大规模 embedding 参数的 norm pass，同时仍覆盖 Transformer、FFN、
projection 和 classifier 等 dense 参数。

代表性 benchmark 结果：

```text
baseline_all_param_clip: per_epoch_wall=1.465858s, val_auc=0.656250
optimized_dense_clip:   per_epoch_wall=1.398854s, val_auc=0.630997
```

说明：速度收益稳定存在，但 `data_sample_1000/` 的验证集只有 100 行，
AUC 波动较大，不能只依赖该样本集判断长期效果。

## 3. Benchmark 工具

新增 `benchmark_gpu_epoch.py`。

主要能力：

- 比较旧的全参数裁剪、当前 dense-only 裁剪、可选 chunked attention、
  可选 sparse embedding gradients，以及完全关闭 clipping。
- 按 row group 切分 train/valid；在 `data_sample_1000/` 上为
  900 行 train、100 行 valid。
- 支持 `--epochs`，默认 `5`。
- 输出 wall time、CUDA event time、AUC、logloss。
- 报告自动写入 `logs/`。
- 提供 `mask_equivalence` 子命令，用于验证 old/new mask 的 logits 和 gradient 差异。

常用命令：

```bash
# 默认：baseline、dense clip、chunk256。
CUDA_VISIBLE_DEVICES=0 conda run -n taac python benchmark_gpu_epoch.py \
  --data_dir data_sample_1000 \
  --device cuda:0 \
  --epochs 5 \
  --repeats 3 \
  --warmup_epochs 1 \
  --log_dir logs

# 关闭 chunk 变体，只比较 baseline 和 dense clip。
CUDA_VISIBLE_DEVICES=0 conda run -n taac python benchmark_gpu_epoch.py \
  --data_dir data_sample_1000 \
  --device cuda:0 \
  --epochs 5 \
  --repeats 3 \
  --warmup_epochs 1 \
  --attention_chunk_sizes '' \
  --log_dir logs

# 完全关闭 gradient clipping。
CUDA_VISIBLE_DEVICES=0 conda run -n taac python benchmark_gpu_epoch.py \
  --data_dir data_sample_1000 \
  --device cuda:0 \
  --epochs 5 \
  --repeats 3 \
  --warmup_epochs 1 \
  --attention_chunk_sizes '' \
  --disable_dense_clip \
  --log_dir logs
```

## 4. 已尝试但未默认启用的方向

### 4.1 Sparse Embedding Gradients

在 `model.py` 和 `train.py` 中加入了 opt-in 开关：

```bash
--sparse_embedding_grads
```

该开关会给 embedding table 传入 `sparse=True`。实验中它确实降低了
embedding backward：

```text
embedding_dense_backward: 约 33.2 ms / 3 steps
embedding_sparse_backward: 约 9.3 ms / 3 steps
```

但 PyTorch `Adagrad` 对大量小 sparse gradient 做 `_coalesce` 的开销很高：

```text
Adagrad.step: 约 29.4 ms / 3 steps -> 约 263 ms / 3 steps
```

端到端 epoch 变慢，因此当前默认仍使用 dense embedding gradients。

结论：只有在更换 optimizer/update kernel，或生产数据中稀疏访问收益明显更大时，
才值得重新评估该路径。

### 4.2 Chunked Local Self-Attention

`benchmark_gpu_epoch.py` 支持实验参数：

```bash
--attention_chunk_sizes 256
```

它只在 benchmark 中 monkey-patch Transformer self-attention，把 `L=512`
拆成两个局部 `L=256` attention。该方法会阻断跨 chunk attention，
**不是语义等价变换**。

代表性结果：

```text
baseline_all_param_clip: per_epoch_wall=1.537451s, val_auc=0.656250
optimized_dense_clip:   per_epoch_wall=1.441592s, val_auc=0.630997
optimized_chunk256:     per_epoch_wall=1.241031s, val_auc=0.644571
```

结论：chunking 对速度很有吸引力，但会改变模型表达能力。若继续推进，
应做成显式模型选项，并在更大的验证集上比较。

### 4.3 No Gradient Clipping

benchmark-only 开关：

```bash
--disable_dense_clip
```

代表性结果：

```text
optimized_no_clip: per_epoch_wall=1.376281s, val_auc=0.631629
```

相对 dense-only clipping 只快一点，但移除了训练稳定性保护。当前不建议默认关闭。

## 5. Mask 方向的进一步建议

当前已经完成低风险的 broadcast mask 修改。后续如果继续围绕 mask 优化，
建议按以下顺序：

1. **Full-length/no-mask 分流**：
   满长样本直接走 no-mask SDPA，非满长样本继续走 padding mask。
   `seq_d` 满长比例约 72%，可能最有收益。
2. **Length bucketing**：
   让相近长度样本组成 batch，尤其针对 `seq_c`。`seq_c` padding 约 37%，
   但当前每个 batch 都包含满长样本，导致 batch-level crop 没收益。
3. **Varlen/unpad attention**：
   理论上最能利用 padding 稀疏性，但工程成本最高。PyTorch 原生 SDPA
   不一定直接提供理想的 varlen 路径，可能需要 FlashAttention varlen API
   或自定义 packing kernel。

## 6. 注意事项

- `data_sample_1000/` 的 valid 只有 100 行，AUC 很容易波动。
  速度优化可以用该样本集快速筛选，但质量结论需要更大的验证集。
- AUC 变化不能直接归因于 mask shape；`mask_equivalence` 已证明 broadcast mask
  和 expanded mask 在同一权重、同一 batch 下 logits 完全一致。
- logs 中的历史报告仅用于本地追踪，不应作为可复现实验的唯一来源；
  可复现入口是 `benchmark_gpu_epoch.py` 和本文档中的相对路径命令。
