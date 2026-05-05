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

## 7. Talking-Head Cross-Attention Triton Kernel 设计

线上探测日志见 `docs/debug.md`。当前已实现 v2：Triton fused forward 已整合
seeded dropout，custom autograd backward 也已切到 Triton kernel；不满足固定形状时
仍回退到 PyTorch 参考实现。

### 7.1 线上输入特征

线上 batch 使用 `batch_size=768`，模型配置与 `run.sh` 的 RankMixer 路径一致：

```text
d_model=64, num_heads=4, head_dim=16, num_queries=2
num_hyformer_blocks=2, seq_domains=seq_a/seq_b/seq_c/seq_d
amp_dtype=bf16, use_rope=False, mask=key_padding
```

8 次 talking-head cross-attention 调用可以聚合成两类固定形状：

| domains | calls | B | Lq | Lk | H | Dh | dtype | valid_len mean | rough temp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| seq_a, seq_b | 4 | 768 | 2 | 256 | 4 | 16 | bf16 | 212.52 | 48.75 MiB |
| seq_c, seq_d | 4 | 768 | 2 | 512 | 4 | 16 | bf16 | 416.05 | 96.75 MiB |

单次调用的 Q/K/V 均为投影后的 transposed view：

```text
Q: (B, H, Lq, Dh), stride=[128, 16, 64, 1]
K/V Lk=256: (B, H, Lk, Dh), stride=[16384, 16, 64, 1]
K/V Lk=512: (B, H, Lk, Dh), stride=[32768, 16, 64, 1]
```

最后一维连续，适合 Triton 按 `Dh=16` 向量化加载。`seq_d` 大部分样本满长
(`p25/p50/p75=512`)，但仍存在 `valid_len=0` 的全 padding 行；kernel 必须显式把
全 padding 输出置零，不能依赖 softmax 后再 `nan_to_num`。

### 7.2 当前分支的代价

当前 `model.py` 中 talking-head 分支会依次执行：

```text
QK^T matmul
logits head-mix einsum
masked_fill
softmax
nan_to_num
prob head-mix einsum
dropout
prob @ V matmul
```

在 `B=768` 的线上形状下，一次 forward 的 8 个 cross-attention 调用会产生约
`145.5 MiB` 的 logits/probs 临时张量估算量，并触发多次小矩阵 kernel launch。
由于 `Lq=2,H=4,Dh=16` 都很小，通用 matmul/einsum 很难把 launch 开销和中间
tensor 写回成本摊平。

### 7.3 Kernel 目标

优先做一个专用 fused forward kernel，覆盖当前线上主路径：

```text
H=4, Dh=16, Lq=2, Lk in {256, 512}, dtype=bf16
mask 仅支持 key_padding_mask，RoPE 在进入 kernel 前已处理
th_logits/th_probs 按一般 dense 4x4 矩阵处理，不能假设 identity
```

Python wrapper 在不满足上述条件时回退到当前 PyTorch 实现。这样可以先把风险限制在
线上已观测到的稳定形状内，后续再泛化。

### 7.4 Forward Kernel 方案

Triton grid 采用一个 program 处理一个 `(batch, query)`，也就是
`grid=(B, Lq)`。每个 program 一次性处理全部 `H=4` heads 和完整 K 维：

```text
BLOCK_N = 256 或 512
load Q[h, d]              -> (4, 16)
load K[h, n, d]           -> (4, BLOCK_N, 16)
logits_h[n] = dot(Q_h, K_h[n]) * rsqrt(16)
mixed_logits_u[n] = sum_h logits_h[n] * th_logits[h, u]
mask padding positions; if no valid key, store zero output
softmax mixed_logits_u over n in fp32
prob_g[n] = sum_u softmax_u[n] * th_probs[u, g]
out_g[d] = sum_n prob_g[n] * V[g, n, d]
store out as (B, H, Lq, Dh)
```

数值策略：

- dot、head mixing、softmax 和 V accumulate 使用 fp32，store cast 回 bf16。
- softmax 使用每个 mixed head 独立的 max/sum，和当前 `dim=-1` 语义一致。
- padding mask 中 `True` 表示 padding；全 padding 行直接输出 0。
- `th_logits` 和 `th_probs` 以 fp32 读取，支持训练后变成非单位矩阵。

这个 kernel 可以消除 logits/probs/head-mix 中间张量写回，把单次调用的显式输出
限制在 `(B,H,Lq,Dh)`，约 `0.1875 MiB`。相对当前分支，主要收益来自减少
临时显存流量和 kernel launch 数量，而不是减少理论 FLOPs。

### 7.5 Dropout 与 backward

当前训练态 `dropout_rate=0.01`，dropout 位于第二次 head-mix 之后、乘 V 之前。
实现参考 Triton low-memory dropout 教程：forward 不保存完整 dropout mask，只保存一个
seed；backward 依据同一个 seed 和 `(B,H,Lq,Lk)` 线性 offset 再生成 keep mask。

1. **v1: fused forward + seeded dropout，已实现**  
   forward 在 Triton 内完成 attention、talking-head mixing、dropout 和 `@V`。若需要
   autograd，custom `Function.backward` 会用保存的 seed 临时重建 keep mask，并通过
   PyTorch 参考公式计算 `Q/K/V/th_logits/th_probs` 梯度。
2. **v2: custom autograd training kernel，已实现**  
   backward 在 Triton 内重算 logits、softmax、head-mix 和 deterministic dropout，
   直接计算 `dQ/dK/dV/d_th_logits/d_th_probs`，避免 backward 阶段物化完整
   `probs/keep_mask`。`dK/dV` 和 talking-head 参数梯度用 fp32 atomic accumulate，
   返回给 `Q/K/V` 的梯度 cast 回输入 dtype，参数梯度保留参数 dtype。

当前 v2 已覆盖训练态主路径。fixed fast path 条件仍是 `H=4,Dh=16,Lq=2,Lk in
{256,512}`、`key_padding_mask`、`attn_mask=None`；其它输入继续走 PyTorch fallback。

### 7.6 验收与 benchmark

实现后先用独立等价测试覆盖：

- `Lk=256/512`、`B` 包含小 batch 与 `768`。
- `th_logits/th_probs` 分别测试 identity 和随机 dense 4x4。
- `valid_len=0`、短序列、满长序列。
- fp32 参考路径和 bf16 autocast 路径；bf16 输出建议以 `atol=3e-3, rtol=3e-3`
  作为初始阈值，再根据实测收紧。

benchmark 建议单独统计 talking-head 分支耗时和端到端 train step：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n taac python probe_talking_head_inputs.py \
  --data_dir data_sample_1000 \
  --device cuda:0 \
  --batch_size 256 \
  --max_batches 1
```

线上 benchmark 使用同样的模型参数和 `batch_size=768`。本地 `data_sample_1000`
只有 100 行，真实端到端 batch 需要以上线日志为准。

当前实现入口：

- `talking_head_triton.py`：固定形状 Triton forward kernel、seeded dropout 和
  custom autograd wrapper。
- `model.py`：在 talking-head 分支中做 fast-path 判断和 PyTorch fallback。
- 可用 `TALKING_HEAD_TRITON=0` 环境变量关闭 fast path 做 A/B 对比。

### 7.7 Backward 实现验收记录

2026-05-05 在 RTX 3090、torch `2.7.1+cu126`、bf16 mixed precision 下验证：

| case | result |
| --- | --- |
| `python -m py_compile model.py talking_head_triton.py` | pass |
| synthetic gradcheck, `B=4,Lk=256,p=0.11` | output max diff `0.0`；`dQ/dK/dV/d_th_logits/d_th_probs` max diff `[0.0, 3.05e-05, 1.22e-04, 9.54e-07, 4.77e-07]` |
| synthetic gradcheck, `B=4,Lk=512,p=0.11` | output max diff `0.0`；`dQ/dK/dV/d_th_logits/d_th_probs` max diff `[0.0, 1.22e-04, 6.10e-05, 1.49e-06, 5.07e-07]` |
| full-model smoke, requested `batch_size=768` | local sample actual `B=100`，loss `0.67348`，logits dtype `torch.bfloat16`，`grad_params=436` |

核心算子 `B=768, H=4, Lq=2, Dh=16, dropout_p=0.01` 的 forward+backward 合成
benchmark 如下；首轮 Triton JIT 编译不计入稳定结果：

| Lk | Triton fwd+bwd | PyTorch reference | speedup |
| ---: | ---: | ---: | ---: |
| 256 | `2.7335 ms` | `4.9458 ms` | `1.81x` |
| 512 | `5.2886 ms` | `9.8337 ms` | `1.86x` |

### 7.8 真实训练 10 epoch A/B

为贴近 `run.sh`，benchmark 已补齐当前 `ModelInput` 时间特征字段，并支持 bf16
autocast 与 focal loss。A/B 只切换 `TALKING_HEAD_TRITON`，模型、seed、batch、
loss、验证集划分保持一致：

```bash
CUDA_VISIBLE_DEVICES=0 TALKING_HEAD_TRITON=0 conda run -n taac python benchmark_gpu_efficiency.py \
  --data_dir data_sample_1000 \
  --schema_path data_sample_1000/schema.json \
  --device cuda:0 \
  --batch_size 768 \
  --epochs 10 \
  --repeats 1 \
  --warmup_epochs 0 \
  --variant optimized_dense_clip \
  --attention_chunk_sizes '' \
  --amp_dtype bf16 \
  --loss_type focal \
  --focal_alpha 0.5 \
  --focal_gamma 2.0 \
  --log_dir logs \
  --json
```

本地只有 `data_sample_1000`，因此实际训练集为 `900` 行、验证集为 `100` 行，
每 epoch `9` steps，10 epoch 共 `90` train steps。结果如下：

| setting | total train wall | train wall / epoch | total train GPU | AUC | logloss | report |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Triton off | `17.6116 s` | `1.7612 s` | `17.6096 s` | `0.638258` | `0.359549` | `logs/benchmark_gpu_epoch_20260505_173519.json` |
| Triton on | `17.7010 s` | `1.7701 s` | `17.6990 s` | `0.649148` | `0.363688` | `logs/benchmark_gpu_epoch_20260505_173557.json` |

端到端训练 wall 在这个小样本上为 `0.995x`，即慢 `0.51%`；AUC 差值为
`+0.010890`。由于 dropout 随机源从 PyTorch dropout 切到 Triton seeded dropout，
训练轨迹不会逐 bit 相同；在 100 行验证集上这个 AUC 差值只能作为 smoke 指标，
不应解读为质量收益。

如果扣除 benchmark 记录的 host-to-device 搬运 wall，模型训练计算段为：

| setting | train wall - H2D | compute speedup |
| --- | ---: | ---: |
| Triton off | `16.2408 s` | baseline |
| Triton on | `15.9703 s` | `1.017x` |

结论：单个 talking-head fwd+bwd core 有约 `1.8x` 加速，但在当前全模型 10 epoch
sample 训练中，这条路径只占总训练时间的一小段，端到端收益被数据搬运、embedding、
self-attention、FFN、梯度裁剪和验证开销稀释。线上 batch 更稳定且数据量更大时，
建议用同一命令在真实数据上复测；`TALKING_HEAD_TRITON=0/1` 可以直接做 A/B。

### 7.9 复制样本后的 B=768 训练 A/B

`benchmark_gpu_efficiency.py` 增加了 `--repeat_batch_to_size`：对每个训练 batch
按 batch 维复制行到目标大小，验证集保持原始数据。这样本地 `data_sample_1000`
也能让全模型 forward/backward 看到真实 `B=768` 形状。

直接 `B=768` 在 RTX 3090 24GB 上会 OOM，且 OOM 发生在 sequence embedding /
RMSNorm 阶段，早于 talking-head cross-attention。为完成本地 A/B，本次启用现有
`--activation_checkpoint_mode all_blocks`，两边使用完全相同的 checkpoint 策略：

```bash
CUDA_VISIBLE_DEVICES=0 TALKING_HEAD_TRITON=0 conda run -n taac python benchmark_gpu_efficiency.py \
  --data_dir data_sample_1000 \
  --schema_path data_sample_1000/schema.json \
  --device cuda:0 \
  --batch_size 768 \
  --repeat_batch_to_size 768 \
  --epochs 10 \
  --repeats 1 \
  --warmup_epochs 0 \
  --variant optimized_dense_clip \
  --attention_chunk_sizes '' \
  --amp_dtype bf16 \
  --loss_type focal \
  --focal_alpha 0.5 \
  --focal_gamma 2.0 \
  --activation_checkpoint_mode all_blocks \
  --log_dir logs \
  --json
```

训练集源数据仍是 900 行，复制后每 epoch 为 `9 × 768 = 6912` 行，10 epoch 共
`90` train steps / `69120` synthetic rows；验证集保持 100 行：

| setting | total train wall | train wall / epoch | total train GPU | AUC | logloss | report |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Triton off | `57.8522 s` | `5.7852 s` | `57.8521 s` | `0.615530` | `0.384455` | `logs/benchmark_gpu_epoch_20260505_174900.json` |
| Triton on | `58.2938 s` | `5.8294 s` | `58.2938 s` | `0.618845` | `0.387244` | `logs/benchmark_gpu_epoch_20260505_175019.json` |

端到端训练 wall 为 `0.992x`，即开启 kernel 慢 `0.76%`。扣除 H2D 后的模型计算段：

| setting | train wall - H2D | compute speedup |
| --- | ---: | ---: |
| Triton off | `50.5223 s` | baseline |
| Triton on | `50.6747 s` | `0.997x` |

这说明在当前全模型训练里，talking-head cross-attention 即使单核有收益，也不是主导
耗时；复制到 `B=768` 后，全模型瓶颈仍更多落在长序列 embedding、self-attention、
FFN 与 activation checkpoint 重算。AUC 差值 `+0.003314` 同样只作为 smoke 指标：
验证集只有 100 行，且 Triton seeded dropout 与 PyTorch dropout 的随机轨迹不同。
