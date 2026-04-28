# PCVRHyFormer 显存 Benchmark 与 Activation Checkpoint 指南

本文记录当前新增的 activation checkpoint 能力、显存 benchmark 启动方式，
以及如何根据 benchmark 日志选择 checkpoint 配置。

## 1. 新增能力

### 1.1 模型侧 checkpoint 接口

`model.py` 中新增了模型级 activation checkpoint 配置：

- `ActivationCheckpointConfig(mode, units)`
- `PCVRHyFormer.configure_activation_checkpointing(...)`
- `PCVRHyFormer.iter_activation_checkpoint_units()`

训练默认不启用 checkpoint，保持原始路径：

```bash
--activation_checkpoint_mode none
```

可选模式：

- `none`：不启用 checkpoint，默认值。
- `all_blocks`：checkpoint 每个 HyFormer block。
- `all_seq_encoders`：checkpoint 每层每个序列域的 sequence encoder。
- `custom`：只 checkpoint 指定 unit。

当前稳定 unit 名称示例：

```text
blocks.0
blocks.0.seq_encoders.seq_a
blocks.0.seq_encoders.seq_b
blocks.0.seq_encoders.seq_c
blocks.0.seq_encoders.seq_d
blocks.1
blocks.1.seq_encoders.seq_a
blocks.1.seq_encoders.seq_b
blocks.1.seq_encoders.seq_c
blocks.1.seq_encoders.seq_d
```

推荐优先使用 `custom`，因为它可以只重算显存贡献最大的模块，避免不必要的
训练变慢。

### 1.2 训练入口

`train.py` 新增参数：

```bash
--activation_checkpoint_mode none|all_blocks|all_seq_encoders|custom
--activation_checkpoint_units blocks.0.seq_encoders.seq_c,blocks.1.seq_encoders.seq_d
```

示例：

```bash
python train.py \
  --batch_size 1024 \
  --activation_checkpoint_mode custom \
  --activation_checkpoint_units blocks.0.seq_encoders.seq_c,blocks.0.seq_encoders.seq_d,blocks.1.seq_encoders.seq_c,blocks.1.seq_encoders.seq_d
```

## 2. Benchmark 启动方式

### 2.1 离线版：写入 logs

`benchmark_gpu_memory.py` 适合本地或可取回文件的环境。它会写出 txt 和 JSON 报告。

```bash
CUDA_VISIBLE_DEVICES=0 /home/hyliu/.conda/envs/taac/bin/python benchmark_gpu_memory.py \
  --data_dir data_sample_1000 \
  --device cuda:0 \
  --batch_sizes 256,512,1024 \
  --memory_budget_gib 20 \
  --check_equivalence \
  --log_dir logs
```

常用参数：

- `--batch_sizes`：要 probe 的 batch size 列表。
- `--memory_budget_gib`：目标显存预算；推荐逻辑会用它判断是否需要 checkpoint。
- `--recommend_max_units`：最多推荐多少个 checkpoint unit，默认 `4`。
- `--check_equivalence`：检查 checkpoint 关闭和推荐配置的 logits/gradient 是否一致。

### 2.2 线上版：只打 logging.info

`benchmark_gpu_memory_online.py` 适合线上平台无法取回 log 文件的场景。
它不会写报告文件，所有结果都通过 `logging.info` 输出。

线上版会从环境变量读取路径，和 `train.py` 的约定保持一致：

- `TRAIN_DATA_PATH`：数据目录。
- `TRAIN_SCHEMA_PATH`：可选，schema 路径；不设置时使用 `<TRAIN_DATA_PATH>/schema.json`。
- `TRAIN_DEVICE` 或 `CUDA_DEVICE`：可选，默认 `cuda:0`。
- `TRAIN_BATCH_SIZE`：可作为默认 batch size。

显存 benchmark 专用环境变量：

- `MEMORY_BENCH_BATCH_SIZES`：覆盖 batch size 列表，例如 `512,768,1024`。
- `MEMORY_BENCH_BUDGET_GIB`：显存预算。
- `MEMORY_BENCH_SEQ_MAX_LENS`：序列长度配置。
- `MEMORY_BENCH_CHECK_EQUIVALENCE=1`：启用等价性检查。

示例：

```bash
TRAIN_DATA_PATH=/data_ams/academic_training_data \
MEMORY_BENCH_BATCH_SIZES=512 \
MEMORY_BENCH_BUDGET_GIB=19 \
MEMORY_BENCH_CHECK_EQUIVALENCE=1 \
python3 -u benchmark_gpu_memory_online.py
```

也可以直接用命令行参数覆盖：

```bash
python3 -u benchmark_gpu_memory_online.py \
  --batch_sizes 512 \
  --memory_budget_gib 19 \
  --check_equivalence
```

注意：线上版只读取一个 probe batch，不会把全量 IterableDataset 转成 list。
这是为了避免大数据集在 GPU probe 前先耗尽 CPU 内存。

## 3. 如何理解日志

代表性输出：

```text
Checkpoint recommendation: --activation_checkpoint_mode none
Recommendation reason: Batch size 512 fits the memory budget; checkpointing is optional unless you raise batch size further.
Memory budget: 19.00 GiB

batch=512 baseline peak_alloc=13.657 GiB peak_reserved=15.211 GiB gpu=2.0131s wall=2.0132s loss=0.313812
batch=512 unit=blocks.0.seq_encoders.seq_c saved_activation=3672.20 MiB
batch=512 unit=blocks.0.seq_encoders.seq_d saved_activation=3672.20 MiB
batch=512 unit=blocks.1.seq_encoders.seq_c saved_activation=3672.20 MiB
batch=512 unit=blocks.1.seq_encoders.seq_d saved_activation=3672.20 MiB

Checkpoint equivalence units=<none> logits_max_abs_diff=0.0000000000e+00 pass_logits=True grad_max_abs_diff=2.9802322388e-08 pass_grads=True
```

### 3.1 `peak_alloc`

`peak_alloc` 是 PyTorch 统计的峰值“实际 tensor 占用”显存。
它来自 `torch.cuda.max_memory_allocated()`。

这个值用于判断模型前向、反向、参数、梯度、optimizer 状态等实际张量最高占用了多少显存。

### 3.2 `peak_reserved`

`peak_reserved` 是 PyTorch CUDA caching allocator 向 CUDA driver 申请并保留的峰值显存。
它来自 `torch.cuda.max_memory_reserved()`。

`peak_reserved` 通常大于 `peak_alloc`，因为它包含 allocator 缓存、内存块碎片和预留空间。
判断 OOM 风险时优先看 `peak_reserved`，因为它更接近进程对 GPU 显存池的真实占用。

### 3.3 `saved_activation`

`saved_activation` 是通过 `torch.autograd.graph.saved_tensors_hooks` 统计的：
某个 checkpoint unit 在 forward 中为了 backward 保存过多少 CUDA tensor bytes。

它表示“如果 checkpoint 这个 unit，理论上最值得重算的 activation 来源在哪里”。

它不是 live peak，也不能直接从 `peak_alloc` 或 `peak_reserved` 里相减。原因：

- `saved_activation` 是累计保存量，不是同一时刻的 live 显存。
- 不同 unit 的 tensor 生命周期可能重叠，也可能不重叠。
- checkpoint 后仍然需要保留输入、输出、参数、梯度和 optimizer 状态。
- PyTorch allocator 的 reserved 显存还包含缓存和碎片，不会严格按 activation 减少量下降。

因此，`saved_activation=3672 MiB` 不等于开启该 unit 后 `peak_reserved` 一定下降
`3672 MiB`。它主要用于排序：数值越大，越优先 checkpoint。

## 4. 如何根据日志调整 checkpoint

### 4.1 当前 batch 已经低于预算

如果日志显示：

```text
peak_reserved < memory_budget_gib
Checkpoint recommendation: --activation_checkpoint_mode none
```

说明当前 batch size 在预算内，不建议开启 checkpoint。
checkpoint 会减少部分 activation 保存，但会增加反向阶段重算，训练会变慢。

例如：

```text
batch=512 peak_reserved=15.211 GiB
memory_budget=19.00 GiB
```

此时推荐保持：

```bash
--activation_checkpoint_mode none
```

### 4.2 想继续提高 batch size

如果当前 batch 低于预算，但目标是继续增大 batch size，建议先扩大 probe 范围：

```bash
python3 -u benchmark_gpu_memory_online.py \
  --batch_sizes 512,768,1024 \
  --memory_budget_gib 19
```

如果更大的 batch OOM，或 `peak_reserved` 超过预算，再根据 top units 设置 checkpoint。

### 4.3 优先 checkpoint 哪些 unit

优先选择日志中 `saved_activation` 最大的 unit。例如：

```text
blocks.0.seq_encoders.seq_c saved_activation=3672.20 MiB
blocks.0.seq_encoders.seq_d saved_activation=3672.20 MiB
blocks.1.seq_encoders.seq_c saved_activation=3672.20 MiB
blocks.1.seq_encoders.seq_d saved_activation=3672.20 MiB
```

对应训练参数：

```bash
--activation_checkpoint_mode custom \
--activation_checkpoint_units blocks.0.seq_encoders.seq_c,blocks.0.seq_encoders.seq_d,blocks.1.seq_encoders.seq_c,blocks.1.seq_encoders.seq_d
```

通常 `seq_c` / `seq_d` 更靠前，是因为它们的配置长度为 `512`，
比 `seq_a` / `seq_b` 的 `256` 保存更多 attention/FFN activation。

### 4.4 推荐仍然是 `none` 的原因

推荐逻辑是保守的：

- 当前最大成功 batch 的 `peak_reserved` 超过预算时，推荐 checkpoint。
- 或者 probe 列表中出现更大 batch OOM 时，推荐 checkpoint。
- 否则推荐 `none`。

所以只 probe 一个 batch 且它低于预算时，即使 top units 的 `saved_activation` 很大，
推荐也会是 `none`。这是因为没有必要为了已经能放下的 batch 付出重算开销。

## 5. 建议工作流

1. 先用目标预算扫多个 batch size：

```bash
python3 -u benchmark_gpu_memory_online.py \
  --batch_sizes 512,768,1024 \
  --memory_budget_gib 19
```

2. 如果推荐是 `none`，直接用最大成功 batch 训练。

3. 如果推荐是 `custom`，复制日志里的：

```text
Checkpoint recommendation: --activation_checkpoint_mode custom ...
```

到 `train.py` 启动参数。

4. 开启 checkpoint 后，再对目标 batch size 跑一次 benchmark，确认：

- `peak_reserved` 降到预算内。
- `gpu` / `wall` 增加幅度可以接受。
- `--check_equivalence` 通过。

