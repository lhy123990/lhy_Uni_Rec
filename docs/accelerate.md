# 混合精度与 Numba 加速说明

本文记录本次围绕训练/验证/推理加速引入的改动、开启方式和基础测试结果。

## 改动摘要

- 训练入口 `train.py` 新增混合精度参数：
  - `--amp_dtype {auto,bf16,fp16,fp32}`
  - `--disable_amp`
  - `--precision_log_every_n_steps`
- `trainer.py` 在训练 forward/loss、验证和 `model.predict` 推理路径中接入
  `torch.autocast`。`auto` 在 CUDA 上优先使用 bf16；如果 bf16 不可用则回退到 fp16。
- 显式 fp16 训练启用 `torch.amp.GradScaler`，bf16 和 fp32 不启用 scaler。
- 使用 `writer.add_scalar` 记录精度稳定性指标：
  - `Precision/train_logits_mean`
  - `Precision/train_logits_var`
  - `Precision/loss_scaler`
  - `Precision/valid_logits_mean`
  - `Precision/valid_logits_var`
  - `Precision/predict_logits_mean`
  - `Precision/predict_logits_var`
  - `Precision/predict_embedding_mean`
  - `Precision/predict_embedding_var`
- `model.py` 中 raw dense feature projection 固定使用 fp32 计算，再 cast 回当前 AMP dtype。
  这样保留主体半精度计算，同时避免显式 fp16 下 dense 特征动态范围导致 NaN。
- `model.py` 中所有 `LayerNorm` 已替换为自定义 `RMSNorm`，归一化统计在 fp32 中计算后
  cast 回输入 dtype，以降低半精度训练/推理中的数值不稳定风险。
- attention 路径已经使用 `F.scaled_dot_product_attention`，本次主要加速未被 SDPA 覆盖的
  CPU 数据转换与时间桶计算。
- `dataset.py` 在 numba 可用时自动启用 JIT 路径，加速：
  - 变长 int 列 padding 与长度统计
  - 变长 float 列 padding
  - sequence side-info 填充
  - timestamp padding 与 time bucket 计算
- 无 numba 时保留原 Python/Numpy fallback，输出语义保持一致。
- 新增 `prepare.sh`：

```bash
conda install -y numba
```

## 开启方式

默认 CUDA 训练会启用 AMP：

```bash
bash run.sh \
  --data_dir data_sample_1000
```

显式使用自动策略：

```bash
bash run.sh \
  --data_dir data_sample_1000 \
  --amp_dtype auto
```

H20 线上建议使用 `auto` 或显式 bf16：

```bash
bash run.sh \
  --data_dir <online_data_dir> \
  --amp_dtype bf16
```

需要验证 fp16 + GradScaler 路径时：

```bash
bash run.sh \
  --data_dir data_sample_1000 \
  --amp_dtype fp16
```

关闭混合精度：

```bash
bash run.sh \
  --data_dir data_sample_1000 \
  --amp_dtype fp32
```

或：

```bash
bash run.sh \
  --data_dir data_sample_1000 \
  --disable_amp
```

调整训练 step 级精度日志频率：

```bash
bash run.sh \
  --data_dir data_sample_1000 \
  --precision_log_every_n_steps 100
```

`--precision_log_every_n_steps 0` 会关闭训练 step 级 logits/scaler 统计；验证阶段的
logits 与 predict embedding 统计仍会在每次 validation 后记录。

Numba 加速无需额外训练参数。只要当前 Python 环境能 `import numba`，`dataset.py` 会自动使用
JIT 路径；否则自动使用 fallback。线上环境可先执行：

```bash
bash prepare.sh
```

## 指标说明

训练与验证指标：

- `Loss/train`：训练 step 的分类损失。当前默认是 BCEWithLogits；如果使用
  `--loss_type focal`，则记录 focal loss。它主要用于观察训练是否正常下降以及是否出现
  `NaN`/`Inf`。
- `AUC/valid`：验证集二分类 AUC，越高越好。它衡量正负样本排序能力，也是当前
  early stopping 使用的主指标。
- `LogLoss/valid`：验证集 BCE logloss，越低越好。它衡量预测概率校准与整体分类损失，
  可辅助判断 AUC 接近时哪个模型更稳。

混合精度稳定性指标：

- `Precision/train_logits_mean`：训练 step logits 的均值。持续快速漂移到很大的正值或负值，
  通常意味着输出分布偏移，可能伴随 loss 异常。
- `Precision/train_logits_var`：训练 step logits 的方差。突然暴涨常见于数值不稳定；
  长时间接近 0 则可能表示输出塌缩。
- `Precision/loss_scaler`：仅 fp16 + GradScaler 路径有意义。数值长期稳定或逐步增长通常正常；
  频繁下降说明 scaler 检测到 overflow，训练可能正在接近 fp16 数值上限。
- `Precision/valid_logits_mean`：验证阶段 logits 的均值。用于对比训练与验证输出分布是否明显偏移。
- `Precision/valid_logits_var`：验证阶段 logits 的方差。用于观察验证输出是否爆炸或塌缩。
- `Precision/predict_logits_mean`：`model.predict` 推理路径 logits 的均值。当前验证通过
  `predict` 路径执行，因此它应与 `valid_logits_mean` 一致。
- `Precision/predict_logits_var`：`model.predict` 推理路径 logits 的方差。当前验证通过
  `predict` 路径执行，因此它应与 `valid_logits_var` 一致。
- `Precision/predict_embedding_mean`：`model.predict` 返回的最终 embedding 均值，用于监控推理侧
  表征分布是否发生明显漂移。
- `Precision/predict_embedding_var`：`model.predict` 返回的最终 embedding 方差。突然变大可能表示
  半精度推理中间激活不稳定；长期接近 0 可能表示表征塌缩。

解读建议：

- bf16 通常不需要 loss scaling；H20 线上优先看 `train_logits_*`、`valid_logits_*` 和
  `predict_embedding_*` 是否平稳。
- fp16 需要重点看 `Precision/loss_scaler`。如果 scaler 频繁下降，同时 logits 方差暴涨或出现
  NaN 预测，应优先排查输入动态范围较大的 dense 特征、RMSNorm 前后激活和 loss。
- 小样本冒烟测试中的 AUC 波动很大，主要用于验证流程和数值稳定性；质量结论需要更大的验证集。

## 基础测试结果

测试环境：

- Conda env: `taac`
- `torch 2.7.1+cu126`
- `pyarrow 23.0.1`
- `sklearn 1.7.2`
- `numpy 2.2.5`
- `numba 0.65.0`
- `tensorboard 2.20.0`
- CUDA 可用，当前测试机 `torch.cuda.is_bf16_supported()` 返回 `True`

已执行检查：

```bash
conda run -n taac python -m py_compile train.py trainer.py dataset.py model.py
git diff --check
bash -n prepare.sh
```

结果：全部通过。

模型归一化替换检查：

- 构造默认 RankMixer 配置模型后统计模块类型。
- 结果：`rmsnorm=59`，`layernorm=0`。

Numba 与 fallback 等价检查：

- 同一个 `RecordBatch` 同时走 numba 和 fallback 路径。
- 比较所有 tensor 输出的 shape 与值。
- 结果：`tensor_keys=18`，`bad=[]`，完全一致。

TensorBoard scalar 检查：

- 已确认 event 文件中包含全部精度监控 tags：
  - `Precision/train_logits_mean`
  - `Precision/train_logits_var`
  - `Precision/loss_scaler`
  - `Precision/valid_logits_mean`
  - `Precision/valid_logits_var`
  - `Precision/predict_logits_mean`
  - `Precision/predict_logits_var`
  - `Precision/predict_embedding_mean`
  - `Precision/predict_embedding_var`

Compile checkpoint 检查：

- 使用 `--torch_compile` 训练后，加载 top-k checkpoint 到未 compile 的普通模型。
- 结果：`orig_mod_keys=0`，`strict_load_ok=420`。
- 说明：checkpoint key 未出现 compiled wrapper 常见的 `_orig_mod.` 前缀。

训练冒烟配置：

```bash
TRAIN_CKPT_PATH=/tmp/lhy_unirec_smoke_<mode>_ckpt \
TRAIN_LOG_PATH=/tmp/lhy_unirec_smoke_<mode>_logs \
TRAIN_TF_EVENTS_PATH=/tmp/lhy_unirec_smoke_<mode>_tf \
conda run -n taac python train.py \
  --data_dir data_sample_1000 \
  --batch_size 128 \
  --num_epochs 1 \
  --patience 1 \
  --num_workers 0 \
  --buffer_batches 0 \
  --valid_ratio 0.1 \
  --train_ratio 0.2 \
  --ns_tokenizer_type rankmixer \
  --user_ns_tokens 5 \
  --item_ns_tokens 2 \
  --num_queries 2 \
  --ns_groups_json "" \
  --emb_skip_threshold 1000000 \
  --reinit_sparse_after_epoch 999 \
  --precision_log_every_n_steps 1 \
  --amp_dtype <mode>
```

结果：

| Mode | AMP 解析 | GradScaler | Loss | AUC | LogLoss | 结果 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `fp32` | disabled | false | 0.6882 | 0.5406 | 0.4008 | 通过 |
| `auto` | bf16 | false | 0.6844 | 0.5443 | 0.4011 | 通过 |
| `fp16` | fp16 | true | 0.6847 | 0.5446 | 0.4012 | 通过 |

`torch.compile` 对照测试：

配置：

```bash
TRAIN_CKPT_PATH=/tmp/lhy_compile_<mode>_ckpt \
TRAIN_LOG_PATH=/tmp/lhy_compile_<mode>_logs \
TRAIN_TF_EVENTS_PATH=/tmp/lhy_compile_<mode>_tf \
conda run -n taac python train.py \
  --data_dir data_sample_1000 \
  --batch_size 128 \
  --num_epochs 2 \
  --patience 10 \
  --num_workers 0 \
  --buffer_batches 0 \
  --valid_ratio 0.1 \
  --train_ratio 0.2 \
  --ns_tokenizer_type rankmixer \
  --user_ns_tokens 5 \
  --item_ns_tokens 2 \
  --num_queries 2 \
  --ns_groups_json "" \
  --emb_skip_threshold 1000000 \
  --reinit_sparse_after_epoch 999 \
  --amp_dtype auto \
  --precision_log_every_n_steps 1 \
  [--torch_compile]
```

结果：

| Variant | AMP 解析 | Total wall | Epoch 1 train | Epoch 1 valid | Epoch 2 train | Epoch 2 valid | Final AUC | Final LogLoss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no compile | bf16 | 19.08s | 1.87s | 1.29s | 0.35s | 0.91s | 0.5754 | 0.4543 |
| compile | bf16 | 189.06s | 116.96s | 46.53s | 2.18s | 2.31s | 0.5773 | 0.4569 |

精度与稳定性对比：

| Metric | no compile step 2 | compile step 2 |
| --- | ---: | ---: |
| `Loss/train` | 0.064077 | 0.065156 |
| `AUC/valid` | 0.575432 | 0.577253 |
| `LogLoss/valid` | 0.454279 | 0.456892 |
| `Precision/train_logits_mean` | -1.119258 | -1.108437 |
| `Precision/train_logits_var` | 0.015431 | 0.014248 |
| `Precision/valid_logits_mean` | -0.842873 | -0.829501 |
| `Precision/valid_logits_var` | 0.006273 | 0.006111 |
| `Precision/predict_embedding_mean` | 0.183261 | 0.178464 |
| `Precision/predict_embedding_var` | 0.966532 | 0.968251 |

结论：

- 当前小样本配置下，`torch.compile` 功能正常，但首次 compile 成本极高，整体 wall time 明显变慢。
- compile 后第二轮 train/valid 已恢复到秒级，但仍没有看到稳定超过 no-compile 的收益。
- AUC、LogLoss、logits 分布和 embedding 方差变化都在小样本波动范围内，没有观察到明显数值异常。
- 线上长训练可以继续保留 `--torch_compile` 作为实验开关，但不建议默认开启。

100 epoch compile 摊销测试：

同样使用 `data_sample_1000`、`train_ratio=0.2`、`valid_ratio=0.1`、`batch_size=128`、
`amp_dtype=auto`、`precision_log_every_n_steps=10`，分别运行 no-compile 与 compile 100 epoch。

| Variant | AMP 解析 | Total wall | Avg wall/epoch | Stable epoch delta 11-100 | Final Loss | Final AUC | Final LogLoss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no compile | bf16 | 166.22s | 1.662s | 1.289s | 0.000193 | 0.606186 | 0.355815 |
| compile | bf16 | 247.46s | 2.475s | 1.678s | 0.000183 | 0.601690 | 0.355288 |

100 epoch 末尾稳定性指标：

| Metric | no compile epoch 100 | compile epoch 100 |
| --- | ---: | ---: |
| `Precision/train_logits_mean` | -1.763906 | -1.762812 |
| `Precision/train_logits_var` | 3.555210 | 3.599245 |
| `Precision/valid_logits_mean` | -2.195169 | -2.185515 |
| `Precision/valid_logits_var` | 0.625355 | 0.615524 |
| `Precision/predict_embedding_mean` | 0.109735 | 0.115779 |
| `Precision/predict_embedding_var` | 0.988355 | 0.986995 |

100 epoch 结论：

- 100 epoch 后 compile 总耗时仍慢于 no-compile：`247.46s` vs `166.22s`。
- 排除前期冷启动后，compile 稳定期 epoch 间隔仍慢于 no-compile：`1.678s` vs `1.289s`。
- compile 末尾 AUC 略低、LogLoss 略低，差异处于小样本随机波动范围内；未观察到 logits 或
  embedding 分布异常。
- 在当前 3090 小样本/小 batch 测试配置下，`torch.compile` 没有带来速度收益。线上 H20、
  更大 batch、更长序列或更稳定 shape 时仍可单独复测，但当前不建议默认开启。

Compile + gradient checkpoint 兼容性检查：

```bash
TRAIN_CKPT_PATH=/tmp/lhy_compile_gc_ckpt \
TRAIN_LOG_PATH=/tmp/lhy_compile_gc_logs \
TRAIN_TF_EVENTS_PATH=/tmp/lhy_compile_gc_tf \
conda run -n taac python train.py \
  --data_dir data_sample_1000 \
  --batch_size 128 \
  --num_epochs 2 \
  --patience 10 \
  --num_workers 0 \
  --buffer_batches 0 \
  --valid_ratio 0.1 \
  --train_ratio 0.2 \
  --ns_tokenizer_type rankmixer \
  --user_ns_tokens 5 \
  --item_ns_tokens 2 \
  --num_queries 2 \
  --ns_groups_json "" \
  --emb_skip_threshold 1000000 \
  --reinit_sparse_after_epoch 999 \
  --amp_dtype auto \
  --precision_log_every_n_steps 1 \
  --torch_compile \
  --activation_checkpoint_mode all_blocks
```

结果：

| Test | Result |
| --- | --- |
| forward/backward | 通过 |
| validation / `model.predict` | 通过 |
| checkpoint save | 通过 |
| checkpoint strict load 到未 compile 模型 | 通过，`orig_mod_keys=0`，`strict_load_ok=420` |
| AMP 监控 scalar | 通过，训练/验证/predict embedding 指标均有记录 |

兼容性测试耗时 `89.70s`。第 1 个 train step 主要消耗在 compile 冷启动，epoch 2 train
恢复到秒级以内；`activation_checkpoint_mode=all_blocks` 与 `--torch_compile` 可同时工作。

说明：

- 最初显式 fp16 曾在 raw dense feature projection 处产生 NaN；已通过 fp32 dense projection
  修复。随后又将 LayerNorm 替换为 RMSNorm，并重新通过 fp32、bf16 auto 和 fp16 冒烟。
- checkpoint 保存格式保持不变，仍保存 `model.state_dict()`。
- RMSNorm 替换会改变模型参数集合；旧 LayerNorm checkpoint 不建议直接 strict 加载，应使用
  RMSNorm 版本重新训练得到 checkpoint。
- `prepare.sh` 只包含线上要求的 numba 安装命令；本地测试中额外安装了 tensorboard，是为了满足
  现有 `train.py` 中 `SummaryWriter` 的运行依赖。
