# Stage 2：Transmission LoRA 训练说明

## 状态

Stage 2 的真实训练流程已经跑通，正式训练未启动。

已验证内容：

- 从官方 WindowSeat rank-128 Transmission LoRA 初始化；
- 冻结 NF4 Qwen DiT、VAE 和 Reflection LoRA；
- 两卡数据并行；
- 512×384 完整画面配对训练；
- PagedAdamW 8-bit optimizer；
- 梯度同步、裁剪、验证和最佳权重；
- checkpoint 保存及中断恢复。

## 监督关系

```text
I = blended/{id}.png              # 45° 原图
T = transmission_layer/{id}.png  # 0° GT

I → Frozen Qwen DiT + trainable WindowSeat LoRA_T → T_pred
```

验证集固定为 11、12、17，其余 50 组用于训练。训练增强只做 I/T 同步水平翻转。

## 冻结与训练参数

| 部分 | 状态 |
|---|---|
| Qwen-Image-Edit-2509 Transformer | NF4，冻结 |
| Qwen VAE | BF16，冻结且 eval |
| WindowSeat `LoRA_T` | 训练 |
| `LoRA_R` | 冻结 |
| Interface Head | 关闭 |

`LoRA_T` 保留 WindowSeat 的 rank 128 和完整 target module 列表：

- trainable parameters：852,180,992；
- 实际产生梯度的 tensors：1,442；
- optimizer：`bitsandbytes.PagedAdamW8bit`；
- 默认学习率：`5e-6`；
- weight decay：`1e-2`；
- gradient clipping：1.0；
- gradient checkpointing：开启。

损失在 `[0,1]` RGB 上计算：

\[
L_T=L_1+0.2(1-SSIM)+0.1L_{edge}
\]

## 多卡策略

正式脚本默认使用四卡：

```text
4 processes × batch 1 = global batch 4
```

每个进程持有一份 4-bit 冻结底座和 T adapter。训练器只对 T adapter 梯度执行 NCCL all-reduce，不用 DDP 包装完整 12.5B 模型。

50 个训练样本在四卡下每 epoch 为 13 optimizer steps。默认 50 epochs，共 650 steps。

## 显存与存储实测

两卡 smoke 的单卡峰值：

| 测试 | allocated | reserved |
|---|---:|---:|
| 第一次 optimizer step | 18.27 GiB | 19.89 GiB |
| checkpoint 恢复后的第二步 | 19.36 GiB | 22.06 GiB |

RTX 3090 24 GB 可以运行，但恢复后的显存余量约 2.5 GiB，不建议增加单卡 batch 或分辨率。

rank-128 adapter 较大：

- `transmission_lora.safetensors`：约 3.2 GB；
- 最佳权重：约 3.2 GB；
- checkpoint 不保存重复底座；
- 默认保留最近两个 checkpoint，加最佳权重，预计约 9.6 GB。

## Checkpoint 恢复策略

PagedAdamW8bit 的分页状态在当前 bitsandbytes 0.47.0 中无法稳定反序列化，直接恢复会触发底层 `pythonInterface.cpp` 错误。因此 Stage 2 checkpoint 保存：

- T LoRA 权重；
- scheduler 状态；
- global step、epoch 和 epoch 内下一个 batch；
- 完整配置。

恢复时重建 PagedAdamW8bit，优化器动量不会恢复。模型、学习率进度和数据位置会恢复。这一取舍保证脚本能可靠继续运行，并把 checkpoint 从约 4.8 GB 降到约 3.2 GB。

若后续需要严格恢复 optimizer momentum，应更换支持可靠状态恢复的 optimizer，或增加 ZeRO/FSDP optimizer state 分片；更换前必须重新测量 24 GB 显存。

## Smoke 结果

两卡执行 2 个 optimizer steps，中间从 step 1 checkpoint 恢复：

| Step | Train loss | Val L1 | Val PSNR | Val SSIM |
|---:|---:|---:|---:|---:|
| 1 | 0.09137 | 0.05215 | 22.609 | 0.8815 |
| 2，恢复后 | 0.08116 | 0.05203 | 22.625 | 0.8115 |

Step 只有两个，且来自不同 batch；这些数值只证明训练链路有效，不表示模型已经收敛。SSIM 波动也说明正式训练必须保存初始化基线并按固定验证图观察。

Smoke 产物位于：

```text
/share/linmingheng-local/xuke/RMagNet/runs/stage2_smoke_2gpu
```

## 执行脚本

### 正式四卡训练

```bash
cd /share/linmingheng-local/xuke/RMagNet
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_stage2.sh
```

### 选择其他 GPU

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/run_stage2.sh
```

### CPU DataLoader worker

`--num-workers` 是每个 GPU 进程的 worker 数。当前数据很小，建议每卡 1 个：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/run_stage2.sh --num-workers 1
```

### 两卡 smoke

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/smoke_stage2.sh
```

### 常用覆盖参数

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
EPOCHS=30 \
LEARNING_RATE=3e-6 \
SAVE_EVERY=100 \
VALIDATE_EVERY=50 \
RUN_DIR=/share/linmingheng-local/xuke/RMagNet/runs/stage2_custom \
bash scripts/run_stage2.sh --num-workers 1
```

`RESUME=auto` 为默认值。重复同一命令会从 `last_checkpoint.txt` 指向的位置恢复模型、scheduler 和数据位置，并重新初始化 optimizer。

## 正式训练前建议

1. 先记录未经微调的 WindowSeat T 在 11、12、17 上的保存后 PNG 指标。
2. 正式训练至少在 step 0、50、100 导出可视化，关注文字和未反射区域是否被误改。
3. 最佳模型同时参考验证 L1、PSNR、SSIM；不能只用训练 loss。
4. 如果 T 很快过拟合 50 张训练图，优先减少 epoch 或降低学习率，不增加 LoRA rank。

## 正式评估

训练完成后执行：

```bash
cd /share/linmingheng-local/xuke/RMagNet
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_stage2.sh
```

默认在 11、12、17 上公平比较：

- 原始输入；
- 官方、未经微调的 WindowSeat T adapter；
- `best_transmission_lora.safetensors`；
- `last_checkpoint.txt` 指向的最终 T adapter。

每个 variant 都重置相同随机种子，以使用相同顺序的 VAE latent sample。PSNR、SSIM 和 L1 从保存后的 8-bit PNG 数值计算。输出位置：

```text
runs/stage2_transmission_r128/evaluation
```

其中包括 `metrics.csv`、`evaluation.json`、`REPORT.md`、各模型预测、×4 误差热图和逐图 panel。

只评估官方 WindowSeat 基线，不要求已有训练权重：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_stage2.sh --variants baseline
```

评估其他样本：

```bash
# 指定 ID
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_stage2.sh --ids 11,12,17,18,19

# 全部 53 张；包含训练集，只能用于诊断，不能作为泛化指标
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_stage2.sh --ids all
```
