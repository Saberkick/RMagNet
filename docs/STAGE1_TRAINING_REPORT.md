# Stage 1：Reflection LoRA 训练流程与实测报告

## 1. 目标与监督定义

Stage 1 只训练共享 Qwen DiT 上的 `LoRA_R`。冻结以下部分：

- Qwen-Image-Edit-2509 NF4 Transformer 底座；
- Qwen VAE；
- WindowSeat `LoRA_T`；
- 固定文本 embedding；
- Interface Head（M1a 首版关闭）。

当前真实数据的监督关系为：

```text
I = {num}_45_aligned.jpg  →  LoRA_R  →  R = {num}_90.jpg
```

这里的 `R` 是 90° 偏振角拍到的“反射加强图”，不是严格满足 `I=T+R` 的纯反射辐射层。因此 Stage 1 学到的是从 45° 图像到 90° 反射增强观测的映射。它能为后续融合提供反射内容提示，但在论文和消融中不能把它直接表述为物理真值反射层。

## 2. 数据

- 位置：`/share/linmingheng-local/xuke/datasets/rmagnet_stage1_512x384`
- 样本数：53 组三元组；每个角色目录各 53 张 PNG。
- 输入尺寸：完整画面 `512×384`，无裁剪、无形变，RGB 8-bit PNG。
- 训练集：除 11、12、17 之外的 50 组。
- 验证集：11、12、17，共 3 组；与此前 RDNet/WindowSeat 对比口径一致。
- 训练增强：I/R 同步随机水平翻转；不做会破坏配准的独立增强。
- `transmission_layer` 在 Stage 1 只用于检查三元组完整性，不进入损失。

## 3. 模型与损失

反射 adapter 从 WindowSeat LoRA 配置复制 target module 列表，单独注册为 `reflection` adapter：

- rank：8；
- alpha：8；
- trainable parameters：53,261,312；
- 真实前向中有梯度的 LoRA tensors：1,442；
- 底座、VAE 和 T adapter 不更新。

模型在一次固定 flow step 后解码到 RGB。损失在 `[0,1]` RGB 上计算：

\[
L_R=L_1+0.2(1-SSIM)+0.1L_{edge}
\]

其中 `L_edge` 是水平和垂直一阶差分的 L1。优化器为 AdamW，默认学习率 `1e-4`、weight decay `1e-2`、梯度裁剪 1.0；学习率采用 20 step warmup 加 cosine decay。

## 4. 多 GPU 实现

正式脚本默认使用 8 张 RTX 3090：

```text
8 processes × batch 1 × accumulation 1 = global batch 8
```

每个进程在自己的 GPU 上持有一份 4-bit 冻结底座。训练器不使用 DDP 包装整个 12.5B 模型，以避免广播和管理大量冻结参数；反向后只对实际产生梯度的 `LoRA_R` 参数执行 NCCL all-reduce。每一步先比较各 rank 的梯度存在掩码，只有所有 rank 一致时才继续。

50 个训练样本在 8 卡上由 `DistributedSampler` 分配，每 epoch 每个 rank 7 个 batch，即 7 个 optimizer steps。默认 100 epochs 共 700 steps。

## 5. Checkpoint 和恢复

每个 checkpoint 包含：

- `reflection_lora.safetensors`：仅 R adapter，约 204 MiB；
- `trainer_state.pt`：AdamW 和 scheduler 状态，约 406 MiB；
- `config.json`：完整命令参数。

默认保留最近两个 checkpoint，并额外保存按最低验证 L1 选择的 `best_reflection_lora.safetensors`。`last_checkpoint.txt` 指向最近 checkpoint。恢复时同时恢复 epoch 内的下一个 micro-batch 位置，因此不会从该 epoch 开头重复样本。

## 6. 执行方法

正式训练：

```bash
cd /share/linmingheng-local/xuke/RMagNet
bash scripts/run_stage1.sh
```

后台执行：

```bash
nohup bash scripts/run_stage1.sh > runs/stage1_launcher.log 2>&1 &
```

默认 `RESUME=auto`。中断后重复相同命令即可从最近 checkpoint 恢复。可用环境变量覆盖参数，例如：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
EPOCHS=50 \
LEARNING_RATE=5e-5 \
RUN_DIR=/share/linmingheng-local/xuke/RMagNet/runs/stage1_custom \
bash scripts/run_stage1.sh
```

两卡短程链路测试：

```bash
CUDA_VISIBLE_DEVICES=0,1 MAX_STEPS=2 bash scripts/smoke_stage1.sh
```

## 7. 已完成的真实训练检查

| 检查 | 结果 |
|---|---:|
| 两卡 2-step 前向、反向、同步、验证、保存 | 通过 |
| 八卡 5-step 训练 | 通过 |
| 从 step 5 恢复并继续到 step 7 | 通过 |
| 单卡峰值 allocated 显存 | 18.54 GiB |
| 八卡 step 1 loss | 0.26419 |
| 八卡 step 5 loss | 0.21093 |
| step 5 验证 L1 / PSNR / SSIM | 0.10832 / 17.210 / 0.5449 |
| 恢复后 step 7 loss | 0.16905 |
| step 7 验证 L1 / PSNR / SSIM | 0.10756 / 17.254 / 0.5550 |

短程 loss 来自不同 mini-batch，只用于证明数值有限、梯度更新有效，不作为收敛结论。验证集只有三张，也不足以支持模型优劣结论。

## 8. 输出与验收

默认正式输出目录为 `runs/stage1_reflection_r8`：

- `train.log`：完整控制台输出；
- `metrics.jsonl`：逐 step 训练指标和验证指标；
- `run_config.json`：数据 split、代码提交、world size 和全部超参数；
- `checkpoint-*`：可恢复训练状态；
- `best_reflection_lora.safetensors`：最低验证 L1 对应的 R adapter；
- `best_metrics.json`：最佳 step 与指标。

Stage 1 的工程验收条件是：700 steps 正常结束、验证指标保持有限、最佳权重可离线加载，且抽样结果确实趋向 90° 反射增强观测。是否适合充当后续 `R` 反参考，需要在 Stage 3 做 R 删除、R shuffle 和纯 T 基线消融后判断。
