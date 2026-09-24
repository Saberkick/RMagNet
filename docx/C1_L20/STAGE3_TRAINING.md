# C1-L20 阶段三：正式训练

## 状态

正式入口已在 4×RTX 3090 上完成一轮验收，并完成跨进程恢复到第 2、3 个 epoch 的连续验证。训练脚本现在每个 epoch 使用独立 `torchrun` 进程，完整保存 LoRA、优化器、调度器、随机数和 early-stopping 状态，再自动恢复下一轮。验证目录在确认成功后已删除，只保留正式一轮验收产物。

## 正式入口

推荐首轮：

```bash
./bin/xuke
cd /share/linmingheng-local/xuke/RMagNet
EPOCHS=10 RUN_NAME=c1_l20_e10 \
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
bash scripts/train_c1_l20.sh
```

也可用位置参数：

```bash
RUN_NAME=c1_l20_e5 CUDA_VISIBLE_DEVICES=0,1,2,3 \
OMP_NUM_THREADS=1 bash scripts/train_c1_l20.sh 5
```

限制优化更新数：

```bash
MAX_STEPS=100 EPOCHS=20 RUN_NAME=c1_l20_s100 \
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
bash scripts/train_c1_l20.sh
```

只运行轻量预检、不加载 Qwen：

```bash
PREFLIGHT_ONLY=1 RUN_NAME=preflight_c1_l20 \
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_c1_l20.sh 2
```

预检会创建一个只含锁和日志目录的运行目录；确认无误后可删除该预检目录。

## 固定训练定义

- 四张 GPU，每卡 batch 1。
- 梯度累积 2，有效 batch 8；最后不足 8 张的更新按实际 microbatch 数归一化。
- 每个 epoch 完整遍历 50 张训练图，每 epoch 约 7 次优化更新。
- Stage 2 最佳 `LoRA_T` 初始化，主干、VAE 均冻结。
- Qwen 教师与学生共享冻结 NF4 主干，教师关闭全部 LoRA，仅运行到 block 20。
- 教师 Q20 图先求到预测图的梯度，随后释放教师计算图，再将组合梯度回传到学生 `LoRA_T`。
- 不做随机翻转，因为缓存的 Q20(GT) 含位置编码和全局上下文，简单翻转缓存特征并不等价于重新编码翻转后的 GT。
- BF16 由 WindowSeat 的 `flow_step` 自动混合精度控制。
- PagedAdamW8bit，学习率 `5e-5`，cosine 调度，5% warmup 且至少 10 步，梯度裁剪 1.0。
- 每个 epoch 重建 CUDA 进程；约 1.7 GiB/卡的 8-bit Adam 状态在前向期间驻留 CPU，只在 `optimizer.step()` 前移入 GPU，更新后立即卸载。
- 每 20 次更新测一次基础项和 Q20 项对预测图的实际梯度范数；第一 epoch 目标比例渐增到 15%，随后为 20%，`lambda_q` 用 EMA 更新并限制在 `[0.02,0.5]`。

损失为：

```text
L = L_base + 0.25*L_weighted_charbonnier
    + 0.10*L_low_response_keep + lambda_q*L_Q20
```

其中保持项约束低响应区域接近输入图，减少非反射区域的无依据改写。

四个损失项、权重归一化、Q20 余弦距离及 `lambda_q` 梯度控制的完整定义见 [LOSS_DEFINITION.md](LOSS_DEFINITION.md)。

## 保护措施

脚本在加载 12.5B 模型前检查：

- 恰好四张可见 GPU。
- 固定 uv 环境、Stage 2 权重和完整 C1-L20 缓存存在。
- 50 份 GT 特征和 50 份权重存在。
- 缓存公式版本、block 20/index 19 和 timestep 499 一致。
- 至少 30 GiB 剩余空间。
- `RUN_NAME` 不含路径字符。
- 非空运行目录必须具有可恢复的 `checkpoints/last/trainer_state.pt`。
- 同一运行目录不能被两个进程同时占用。

训练器还会逐图核对 I、GT、DoLP 的 SHA-256，并检查缓存、损失和梯度的有限性。

## 输出与恢复

输出位于：

```text
runs/c1_l20/<RUN_NAME>/
├── config.yaml
├── logs/
├── checkpoints/
│   ├── best/
│   ├── last/
│   ├── epoch_*.safetensors
│   └── top_epochs.json
├── validation/
│   ├── predictions/
│   ├── error_maps/
│   └── metrics.csv
└── training_summary.md
```

只保留 PSNR 最好的三个 epoch 文件；`best` 和 `last` 使用硬链接，避免重复占用约 3.2 GB 的 LoRA 文件空间。`last/trainer_state.pt` 保存 8-bit 优化器、调度器、全局步数、下一数据位置、`lambda_q` 和 early-stopping 状态。

相同命令和相同 `RUN_NAME` 默认 `RESUME=auto`。外层脚本读取 `resume_meta.json`，逐 epoch 启动独立的四卡训练进程，并检查每次生成的 `launch_status.json`。这会回收 CUDA allocator 和 Qwen/VAE 临时状态，同时保持 Adam 动量、学习率调度和随机数据顺序连续。需要明确新开实验时使用新的 `RUN_NAME`；`RESUME=none` 会拒绝写入非空目录。

## BF16 缓存前置条件

正式训练会在加载 Qwen 前逐个读取并验证 50 份 `Q20(GT)`：必须为 BF16、形状 `768×3072`、全部有限且 SHA-256 与 manifest 一致。旧 FP16 特征会因第20层激活超出 65504 而溢出，不能用于训练。

## 2026-09-23 四卡一轮验收

验收运行目录：`runs/c1_l20/c1_l20_fixverify_e1/`。

固定配置为 4×RTX 3090、每卡 batch 1、梯度累积 2、有效 batch 8、50 张训练图、7 次优化更新，并从 Stage 2 LoRA_T 重新初始化。完整一轮已经依次通过训练、`11/12/17` 验证、预测图输出、checkpoint 保存和 `last` checkpoint 恢复。

验证结果：

- 平均 PSNR：`22.931248 dB`
- 平均 SSIM：`0.820134`
- 峰值显存：每卡约 `22.637 GiB`
- LoRA_T 有梯度的张量数：`1442`
- checkpoint 实际占用约 `4.8 GiB`；`best`、`last` 与 `epoch_0001` 的 LoRA 使用同一 inode

本次验收修正了三个运行期问题：

1. Q20 教师的 checkpoint 重算必须与原前向保持相同的 LoRA 禁用状态。
2. 每个 microbatch 结束后必须释放输出梯度图并清理 CUDA 缓存，避免 24 GiB GPU 在优化器状态建立后发生显存碎片 OOM。
3. 验证集只读取 `I/GT`，不应访问仅为 50 张训练图生成的 Q20/权重缓存。

### 梯度比例观察

首轮记录的原始 Q20 输出梯度约为基础输出梯度的 97 倍。由于当前配置把 `lambda_q` 下限固定为 `0.02`，实测 Q20/基础梯度比例约为 `1.945`，高于“首轮最高 15%”的目标。该值已记录在 `logs/train.jsonl`；运行流程已经可用，但在长轮次实验前需要单独决定是否修改梯度控制定义或 `lambda_q` 下限。


## 2026-09-23 多轮恢复与显存修复

最初的多轮训练会在第 3 个 epoch OOM。仅在 epoch 间重启进程仍不足以解决：bitsandbytes 默认在恢复时把约 `1.7 GiB` 的 Adam 状态预先放到每张 GPU，导致恢复后的第一次 Qwen/VAE 前向没有余量。

修复包含两部分：

1. 每个 epoch 结束保存完整恢复状态，退出本轮 `torchrun`，下一轮由同一 Shell 自动恢复。
2. 恢复优化器时使用 CPU map；前向期间 Adam 状态保留在 CPU，计算图释放后才移入 GPU执行更新，随后立即移回 CPU。

三轮验证从第 1 轮 checkpoint 连续恢复并完成第 2、3 轮，共达到 `21` 次优化更新。两次进程恢复均保持调度器和 Adam 状态连续；各卡峰值稳定为约 `22.612 GiB`，更新后已分配显存回落到约 `14.422 GiB`。第 3 轮验证平均指标为 `23.233635 dB / 0.823874 SSIM`。该运行只用于验证恢复边界，验证目录已清理，不能作为正式实验结果引用。
