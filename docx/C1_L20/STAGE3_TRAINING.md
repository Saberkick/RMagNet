# C1-L20 阶段三：正式训练

## 状态

四卡 smoke 在 Qwen 分片加载期间按要求人工中止，尚未发生任何优化更新。相关 `runs/c1_l20/smoke` 目录已删除。因此本入口完成了静态编译、Shell 语法和不加载模型的 preflight 检查，但没有得到运行时 smoke 验证。

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
- 每 20 次更新测一次基础项和 Q20 项对预测图的实际梯度范数；第一 epoch 目标比例渐增到 15%，随后为 20%，`lambda_q` 用 EMA 更新并限制在 `[0.02,0.5]`。

损失为：

```text
L = L_base + 0.25*L_weighted_charbonnier
    + 0.10*L_low_response_keep + lambda_q*L_Q20
```

其中保持项约束低响应区域接近输入图，减少非反射区域的无依据改写。

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

相同命令和相同 `RUN_NAME` 默认 `RESUME=auto`。需要明确新开实验时使用新的 `RUN_NAME`；`RESUME=none` 会拒绝写入非空目录。

## BF16 缓存前置条件

正式训练会在加载 Qwen 前逐个读取并验证 50 份 `Q20(GT)`：必须为 BF16、形状 `768×3072`、全部有限且 SHA-256 与 manifest 一致。旧 FP16 特征会因第20层激活超出 65504 而溢出，不能用于训练。
