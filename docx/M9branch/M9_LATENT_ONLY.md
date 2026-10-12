# M9：M4 编辑 LoRA 的 latent-only 微调

## 实验目的

从 M4 final best 初始化，只用最终归一化 latent 与 GT 编码 latent 的 L1 微调已有编辑 LoRA。测试直接监督内部图像表示是否有效。此实验不预先认定 latent 距离就是语义距离。

$$
z_I=\operatorname{Normalize}(\operatorname{mode}(E(I))),\quad z_{GT}=\operatorname{Normalize}(\operatorname{mode}(E(GT)))
$$

$$
\hat z=z_I-v_\theta(z_I,t),\qquad L=\operatorname{mean}|\hat z-z_{GT}|
$$

公式中的一步更新严格复用固定 WindowSeat native full forward。当前 BF16 计算 `BF16(499)/1000` 实际为 0.5；保持历史协议，不改写为 BF16(0.499)。监督最终 latent，不监督速度或中间 token。

## 来源与冻结范围

- 初始化 `/share/linmingheng-local/xuke/RMagNet/runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors`。
- SHA-256：`897282b1bb9cfe61f96530df72edcf8a44a066bb819a3663e9100862aefdb2a3`。
- 冻结 NF4 Qwen DiT 主干、BF16 VAE Encoder/Decoder、固定文本条件。
- 继续训练已有 default 编辑 LoRA（保持原 rank），FP32 参数。没有新增门控或其他分支。
- 训练没有 Decoder 调用、像素重建、Qwen 抽层、DoLP、P90、一致性、边缘、权重衰减或曝光对齐损失。
- 因取消 Decoder 反传，节约训练计算，但不会消除 VAE 的重建损失。

## 数据、缓存与预算

- `datasets/rmagnet_sma_dataset3`：204 train、25 validation、24 sealed test。保持比例和既有空间尺寸，不修改数据或划分。
- 仅 train 和 validation 的输入、GT posterior mode latent 预缓存；封存测试不编码、不使用。
- 缓存 `data_cache/m9_vae_latents_v1/`，来源哈希、归一化方式、权重来源进入 manifest；验证缓存等于重新编码。
- 四张卡为模型并行，每张承担15个 DiT block；不是四份 DDP。
- microbatch=1，梯度累积6，有效 batch=6；34 updates/epoch，4 epochs=136 updates，完整遍历训练集。
- 学习率 `1e-5`，前10 updates warmup，cosine 到 `1e-6`；AdamW weight_decay=0，梯度裁剪1。
- 没有 early stopping。训练随机顺序固定 seed 2026+epoch，每 epoch 将最大面积训练样本置于首位，用实际大尺寸检查显存。

## 验证、保存与稳定性检查

每个 epoch 才解码验证。预测保存为 RGB8 PNG 后，沿用项目历史 PSNR/SSIM、L1、edge 等指标；这些像素指标不参与反向传播。日志同时记录验证 latent L1。

best 按25张验证图平均 PSNR选择，初始化 M4 baseline 也参与候选；若没有提升，best 可以仍是 epoch0。latest 保存最后训练模型和可恢复优化器状态。封存测试不用于模型选择。

只保留 best/latest 两份模型，不保存 epoch 权重。latest 在首步、每20步、每个 epoch 和正常结束时更新。原子写入临时文件后替换；恢复时验证模型与优化器的 cursor 一致。恢复要求相同总 epoch、卡数和梯度累积配置，不静默修改 scheduler。

由于完整 LoRA 本身约3.4GB，best/latest 加 FP32 Adam 状态约13.6GB；原子替换优化器状态时峰值约20.4GB，加缓存、日志和验证图。启动要求25GiB剩余空间。

启动自动确认：

1. 标签纠正、数据哈希、配对、尺寸和分组划分通过既有审计。
2. 缓存等于新编码；baseline 与已验证 M4 管线指标一致。
3. 仅 default LoRA 可训练；主干和 VAE 无梯度。
4. 四个逻辑设备的 LoRA 均有非零有限梯度；首步参数实际更新。
5. 损失有限，记录实际最大尺寸显存峰值。

## 启动

```bash
cd /share/linmingheng-local/xuke/RMagNet-M9
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 EPOCHS=4 bash scripts/run_m9_latent_only.sh
```

后台使用 tmux。日志：`runs/m9_m4_latent_only_e4/logs/cache.log`、`logs/train.log`。最终退出码：`pipeline.exit_code`。

输出：`best/transmission_lora.safetensors`、`latest/transmission_lora.safetensors`、`latest/trainer.pt`、`best_metrics.json`、`latest_metrics.json`、`latest_validation/predictions/`、`train.jsonl`、`metrics.jsonl`、`training_summary.json`。

这次单独运行 latent-only 实验；没有启动同预算像素损失对照。与历史 M4 的比较不能单独证明增益来自损失替换而不是继续训练。

训练模型的验证使用 `no_grad`，不能使用 `inference_mode`：Qwen 会缓存 RoPE 张量，推理模式产生的缓存不能在后续训练中保存用于反向。离线 VAE 缓存为独立进程，保存后重新加载为普通张量，不受影响。首次启动在任何优化更新前失败；日志与数字记录保存在 `results_archive/M9/failed_start_inference_cache/`，失败启动的权重已清理。
