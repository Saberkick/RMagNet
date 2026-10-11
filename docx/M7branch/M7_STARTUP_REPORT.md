# M7 启动记录

日期（上海）：2026-10-11。

- 已在 srtp_xuke6 的独立目录 RMagNet-M7 启动，tmux 会话 `m7-latent-tail`。
- 运行目录：`runs/m7_m4_latent_tail_e5/`。
- 配置：GPU 1、2、3；5 epoch；有效 batch 6；每 epoch 34 更新，总目标 170 更新；无早停。
- 新增可训练参数 4,062,464；原 M4 全部冻结。
- 229 个输入缓存完成（204 train、25 validation），样本约 2.976 GiB，复制尾部基座约 1.266 GiB，总计约 4.24 GiB；没有封存测试缓存。
- 首次启动发现 pinned VAE 不支持内置 gradient checkpointing；改用对 Decoder 调用的 non-reentrant checkpoint，缓存未重建。
- 零初始化检查通过：修正前后 latent 完全一致。基线验证 PSNR 24.0577 dB、SSIM 0.875924（25 张验证图，不是封存测试）。M4 对照中约 4e-6 的 SSIM 差来自 CPU/GPU 浮点转换，不能当作提升。
- 检查时已观察 96 次更新；新 LoRA 第3步梯度范数 3.2337e-07，输出头参数变化检查通过；每步检查冻结 VAE 与复制基座无梯度。
- 已观察 rank0 显存峰值 6.014 GiB，训练日志同时记录各 rank 平均峰值；无 OOM、无非有限损失。
- latest 模型与 optimizer/scheduler 状态已保存；此时 checkpoint cursor：`{'epoch': 2, 'batch': 22, 'step': 90}`。best 使用验证 RGB8 L1 选择，epoch0 基线也可以保留为 best。
- 本记录仅说明训练启动与梯度链路正常，不是训练结论。SSH 断开后由 tmux 继续运行；不监控完整 5 epoch，不运行额外测试。

## 查看状态

```bash
./bin/xuke
cd /share/linmingheng-local/xuke/RMagNet-M7
tail -n 20 runs/m7_m4_latent_tail_e5/logs/train.log
tmux attach -t m7-latent-tail
```

按 Ctrl+B，然后 D，脱离 tmux 会话。

设计、损失、数据、预算及续训命令见 `M7_LATENT_TAIL_EXPERIMENT.md`。
