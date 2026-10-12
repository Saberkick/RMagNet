# M9 启动检查记录

2026-10-12，从 M4 final best 初始化，latent-only 编辑 LoRA 微调。

最终使用 GPU 0、1、2、3 模型并行，4 epoch / 136 updates，无 early stopping。

已检查前三次真实训练更新，训练继续由 tmux 后台执行，未等待完整实验结束。

| 检查 | 结果 |
|---|---|
| 输入缓存与重新编码 | 最大面积训练样本逐元素相同 |
| 初始 M4 验证 | 与历史同协议结果完全一致：24.05781166 dB / 0.87592499 SSIM |
| 冻结主干与 VAE | 三步均无参数梯度 |
| 四卡 LoRA | 均有非零有限梯度，首步参数均变化 |
| 三步 latent L1 | 0.14084051、0.15702998、0.14025533；样本不同，不构成下降曲线 |
| 第二/第三步显存 | 最大约6.9911 / 6.9966 GiB，未持续大幅增长 |
| Decoder 参与训练 | 0次，仅验证解码 |
| latest 保存 | 模型、state、约6.78GB优化器状态均已写入 |
| 封存测试 | 未运行、未参与权重选择 |

首次启动的 inference-mode RoPE 缓存问题已修复为 no_grad 验证；此前没有任何优化更新。另一次加载 OOM 的日志保留在 results_archive/M9 中。失败启动权重已清理，VAE latent 缓存复用。

运行目录：`/share/linmingheng-local/xuke/RMagNet-M9/runs/m9_m4_latent_only_e4/`。
tmux：`m9-latent-only`。完整配置见 M9_LATENT_ONLY.md，机器记录见 M9_STARTUP_CHECK.json。

当前 PSNR/SSIM 是初始 M4 baseline，并非微调后结果；微调后每个 epoch 的验证会写入 metrics.jsonl 和 latest_metrics.json。
