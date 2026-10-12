# M9 latent-only 四 epoch：验证与封存测试结果

从 M4 final best 小幅微调已有编辑 LoRA，训练只用归一化最终 latent L1，学习率1e-5。主干和VAE冻结。

训练完成4 epoch / 136次更新，正常退出。没有使用封存测试选择权重。

## 25张验证集

| Epoch | latent L1 | PSNR / dB | SSIM |
|---:|---:|---:|---:|
| 0 | 0.172884 | 24.0578 | 0.875925 |
| 1 | 0.104530 | 23.9496 | 0.865090 |
| 2 | 0.100090 | 23.9900 | 0.869740 |
| 3 | 0.098624 | 24.0235 | 0.869206 |
| 4 | 0.098483 | 24.0035 | 0.868087 |

## 24张封存测试集

| 模型 | PSNR / dB | SSIM | L1 | Edge L1 |
|---|---:|---:|---:|---:|
| M4 final best | 24.8994 | 0.801283 | 0.060470 | 0.011720 |
| M9 best, epoch 0 | 24.8994 | 0.801283 | 0.060470 | 0.011720 |
| M9 latest, epoch 4 | 24.7810 | 0.794732 | 0.061148 | 0.012831 |

所有模型使用同一24张清单、同一推理实现、原有动态尺寸；保存RGB8 PNG后沿用项目指标计算逐图算术均值。SSIM为RGB、11×11均匀窗口、边界零填充5。

best 是否逐张量等于 M4：True。若相等，best复用本轮M4预测；其metrics.csv和metrics.json独立保存并明确标记预测位置。

- M9 best, epoch 0 相对 M4：PSNR +0.0000 dB；SSIM +0.000000；逐图PSNR胜出 0/24，SSIM胜出 0/24。
- M9 latest, epoch 4 相对 M4：PSNR -0.1184 dB；SSIM -0.006552；逐图PSNR胜出 11/24，SSIM胜出 4/24。

## 解读边界

latent L1下降与最终RGB图像质量提升是不同结果，需同时查看PSNR/SSIM、边缘误差及局部图像。单次4 epoch实验不能证明所有latent监督都无效，也不能证明VAE是唯一瓶颈。

结果目录：`/share/linmingheng-local/xuke/RMagNet-M9/results_archive/M9/m9_latent_only_e4_sealed_test`。各模型逐图表见子目录metrics.csv，固定权重身份见config.json。

## 本轮结论

验证 latent L1下降 43.03%，但验证和封存测试的PSNR/SSIM均未超过M4。封存latest的边缘L1增加 9.48%，PSNR胜出11/24，SSIM胜出4/24。

因此，这次小学习率、4 epoch的纯latent监督没有改善平均RGB恢复质量；best保留epoch0初始化权重。该结果支持当前latent距离与最终图像评价目标存在不一致。它不能单独证明VAE是唯一原因，也不能证明DiT容量已经饱和。

可能机制包括：GT编码latent并非Decoder输出最接近真实GT的唯一或最优latent；Decoder非线性使等权latent误差与RGB误差的方向、尺度不同；latent差异混合颜色/曝光/结构，未被验证为纯语义目标。这些是解释假设，并非本轮分别验证的因果结论。

没有启动额外训练或real20评估，未按封存测试重新选择权重。
