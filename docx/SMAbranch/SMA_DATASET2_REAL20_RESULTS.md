# real20：合并数据 SMA-best / SMA-latest 与 M4-best

- SMA 来自 `runs/sma_dataset2_e50`：best 为 epoch 2 / step 102，latest 为 epoch 50 / step 2550。
- 三个模型均采用确定性 VAE posterior mode、种子 2026、官方短边分块和 Lanczos 拼接。
- 对原始尺寸的 20 张保存后 RGB 8 位 PNG 与 GT JPEG 计算逐图指标，再取宏平均。
- 测试时只输入普通图 I；GT 仅评分，不输入 P90、DoLP 或训练语义缓存。

| 模型 | L1 ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|
| M4-best | 0.040766 | 25.5068 | 0.819492 |
| SMA-best (epoch 2) | 0.041321 | 25.3860 | 0.818244 |
| SMA-latest (epoch 50) | 0.047496 | 24.3799 | 0.806105 |

## 相对同口径 M4-best

- SMA-best (epoch 2)：PSNR -0.1208 dB，SSIM -0.001248，L1 +0.000556；逐图 PSNR 胜出 7/20，SSIM 胜出 6/20。
- SMA-latest (epoch 50)：PSNR -1.1269 dB，SSIM -0.013387，L1 +0.006730；逐图 PSNR 胜出 4/20，SSIM 胜出 6/20。

## 结论

SMA-best 相对同口径 M4-best 的平均 PSNR 下降 0.1208 dB、SSIM 下降 0.001248；20 张中有 7 张 PSNR 更高。SMA-latest 平均 PSNR 下降 1.1269 dB、SSIM 下降 0.013387，仅 4 张 PSNR 更高。
本轮未显示 real20 上的整体泛化收益。训练到 50 epoch 的版本退化更明显，与此前合并封存测试的退化方向一致；这不能单独确定是 SMA 结构、训练数据分布还是训练时长造成。
SSIM 沿用项目原 real20 实现：RGB [0,1]，11×11 均值窗口、边界补零，C1=0.01²、C2=0.03²。并非另换一种 SSIM 实现。

## 历史口径说明

本次 M4-best 重新推理，VAE 改用与 SMA 相同的 posterior mode。历史 `runs/real20_windowseat_m4` 使用 posterior sample，不能将两次差异完全归因于 SMA。历史 WindowSeat / M0 / M4 表仍保留在原目录，未覆盖。

## 文件

- 服务器根目录：`/share/linmingheng-local/xuke/RMagNet/runs/real20_sma_dataset2_e50/`。
- `best/`、`latest/`、`m4best/`：20 张预测 PNG、逐图 `metrics.csv`、含权重和数据哈希的 `evaluation.json`。
- `summary.json`：宏平均、差值、逐图胜出数与模型身份。
- `metrics.csv`：三个模型的 60 行逐图指标。
- `panels/`：20 张 Input / GT / M4-best / SMA-best / SMA-latest 五列对比图，图上标注 PSNR/SSIM。
- 启动脚本：`bash scripts/eval_sma_real20.sh`；已有输出时拒绝覆盖。
