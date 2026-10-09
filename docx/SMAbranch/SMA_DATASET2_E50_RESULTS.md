# SMA 合并数据 / 50 epoch 训练与测试结果

## 训练完成

50完整epoch、2550次更新，退出码0，无早停。北京时间2026-10-09 03:27完成；正式训练至最后更新约4小时37分钟（含中途验证，不含准备与启动加载）；显存峰值17.994GiB。仅保留best/latest权重。

best按26张验证集macro L1选择，位于epoch2/step102，同样是本轮验证PSNR/SSIM最高点。latest位于epoch50/step2550；其间48个epoch没有刷新best。

| 验证模型 | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| M4初始化 | 0.043888 | 24.0311 | 0.86901 | 0.11791 |
| SMA-best epoch2 | 0.042382 | 24.2359 | 0.87128 | 0.11963 |
| SMA-latest epoch50 | 0.045241 | 23.7721 | 0.86279 | 0.13095 |

训练重建项普通输入/P90从epoch1的0.07117/0.10932降至epoch50的0.04543/0.06182，而验证从早期最佳点回落，呈现过拟合迹象。重建项为L1 + 0.2(1-SSIM) + 0.1 edge，不能当作PSNR曲线。

## 测试口径

纠正标签后的同一26张测试图（旧17+新9），原处理尺寸、保存后8位RGB PNG、逐图macro均值；LPIPS使用SqueezeNet。推理仅输入普通I，GT只用于评分，不输入P90/DoLP/训练缓存。checkpoint按验证集选择，没有用本次测试重新选权重。

SMA使用VAE posterior mode，首次M4测试发现官方默认是posterior.sample。为公平比较，已用相同mode重算M4；sample版保存在test_m4best_sampled_vae，仅作补充，不进入主表。mode版M4旧17张与封存历史指标一致（误差小于1e-7）。

## 全26张

| 模型 | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| M4-best | 0.060430 | 24.2355 | 0.79966 | 0.15926 |
| SMA-best (epoch 2) | 0.061053 | 24.2529 | 0.79913 | 0.16057 |
| SMA-latest (epoch 50) | 0.055337 | 23.8461 | 0.80113 | 0.17229 |

## 原17张

| 模型 | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| M4-best | 0.046786 | 24.1575 | 0.83933 | 0.11132 |
| SMA-best (epoch 2) | 0.046382 | 24.2567 | 0.84101 | 0.11110 |
| SMA-latest (epoch 50) | 0.049802 | 23.5279 | 0.83243 | 0.11891 |

## 新9张

| 模型 | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| M4-best | 0.086201 | 24.3829 | 0.72471 | 0.24980 |
| SMA-best (epoch 2) | 0.088764 | 24.2457 | 0.72001 | 0.25403 |
| SMA-latest (epoch 50) | 0.065793 | 24.4472 | 0.74201 | 0.27312 |

## 分析

- 全26张SMA-best相对M4仅增加0.0174dB，SSIM变化-0.00053，LPIPS变化+0.00132；没有明确整体优势。
- 原17张best小幅改善；新9张best的PSNR、SSIM、LPIPS均逊于M4，改善没有稳定迁移到新场景。
- latest在新9张相对M4的PSNR/SSIM增加0.0643dB/0.01729，但LPIPS更差，旧17张退化明显。
- latest高变化区L1下降，但低变化区保持误差上升，说明恢复与保持存在取舍；仅凭这些结果不能确定某一损失是退化原因。
- 本轮为固定数据/种子的单次实验，旧17张已有多轮观察历史；小幅差值不能证明稳定泛化或统计显著性。

## 结果位置

基础目录：`/share/linmingheng-local/xuke/RMagNet/runs/sma_dataset2_e50`

- `test_comparison.json`：完整分组汇总与checkpoint/数据哈希。
- `test_comparison_per_image.csv`：逐图三模型对比。
- `epoch_summary.json`：50个epoch的训练与验证曲线数据。
- `test_best/validation/step_000000/predictions`：SMA-best结果。
- `test_latest/validation/step_000000/predictions`：SMA-latest结果。
- `test_m4best/validation/step_000000/predictions`：匹配编码的M4-best结果。
- 各模型`evaluation.json`保存精确汇总和身份，`validation/step_000000/metrics.csv`保存逐图指标。

以上子路径均相对于基础目录。step_000000是评估目录编号，SMA实际训练step分别为102和2550。本次仅运行推理评估，没有继续训练或删除权重。
