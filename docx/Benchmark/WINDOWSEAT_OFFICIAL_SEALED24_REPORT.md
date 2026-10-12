# 原生 WindowSeat：当前24张封存测试

官方发布权重，无个人微调。与M4/M9使用相同处理后输入、固定文本、NF4主干、BF16激活、posterior mode VAE编码、整图一步更新及保存后RGB8指标。
不重新缩放、裁剪、填充或切块；GT只用于评价。本轮目标是统一模型对照，不是复现官方CLI切块和随机采样协议。

| 模型 | PSNR / dB | SSIM | 相对原生ΔPSNR | 相对原生ΔSSIM |
|---|---:|---:|---:|---:|
| WindowSeat official | 24.3425 | 0.786663 | +0.0000 | +0.000000 |
| M4 final best | 24.8994 | 0.801283 | +0.5569 | +0.014620 |
| M9 best (M4 initialization) | 24.8994 | 0.801283 | +0.5569 | +0.014620 |
| M9 latest epoch4 | 24.7810 | 0.794732 | +0.4385 | +0.008068 |

本表覆盖同一24张，包括现有清单中的55_1672_1162；不使用之前历史9张或12张过滤子集的均值。

全部24张预测PNG：predictions/。逐图metrics.csv；权重、数据、输入GT输出哈希和协议见config.json、metrics.json。

结果目录：`/share/linmingheng-local/xuke/RMagNet-M9/results_archive/WindowSeat/windowseat_official_sealed24_mode`。M4/M9为前次已完成的同协议结果，本次没有重新训练或重测这些模型。
