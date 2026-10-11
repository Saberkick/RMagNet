# M7 best 封存测试集评估

同一组 24 张封存测试图；M7 best 来自第3 epoch。M4 与 M7 同次生成、相同输入/GT/尺寸，模型只输入普通 I。
指标基于保存后的 RGB8 PNG，逐图宏平均；SSIM 沿用项目 uniform 11×11、zero padding 5。

| 模型 | 图数 | PSNR / dB | SSIM | L1 | 边缘 L1 |
|---|---:|---:|---:|---:|---:|
| M4 final best | 24 | 24.9000 | 0.801285 | 0.060469 | 0.011719 |
| M7 best | 24 | 24.9258 | 0.800218 | 0.060758 | 0.011722 |

M7 相对 M4：PSNR +0.0258 dB，SSIM -0.001067。
逐图 PSNR 胜出 10/24，SSIM 胜出 9/24。
低变化区 L1 差 -0.000368，高变化区 L1 差 +0.001056（误差下降为改善）。

预测位于 `m4/predictions/`、`m7/predictions/`，逐图指标位于对应 `metrics.csv`；模型与数据哈希见 `config.json`。
没有训练、修改训练缓存或依据封存测试选择 checkpoint。
