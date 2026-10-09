# C1 + M5 像素恢复实验结果

训练已完成：7 epoch / 357 次更新；连续4个epoch验证L1无提升早停。best为第3epoch，latest为第7epoch。两份M5权重的上游都是固定C1-best，不是C1-latest。

## 26张纠正标签后封存测试图

| 模型 | L1 ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|
| C1-best | 0.060906 | 24.1925 | 0.798305 |
| C1+M5-best | 0.060942 | 24.2244 | 0.798535 |
| C1+M5-latest | 0.061157 | 24.1771 | 0.799323 |
| M4-best(reference) | 0.060430 | 24.2355 | 0.799657 |

## 结论

- M5-best比C1的PSNR增加0.0319dB，SSIM增加0.000230；PSNR逐图胜出11/26，但L1略差，未超过M4-best的PSNR/SSIM。不能据此称为稳定的细节恢复提升。
- best的低变化区域保持误差改善，但高变化恢复误差和全图边缘误差略差：当前更像小幅像素校正，尚未显示对强反射/纹理的稳定修复。
- latest的SSIM比C1略高，但PSNR和L1均变差；继续相同训练并未产生一致收益。best依据验证L1选择，不根据测试集PSNR选权重。
- 对系数、容量、残差幅度是否限制能力，目前只有假设；此轮结果无法单独确定原因。先不新增训练或收益门控。

## 口径与位置

- 使用现有长宽比和处理尺寸，保存后的8位RGB PNG，逐图宏平均。SSIM沿用项目11×11均匀窗口口径。
- 本次C1基线用相同PNG重新计算，主指标与历史记录误差小于1e-6。低/高变化mask在本次统一的RGB8读取方式下重算，只用于本次C1与M5对照。
- M4为历史同测试集主指标参考，没有重新运行M4。real20评测已补充，见 [M5_C1_REAL20_RESULTS.md](M5_C1_REAL20_RESULTS.md)；没有重新训练。
- 该测试集已经多次用于研究，且历史初始化有场景重叠，仅属回顾性对照。
- best结果图：`runs/m5_c1_pixel_e30/eval_test_best/predictions/`。
- latest结果图：`runs/m5_c1_pixel_e30/eval_test_latest/predictions/`。
- 三张固定索引抽样对比图：`runs/m5_c1_pixel_e30/test_panels/`，包含Input / GT / C1-best / M5-best / M5-latest。
- 全部结果与逐图表：`runs/m5_c1_pixel_e30/TEST_REPORT.md`、`test_comparison.json`、`test_comparison_per_image.csv`。关键数值同步在本目录materials中。
