# M8 best/latest 封存测试

同一组24张纠正标签后的封存图。best由验证集L1选定为epoch2，latest为epoch5；均在读取本轮封存结果前固定。
输入仅普通I，保留现有动态尺寸；GT只用于评价。冻结M4、NF4主干、BF16激活、FP32门控；无训练、无教师缓存。
M4、best、latest在同一模型/同次任务下生成；模型输出保存成RGB8 PNG后统一计算逐图宏平均。SSIM为项目uniform11×11、zero-pad5。

| 模型 | 图数 | PSNR / dB | SSIM | L1 | 边缘L1 |
|---|---:|---:|---:|---:|---:|
| M4 final best | 24 | 24.8994 | 0.801283 | 0.060470 | 0.011720 |
| M8 best, epoch2 | 24 | 24.8926 | 0.801245 | 0.060524 | 0.011719 |
| M8 latest, epoch5 | 24 | 24.8975 | 0.801451 | 0.060496 | 0.011715 |

- M8 best, epoch2：PSNR -0.0067 dB、SSIM -0.000038；逐图PSNR胜出 11/24，SSIM胜出 13/24。
- M8 latest, epoch5：PSNR -0.0019 dB、SSIM +0.000168；逐图PSNR胜出 11/24，SSIM胜出 13/24。

所有模型均保留24张PNG、逐图metrics.csv；best/latest还记录门控统计和固定前三张灰度门控图。
运行目录：`/share/linmingheng-local/xuke/RMagNet-M8/runs/m8_m4_gates_14_37_54_e5/sealed_test_best_latest`。预测位于`m4/predictions/`、`best/predictions/`、`latest/predictions/`；身份与哈希见config.json。
本轮未根据封存测试结果更换checkpoint，未运行real20。
