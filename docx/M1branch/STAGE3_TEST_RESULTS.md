# Stage 3 测试程序与当前结果

日期：2026-09-21。

## 正式训练状态

检查时不存在 `runs/stage3_fusion_r8`，也没有 `stage3_train` 或 `torch.distributed.run` 进程。因此正式 Stage 3 训练没有启动，当前没有正式 best/final 权重可供质量验收。

完整候选缓存已经生成：`cache/stage3_candidates` 包含 53 张 T 和 53 张 R，以及权重哈希清单。训练前置数据已经准备完成。

未启动的原因是后台命令将输出重定向到尚不存在的 `runs/stage3_fusion_r8/nohup.log`。Shell 会先打开重定向目标，再执行 `run_stage3.sh`，所以脚本内部的 `mkdir` 没有机会运行。

修正后的启动命令：

```bash
cd /share/linmingheng-local/xuke/RMagNet
mkdir -p runs/stage3_fusion_r8
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 \
  bash scripts/run_stage3.sh \
  > runs/stage3_fusion_r8/nohup.log 2>&1 &
echo $!
```

## 测试程序

入口为 `scripts/eval_stage3.sh`，实现位于 `src/rmagnet/stage3_eval.py`。默认测试 11、12、17，并比较：

- 原始反射输入；
- Stage 2 缓存的 T；
- Stage 3 best；
- Stage 3 final；
- R 置零消融；
- R 跨样本置乱消融。

所有 PSNR/SSIM/L1 均基于保存后的 8 位 RGB PNG 计算。输出包含预测图、四倍误差热图、横向 panel、`metrics.csv`、`evaluation.json` 和 `REPORT.md`。

正式训练完成后的命令：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_stage3.sh
```

## Smoke checkpoint 测试

为了验证测试程序本身，使用只训练一个 optimizer step 的 `stage3_smoke_2gpu` checkpoint，在 ID 11 上执行测试。best 与 final 是同一份 step-1 权重。

| 变体 | L1 ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|
| 原始输入 | 0.087710 | 17.509 | 0.7458 |
| Stage 2 T | 0.071111 | 19.579 | 0.7535 |
| Stage 3 step-1 | 0.168839 | 12.773 | 0.4607 |
| Stage 3 step-1，R 置零 | 0.167745 | 12.832 | 0.4636 |

单步 Stage 3 明显低于 Stage 2，且 R 置零没有造成有效下降。这只说明随机初始化的 Fuse 尚未学习，不构成对正式训练效果或架构上限的判断。单样本无法进行 R 置乱消融，因此程序自动跳过该项。

测试产物位于 `runs/stage3_smoke_2gpu/evaluation`。

## 正式验收判据

正式训练完成后至少检查：

1. Stage 3 best 相对 Stage 2 T 的平均 PSNR/SSIM 是否提升；
2. 11、12、17 是否出现文字破坏、结构新增或内容幻觉；
3. R 置零及 R 置乱是否导致指标下降；
4. 若正常 R 与置乱 R 几乎相同，应判定融合器没有有效利用 R；
5. best 与 final 的差距是否显示后期过拟合。

---

# 正式训练与评估结果（2026-09-21）

此前“正式训练未启动”的状态已经解决。本次正式训练使用 GPU 1、2、3、4，完成 50 epochs、650 optimizer steps，正常生成 final checkpoint。单卡峰值 allocated 18.455 GiB、reserved 19.023 GiB。

训练期最佳 checkpoint 位于 step 400（epoch 30）：验证 L1 0.050158、PSNR 23.0377 dB、SSIM 0.8161。final step 650：验证 L1 0.050328、PSNR 23.0197 dB、SSIM 0.8116。best 优于 final，后半程没有继续提升验证集。

## 保存后 PNG 的正式比较

测试集为 11、12、17，以下指标均从保存后的 8 位 RGB PNG 重新计算。

| 变体 | L1 ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|
| 原始输入 | 0.064340 | 20.123 | 0.7670 |
| Stage 2 T | **0.048496** | **23.142** | **0.8215** |
| Stage 3 best | 0.050149 | 23.035 | 0.8157 |
| Stage 3 best，R 置零 | 0.049996 | 23.061 | 0.8155 |
| Stage 3 best，R 置乱 | 0.049571 | 23.102 | 0.8159 |
| Stage 3 final | 0.050341 | 23.015 | 0.8152 |
| Stage 3 final，R 置零 | 0.050450 | 23.013 | 0.8148 |
| Stage 3 final，R 置乱 | 0.049942 | 23.061 | 0.8152 |

Stage 3 best 相对 Stage 2 T：L1 增加 0.001653、PSNR 降低 0.107 dB、SSIM 降低 0.0058。当前融合阶段没有带来平均收益。

## 逐图 PSNR / SSIM

| ID | Stage 2 T | Stage 3 best | Stage 3 final |
|---|---:|---:|---:|
| 11 | 19.579 / 0.7535 | 19.598 / 0.7510 | 19.650 / 0.7516 |
| 12 | 22.717 / 0.8336 | 22.304 / 0.8218 | 22.278 / 0.8217 |
| 17 | 27.129 / 0.8774 | 27.202 / 0.8744 | 27.118 / 0.8723 |

11 和 17 的 PSNR 有小幅变化，但 SSIM 均下降；12 在 PSNR 和 SSIM 上均明显退化，是平均结果变差的主要来源。

## R 消融结论

R 置零和跨样本置乱没有造成稳定下降，置乱后的平均 PSNR 甚至略高于正常 R。这说明当前 `LatentFusionMixer + LoRA_Fuse` 没有学会可靠利用 Reflection 分支，不能据此宣称 R 反参考有效。

## 视觉检查

- 11：Stage 3 与 Stage 2 外观非常接近，没有明显新增物体；招牌和橱窗小字没有得到额外恢复。
- 12：Stage 3 延续并略加强了 Stage 2 的颜色/亮度变化，细节没有明显改善，与数值退化一致。
- 17：反射抑制基本来自 Stage 2；Stage 3 对“安全检查”文字和细线没有恢复优势，局部仍偏平滑。
- 三张图未观察到 WindowSeat 式明显新增灯具等大结构幻觉，但样本数量不足以形成稳健的幻觉率结论。

## 当前判定

Stage 3 工程链路已经完整跑通，但当前训练结果未超过 Stage 2，且 R 消融失败。下一轮不应直接扩大相同训练；应先增加显式的 R 使用约束或门控监督，并重新设计验证集与早停策略。

正式测试产物位于 `runs/stage3_fusion_r8/evaluation`。
