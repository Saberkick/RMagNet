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
