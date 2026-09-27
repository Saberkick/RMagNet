# M3-noLrec：移除基础重建项的 70 步消融

> 状态：实现完成，等待/正在运行 70 步实验  
> 日期：2026-09-27  
> 分支：`experiment/m3-semantic-separation`

## 目的

直接从 M3 总目标中删除 `Lrec`，检验语义分离与偏振监督能否独立驱动 LoRA_T。对照为已经完成的 `M2-A-data70-noq20`：两者都从同一个 Stage 2 最佳权重开始，使用相同 M2 train split、样本顺序和 70 次 optimizer update。

## 唯一训练目标

\[
L_{noLrec}
=0.25L_{cluster}
+0.10L_{relation}
+0.10L_{cons}
+0.05L_{boundary}
\]

明确没有：

- 整图 `Lrec`；
- Stage 2 的 L1 + SSIM + edge 基础组合；
- M2-B 的直接逐 token `L_Q20`；
- M2-B 的旧位置 weighted Charbonnier 与 low-response keep。

`Lcluster` 仍然以 GT 为目标，所以该实验不是无 GT 训练，而是移除不分区域的整图基础重建项。

## 两路训练

- 普通输入：`T_I=F(I)`。
- 反射增强输入：`T_90=F(P90)`。
- 两路共用同一个 LoRA_T。
- `Lcluster` 与 `Lboundary` 对两路取平均。
- `Lrelation` 只约束部署时使用的 `T_I`。
- `Lcons` 同时把两路输出向彼此拉近。
- P90 只是训练期第二观测，不作为 Reflection GT。

## 显存实现

单卡不同时保留两路完整 DiT 图：

1. 冻结 VAE，确定性编码 `I/P90` latent。
2. 无梯度生成 `T_90` 参考。
3. 前向 `T_I`，冻结并关闭 LoRA 的 Qwen 教师运行到 block 20。
4. 先求损失对 `T_I` 的输出梯度，再恢复 LoRA，把输出梯度传回 LoRA_T。
5. 释放第一路图后前向 `T_90`，传回第二路梯度。
6. 两路梯度累加后只执行一次四卡同步和 optimizer step。

## 固定配置

| 项 | 值 |
|---|---:|
| 初始化 | Stage 2 best LoRA_T |
| GPU | 4 |
| 每卡 batch | 1 |
| 有效 batch | 4 |
| optimizer update | 70 |
| 学习率 | `5e-6` |
| warmup | 20步 |
| 验证/保存 | step 35、70 |
| 数据增强 | 与 M2-A 相同的固定种子水平翻转 |
| test split | 封存 |

## 运行

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OMP_NUM_THREADS=2 \
bash scripts/run_m3_nolrec.sh
```

后台运行：

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 \
  bash scripts/run_m3_nolrec.sh \
  > runs/m3_nolrec_70.console.log 2>&1 &
```

输出：

```text
runs/m3_nolrec_70/
```

日志必须逐步记录 `l_rec: null`、四个有效损失、I/P90 输出梯度、LoRA 梯度、学习率和显存。

## 启动记录

- 已在 2026-09-27 使用 GPU 0–3 后台启动正式 70 步任务。
- 后台 PID 文件：`runs/m3_nolrec_70.pid`。
- 控制台日志：`runs/m3_nolrec_70.console.log`。
- 训练目录：`runs/m3_nolrec_70/`。
- 离开 SSH 前已稳定运行至 step 12。
- `l_rec` 为 `null`，四项有效损失及 I/P90 输出梯度均为有限非零值。
- 每步活跃 LoRA 梯度张量为 1,442。
- step 12 时单卡峰值 allocated 约 21.96 GiB，reserved 约 23.25 GiB；前 12 步没有持续增长。
- GPU 0–3 均参与计算；离开前利用率均为 100%。
