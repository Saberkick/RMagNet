# M3-withLrec：恢复基础重建项的 70 步对照

> 日期：2026-09-27  
> 分支：`experiment/m3-semantic-separation`

## 目的

在 M3-noLrec 的同一 Stage 2 初始化、纠正标签后的 M2 数据、离线语义缓存、样本顺序和 70 次更新预算上，只恢复设计文档规定的基础重建锚点，检验 `L_rec` 能否保留全局结构和低变化区域，同时保留语义分离对高变化反射区域的收益。

## 固定目标

\[
L=L_{rec}+0.25L_{cluster}+0.10L_{relation}+0.10L_{cons}+0.05L_{boundary}
\]

其中：

\[
L_{rec}=\frac{1}{2}[L_{base}(T_I,GT)+L_{base}(T_{90},GT)]
\]

\[
L_{base}=L_1+0.2(1-SSIM)+0.1L_{edge}
\]

`I` 和 `P90` 共用同一个 Transmission LoRA。Qwen 主干、VAE 和教师保持冻结；仅 LoRA_T 更新。该实验不加入旧的逐 token `L_Q20`、位置 weighted Charbonnier 或 low-response keep。

## 预算与输出

- GPU：4 张；每卡 batch 1；有效 batch 4。
- 初始化：Stage 2 best LoRA_T，固定 SHA-256 检查。
- 更新：70；学习率 `5e-6`；warmup 20 步。
- 验证和保存：step 35、70。
- 输出：`runs/m3_with_lrec_70/`。
- 控制台：`runs/m3_with_lrec_70.console.log`。
- PID：`runs/m3_with_lrec_70.pid`。

## 运行

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 \
  bash scripts/run_m3_with_lrec.sh \
  > runs/m3_with_lrec_70.console.log 2>&1 &
```
