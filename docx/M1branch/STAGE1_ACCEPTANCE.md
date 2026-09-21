# Stage 1 验收报告

## 结论

| 验收层级 | 结论 |
|---|---|
| 训练工程链路 | **通过** |
| 对 90° 目标的拟合 | **部分通过** |
| `LoRA_R` 作为空间反射专家 | **未通过** |

Stage 1 已正常完成 100 epochs，checkpoint、最佳权重、自动恢复和八卡/四卡训练链路均可用。但目前的 `LoRA_R` 只比复制输入的 identity baseline 略好，并且不如逐图、逐通道的仿射亮度基线。现有证据更符合“模型学会了部分色调映射”，不足以证明它提取了可供 Fuse 使用的空间反射层。

因此保留 step 500 最佳权重作为 Stage 1 基线，暂不把它标记为已经合格的 Reflection expert。

## 运行与权重

- 数据：`datasets/rmagnet_stage1_512x384`
- 验收样本：11、12、17
- 最佳权重：`runs/stage1_reflection_r8/best_reflection_lora.safetensors`
- 最佳权重 SHA-256：`19e3d1e67dfc55f6ada233e6c25ba74dd07fd9c49ec235fd4f0dca2e061e73f7`
- 最佳训练位置：step 500，epoch 64
- 最终权重：`checkpoint-0000958/reflection_lora.safetensors`
- 验收输出：`runs/stage1_reflection_r8/acceptance`

训练中途从 8 卡切换到 4 卡。最终 `run_config.json` 的计划是每 epoch 13 steps、总计 1300 steps，但从已有八卡 checkpoint 恢复后在 epoch 100、step 958 结束，最终学习率为 `1.66e-5`。训练按 epoch 完成，cosine schedule 没有走到零。最佳权重出现在 step 500，因此验收使用最佳权重，不使用最终权重。

## 评价口径

所有指标重新从保存后的 8-bit RGB PNG 数值计算：

- `identity`：直接令预测等于输入 45° 图；
- `oracle_affine`：使用 GT 为每张图、每个颜色通道拟合 `aI+b`。它不是可部署方法，只用于判断任务能否由简单亮度/颜色变化解释；
- `best`：step 500 最佳 LoRA；
- `final`：step 958 最终 LoRA。

另记录：

- `delta_l1_from_input`：预测相对输入改动多少；
- `delta_cosine_to_target_change`：预测变化 `prediction-I` 与真实变化 `R90-I45` 的方向一致程度。

## 三张验证图平均结果

| 方法 | L1 ↓ | PSNR ↑ | SSIM ↑ | 相对输入改动 | 变化方向 cosine ↑ |
|---|---:|---:|---:|---:|---:|
| Identity | 0.05679 | 22.139 | **0.7696** | 0 | 0 |
| Oracle affine | **0.05225** | **23.071** | **0.7739** | 0.02947 | **0.4203** |
| Best LoRA, step 500 | 0.05337 | 22.674 | 0.7684 | 0.03006 | 0.3716 |
| Final LoRA, step 958 | 0.05353 | 22.548 | 0.7677 | 0.03200 | 0.3760 |

Best LoRA 相对 identity：

- L1 改善 0.00341；
- PSNR 提升 0.535 dB；
- SSIM 下降 0.00120。

Best LoRA 相对 oracle affine：

- L1 差 0.00113；
- PSNR 低 0.397 dB；
- SSIM 低 0.00548。

## 逐图结果

| ID | 方法 | L1 ↓ | PSNR ↑ | SSIM ↑ |
|---|---|---:|---:|---:|
| 11 | Identity | **0.04160** | **23.597** | **0.9082** |
| 11 | Best LoRA | 0.04381 | 23.470 | 0.8808 |
| 12 | Identity | 0.08690 | 17.648 | 0.5212 |
| 12 | Best LoRA | **0.08335** | **18.056** | **0.5424** |
| 17 | Identity | 0.04186 | 25.171 | 0.8793 |
| 17 | Best LoRA | **0.03296** | **26.496** | **0.8819** |

模型在 12、17 上有效，在 11 上三个指标全部退化。三张验证图太少，当前均值也可能受单个场景影响。

## 视觉检查

每张 panel 从左到右为：45° 输入、90° 目标、最佳预测、最终预测、最佳预测绝对误差 ×4。

- 11：预测引入明显的全局色调变化，但没有更准确地还原目标；细密文字和玻璃边缘误差突出。
- 12：玻璃和植被区域接近目标一些，SSIM 有改善；主要变化仍接近整体曝光与颜色调整。
- 17：数值改善最明显，预测变化方向与目标较一致；细线、文字和玻璃结构仍是主要误差区。
- 三张图的高误差都集中在高频边缘、文字、玻璃框和反射纹理处，正是后续 Fuse 需要的空间信息。

## 原因判断

当前损失由整图像素主导，而 `I45` 与 `R90` 的大部分区域相似。即使目标定义保持为 `R90`，网络也可以通过复制输入并调整整体色调获得较低损失。普通 L1、SSIM 和边缘损失没有给予偏振变化区域足够权重。

## Stage 1b 修正建议

保持用户定义的 `90°=R`，不更换监督图片。改变训练目标的权重方式：

1. 构造变化强度：`D = mean(abs(R90 - I45))`。
2. 使用 `W = 1 + λ·normalize(D)` 对 RGB reconstruction loss 加权，让反射变化区域主导梯度。
3. 显式监督残差：`Δpred = Rpred-I45`、`Δtarget = R90-I45`。
4. 给残差加入 gradient/高频损失，避免只学习逐通道仿射变换。
5. 加入 affine baseline margin：验证时要求 LoRA 至少在多数图上超过 oracle affine，而不只超过 identity。
6. 将验证集扩大到至少 10 个独立场景；三张图保留为固定可视化样本。

建议从 step 500 权重继续做短程 Stage 1b，也同时保留从随机 LoRA 初始化的对照实验。Stage 1b 通过后再进入 T/Fuse 主线，避免 Fusion 把一个主要编码色调变化的 R 分支当作反射证据。

## 复现

```bash
cd /share/linmingheng-local/xuke/RMagNet
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_stage1.sh
```

输出：

- `acceptance.json`：均值、差值、权重哈希；
- `metrics.csv`：逐图、逐方法指标；
- `{id}_best.png` / `{id}_final.png`：两个 checkpoint 的输出；
- `{id}_best_error_x4.png`：误差热图；
- `{id}_panel.png`：验收拼图。
