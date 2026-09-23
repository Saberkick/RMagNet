# C1-L20：10 Epoch 正式训练总结

## 1. 结论

正式运行 `c1_l20_e10` 已完整训练 **10 epoch / 70 次优化更新**，`training_complete=true`，未触发 early stopping。第 10 epoch 同时是本次运行的最佳与最终 checkpoint：

- 验证 L1：`0.047295`
- 验证 PSNR：`23.480005 dB`
- 验证 SSIM：`0.825119`
- 验证图片：`11 / 12 / 17`
- 训练分辨率：`512×384`
- 训练提交：`1bebeb992f26ff80629e89805227ed360e16a55d`

相对 Stage 2 最佳初始化点的浮点验证均值（L1 `0.048505`、PSNR `23.142249 dB`、SSIM `0.821929`），C1-L20 的变化为：

- L1：`-0.001210`
- PSNR：`+0.337755 dB`
- SSIM：`+0.003190`

这说明 C1-L20 在当前三张固定验证图上取得了明确的平均收益，但仍属于小验证集上的方向性结果，不能直接当作跨场景泛化结论。

## 2. 正式训练配置

- 初始化：`runs/stage2_transmission_r128/best_transmission_lora.safetensors`
- 训练集：50 张，索引为 `13–16、18–59、61、64、65、66`
- 验证集：`11、12、17`
- GPU：4×RTX 3090
- 每卡 batch：1
- 梯度累积：2
- 有效 batch：8
- 每 epoch：约 7 次更新
- 学习率：`5e-5`
- 调度：10 步 warmup + cosine
- 优化器：`PagedAdamW8bit`
- 精度：BF16
- 主干、VAE、教师 Qwen：冻结
- 可训练参数：Transmission LoRA，共 1442 个有梯度张量
- Qwen 监督层：block 20（代码索引 19）
- Q20 GT 缓存：BF16，形状 `768×3072`
- 推理输入：仅输入待处理图 `I`，不需要 GT 或 DoLP

训练目标为：

```text
L = L_base
    + 0.25 * L_weighted_charbonnier
    + 0.10 * L_low_response_keep
    + lambda_q * L_Q20
```

离线空间权重采用：

```text
S = D_Q * (0.7 + 0.3 * D_DoLP)
W = clip(1 + 2*S, 1, 3)，随后逐图归一化到均值 1
```

## 3. 训练曲线

| Epoch | L1 ↓ | PSNR ↑ | SSIM ↑ |
|---:|---:|---:|---:|
| 1 | 0.049226 | 23.098609 | 0.821828 |
| 3 | 0.047938 | 23.267325 | 0.823837 |
| 5 | 0.047450 | 23.398593 | 0.824924 |
| 7 | 0.047277 | 23.451829 | 0.825048 |
| 8 | 0.047407 | 23.462287 | 0.825050 |
| 9 | 0.047318 | 23.472918 | 0.825078 |
| **10** | **0.047295** | **23.480005** | **0.825119** |

第 3 至第 10 epoch 的总体趋势稳定向上，没有出现后期明显退化。第 9 到第 10 epoch 只增加约 `0.0071 dB`，曲线已接近平台，因此当前结果适合作为首轮 C1-L20 checkpoint，不建议在没有扩大验证集前仅靠追加 epoch 推断还能稳定获益。

## 4. 第 10 Epoch 逐图结果

| ID | L1 ↓ | PSNR ↑ | SSIM ↑ | 相对 Stage 2 的主要变化 |
|---:|---:|---:|---:|---|
| 11 | 0.069182 | 19.858700 | 0.755946 | PSNR 约 +0.27 dB，SSIM 约 +0.0023 |
| 12 | 0.048179 | 22.677639 | 0.834201 | PSNR 约 -0.04 dB，SSIM 约 +0.0007 |
| 17 | 0.024524 | 27.903675 | 0.885210 | PSNR 约 +0.78 dB，SSIM 约 +0.0076 |

逐图相对值使用现有 Stage 2 保存后 PNG 评价表作为参照，而 C1-L20 验证指标在保存 PNG 前对浮点张量计算，因此这些逐图差值只用于观察趋势。正式均值比较使用 Stage 2 的浮点 `best_metrics.json`。

## 5. 表现较好的可能原因

### 5.1 从成熟的 Stage 2 Transmission LoRA 开始

本轮没有从随机 LoRA 开始，而是在已经达到约 `23.14 dB` 的 Stage 2 最佳权重上做小步优化。主干和 VAE 全程冻结，降低了 50 张训练图条件下破坏原有生成先验的风险。C1-L20 更像是对已有去反射能力做局部定向校正。

### 5.2 Qwen 差异图承担主要定位，DoLP 只做温和增强

`D_Q` 由同一个冻结 Qwen 第 20 层比较 `I` 与 `GT` 得到，先确定训练时值得重点修正的位置；DoLP 只通过 `0.7 + 0.3D_DoLP` 调制强度。这样即使真实反射区域的 DoLP 不高，Qwen 差异仍能保留至少 70% 的权重，不会像硬蒙版那样把区域直接切成 0/1。

这与前面的 M1b 结论一致：单纯提高硬 DoLP 蒙版语义项强度没有稳定收益；C1-L20 改善的是监督位置与权重形式，而不只是放大同一个损失。

### 5.3 基础、局部、保持和语义约束同时存在

- `L_base` 保持像素、结构和边缘的全局质量。
- 加权 Charbonnier 将更多优化容量放在 Qwen 差异明显、且可能有偏振反射的区域。
- 低响应保持项约束模型不要无依据修改低响应区域。
- Q20 特征项直接约束高层表示，补充纯像素损失对文字、结构和细纹理表达不足的问题。

这组目标比单一的整图 L1/SSIM 更符合“重点修改反射区域，同时保留其他内容”的任务结构。

### 5.4 训练预算较保守，调度连续

总计只有 70 次更新，并使用 warmup、cosine、梯度裁剪和固定 Stage 2 初始化。每个 epoch 虽然重建 CUDA 进程，但完整恢复 LoRA、Adam、调度器和随机数状态，训练曲线连续。较小预算降低了在 50 张训练图上快速过拟合或产生大幅幻觉改写的概率。

### 5.5 收益不是所有图片均匀产生

平均提升主要由 `17` 的明显改善和 `11` 的中等改善贡献；`12` 的 PSNR 轻微下降。这说明新监督对某些反射形态更有效，不能解释为模型在所有场景上等比例增强。后续判断泛化时应扩大验证集，并单独检查文字、细纹理和低 DoLP 反射区域。

## 6. 已知限制

1. 验证集只有三张图，并且每个 epoch 都用于观察和选择最佳权重，结果可能受到验证集适配影响。
2. 当前验证代码在保存 PNG 前对浮点输出计算 PSNR/SSIM；若要与外部模型严格比较，应统一对保存后的 8 位 PNG 重算。本文没有启动额外评测。
3. 当前 `lambda_q` 受到下限 `0.02` 限制。首个测量点的实际 Q20/基础输出梯度比例约为 `1.945`，高于设计目标 15%–20%。这说明 Q20 监督确实很强，可能参与了收益，也带来语义项过强和局部改写的风险，不能把本轮提升全部归因于空间权重公式。
4. 第 10 epoch 虽为最佳，但相对第 9 epoch 的增益很小；现有证据不支持直接继续增加训练轮数。

## 7. 结果位置

运行根目录：

```text
/share/linmingheng-local/xuke/RMagNet/runs/c1_l20/c1_l20_e10
```

推荐推理权重（第 10 epoch，本次最佳）：

```text
runs/c1_l20/c1_l20_e10/checkpoints/best/transmission_lora.safetensors
```

最终权重与可恢复训练状态：

```text
runs/c1_l20/c1_l20_e10/checkpoints/last/transmission_lora.safetensors
runs/c1_l20/c1_l20_e10/checkpoints/last/trainer_state.pt
runs/c1_l20/c1_l20_e10/checkpoints/last/resume_meta.json
```

保留的最佳三个 epoch：

```text
runs/c1_l20/c1_l20_e10/checkpoints/epoch_0008.safetensors
runs/c1_l20/c1_l20_e10/checkpoints/epoch_0009.safetensors
runs/c1_l20/c1_l20_e10/checkpoints/epoch_0010.safetensors
```

第 10 epoch 生成结果：

```text
runs/c1_l20/c1_l20_e10/validation/predictions/epoch_0010/{11,12,17}.png
runs/c1_l20/c1_l20_e10/validation/error_maps/epoch_0010/{11,12,17}.png
```

完整指标与日志：

```text
runs/c1_l20/c1_l20_e10/validation/metrics.csv
runs/c1_l20/c1_l20_e10/logs/validation.jsonl
runs/c1_l20/c1_l20_e10/logs/train.jsonl
runs/c1_l20/c1_l20_e10/logs/console.log
runs/c1_l20/c1_l20_e10/config.yaml
runs/c1_l20/c1_l20_e10/checkpoints/top_epochs.json
```

本次总结只整理现有训练产物，没有启动新训练或额外评测。
