# AfterFullM2-B1 完整训练实验总结

## 1. 实验目的

AfterFullM2-B1 将纠正标签后的 M2-B1 从 70 步方向性实验扩展为按完整 epoch 训练的实验，用于回答两个问题：

1. 严格控制 Q20 输出梯度占基础损失梯度约 30% 时，增加训练量能否继续改善模型；
2. 验证集选出的权重能否在 18 张封存测试集上稳定超过 M4-best。

本实验从 Stage 2 最佳 Transmission LoRA 初始化，不从此前 70 步 M2-B1 权重继续训练。

## 2. 模型与监督来源

模型沿用 WindowSeat 的冻结 Qwen-Image-Edit DiT、冻结 VAE 和可训练 Transmission LoRA。训练输入为纠正标签后的普通反射图，目标为对应 GT；P90 不参与本实验。

Q20 教师监督固定为 Qwen 第 20 个 block，即代码索引 19，flow timestep 为 499。原始 `data_cache/m2a_q20` 曾被清理，因此本实验从兼容的基础教师缓存 `data_cache/m4_multilayer_v1` 复用了 `q20_input` 和 `q20_gt`。该缓存满足：

- 教师为冻结的基础 Qwen，所有 LoRA 关闭；
- 数据集清单 SHA-256 与纠正标签后的 M2 完全一致；
- 144 个训练样本及顺序一致；
- block、timestep、特征维度和图像尺寸一致。

随后离线重新计算 M2-B1 所需的 Q20 差异与 DoLP 权重，没有使用 M4-best LoRA 提取的特征。

Q20 差异图为：

$$
D_Q(x)=\operatorname{RobustNorm}_{2\%,98\%}\left(1-\cos\left(Q_{20}(I)_x,Q_{20}(GT)_x\right)\right)
$$

DoLP 只对 Q20 差异提供最多 30% 的空间调制：

$$
S(x)=D_Q(x)\left(0.7+0.3D_{DoLP}(x)\right)
$$

原始权重及最终均值归一化权重为：

$$
W_{raw}(x)=\operatorname{clip}\left(1+2S(x),1,3\right)
$$

$$
W(x)=\frac{W_{raw}(x)}{\operatorname{mean}(W_{raw})}
$$

## 3. 损失配置

基础重建损失为：

$$
L_{base}=L_1+0.2L_{SSIM}+0.1L_{edge}
$$

其中：

$$
L_{SSIM}=1-SSIM(\hat T,GT)
$$

完整损失为：

$$
L=L_{base}+0.25L_{weighted\_charbonnier}+0.10L_{low\_response\_keep}+\lambda_Q L_{Q20}
$$

Q20 项在每次更新中根据预测图输出梯度动态计算系数：

$$
\lambda_Q=\operatorname{clip}\left(0.30\frac{\lVert\nabla_{\hat T}L_{base}\rVert_2}{\lVert\nabla_{\hat T}L_{Q20}\rVert_2},0,0.5\right)
$$

因此实际比例满足：

$$
\frac{\lVert\lambda_Q\nabla_{\hat T}L_{Q20}\rVert_2}{\lVert\nabla_{\hat T}L_{base}\rVert_2}\approx0.30
$$

训练日志中该比例始终为 0.3000；最后一步的 `lambda_q` 为 0.006624。这说明 Q20 原始梯度明显大于基础项，需要很小的系数才能达到 30% 的实际影响。

## 4. 训练配置

| 配置 | 数值 |
|---|---:|
| 训练样本 | 144 |
| 验证样本 | 18 |
| 封存测试样本 | 18 |
| GPU | 4 × RTX 3090 |
| 每卡 batch | 1 |
| 有效 batch | 4 |
| 每 epoch 更新 | 36 |
| 最大 epoch | 30 |
| 实际完成 epoch | 12 |
| 实际更新 | 432 |
| 学习率 | 5e-6 |
| Warmup | 20 updates |
| 调度器 | Cosine |
| 优化器 | PagedAdamW 8-bit |
| Weight decay | 0.01 |
| 梯度裁剪 | 1.0 |
| 精度 | BF16 / NF4 Qwen |
| 早停 | 验证 L1 连续 4 epoch 未改善 |
| 数据增强 | RGB、权重图及 Q20 token 同步水平翻转 |

训练未出现 NaN、OOM、NCCL 错误或冻结参数收到梯度的问题。

## 5. 验证曲线

所有验证指标均从保存后的 8 位 RGB PNG 重新读取并按图片宏平均。

| Epoch | Step | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 高变化区 L1 ↓ | 验证 L1 是否刷新 |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 36 | 0.044909 | 23.8877 | 0.844704 | 0.112220 | 0.096577 | 是 |
| 2 | 72 | 0.043948 | 24.0110 | 0.846901 | 0.113358 | 0.093582 | 是 |
| 3 | 108 | 0.043697 | 24.0574 | 0.847520 | 0.113557 | 0.090217 | 是 |
| 4 | 144 | 0.043337 | 24.0918 | 0.848555 | 0.111740 | 0.088294 | 是 |
| 5 | 180 | 0.042520 | 24.1852 | 0.849711 | 0.110888 | 0.086318 | 是 |
| 6 | 216 | 0.041811 | 24.2638 | 0.850863 | 0.109816 | 0.082957 | 是 |
| 7 | 252 | 0.041047 | 24.2231 | 0.852655 | 0.108428 | 0.082009 | 是 |
| 8 | 288 | **0.040886** | **24.3154** | 0.853048 | 0.108228 | 0.080449 | 是 |
| 9 | 324 | 0.041476 | 24.1775 | 0.852174 | 0.108524 | **0.079361** | 否，1/4 |
| 10 | 360 | 0.041093 | 24.1848 | 0.852928 | 0.107173 | 0.081591 | 否，2/4 |
| 11 | 396 | 0.041305 | 24.2226 | 0.853314 | **0.106512** | 0.081225 | 否，3/4 |
| 12 | 432 | 0.041014 | 24.2006 | **0.853461** | 0.106865 | 0.080514 | 否，4/4 |

按预设规则，epoch 8 之后验证 L1 连续四轮没有刷新，训练在 epoch 12 自动结束。各指标的最优 epoch 并不一致：L1 和 PSNR 在 epoch 8 最好，LPIPS 在 epoch 11 最好，SSIM 在 epoch 12 最好，高变化区域 L1 在 epoch 9 最好。

## 6. 封存测试结果

封存测试使用纠正标签后的固定 18 张测试图，所有指标同样基于保存后的 8 位 PNG。

| 模型 | L1 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 低变化区 L1 ↓ | 高变化区 L1 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| AfterFull epoch 8 | 0.048931 | 23.7819 | 0.828910 | 0.113414 | 0.025993 | 0.091288 |
| AfterFull epoch 12 | **0.047578** | **24.0239** | **0.832118** | **0.109632** | **0.023330** | 0.091938 |
| M4-best | 0.047042 | 24.0414 | 0.833084 | 0.112665 | 0.024219 | **0.090278** |
| 原 70 步 M2-B | 0.048769 | **24.1618** | 0.829621 | 0.114754 | 0.023105 | 0.094873 |
| 原 70 步 M2-B1 | 0.048828 | 24.1550 | 0.829759 | 0.115080 | 0.023095 | 0.095128 |

虽然 epoch 8 具有最好的验证 L1，epoch 12 在封存测试集上明显更好：相对 epoch 8，PSNR 提高 0.2420 dB，SSIM 提高 0.003321，LPIPS 降低 0.003782。逐图比较中，epoch 12 在 11/18 张图的 PSNR、15/18 张图的 SSIM 和 16/18 张图的 LPIPS 上胜出。

epoch 12 与 M4-best 基本持平：PSNR 仅低 0.0175 dB，SSIM 低 0.000966；同时 LPIPS 好 0.003033，低变化区域 L1 好 0.000889，但高变化区域 L1 差 0.001660。因此它改善了感知质量和纹理保持，仍未全面超过 M4-best 的整体像素质量与强变化区域恢复。

原 70 步 M2-B 的 PSNR 仍最高。这表明延长 M2-B1 训练主要换来了 SSIM、LPIPS、低变化区域保护及高变化区域恢复的综合改善，并没有继续提高封存 PSNR。

## 7. 主要结论

1. **训练量有效，但收益改变了方向。** 从 70 步扩展到 432 步后，模型的感知指标、SSIM 和区域误差明显改善，PSNR却低于短训练 M2-B/M2-B1。
2. **验证 L1 不适合单独选择最终权重。** epoch 8 是验证 L1 最优点，但 epoch 12 的封存泛化显著更好。后续可在不接触封存测试集的前提下，使用验证集 L1、SSIM 和 LPIPS 的预注册组合分数。
3. **30% Q20 梯度控制工作正常。** 训练稳定且比例精确，但它不能保证所有指标共同提升；Q20 监督更偏向感知与结构约束。
4. **与 M4-best 的差距已经很小。** epoch 12 在 LPIPS 和低变化区保护上超过 M4-best，在 PSNR、SSIM 和高变化区恢复上仍略差。
5. **最终保留 epoch 12。** 用户决定删除 epoch 8 权重，后续推理以 latest 为准。

## 8. 产物与清理状态

当前保留的模型权重：

```text
runs/AfterFullM2-B1/latest_transmission_lora.safetensors
```

其 SHA-256 为：

```text
f92da6139e7dde94fe9ad9e3874c09b54eb4370f101571a4a05f683b90830f90
```

已删除的 epoch 8 权重：

```text
runs/AfterFullM2-B1/best_transmission_lora.safetensors
```

删除前 SHA-256：

```text
1edbe0ba8e1fd1efa0b0482e8f1717eed06d491c52e2b521e8494b86100aa055a
```

其历史验证指标与封存测试结果仍保存在 JSON、CSV 和预测图中，因此删除权重不影响本报告的可追溯性。

关键材料：

```text
runs/AfterFullM2-B1/metrics.jsonl
runs/AfterFullM2-B1/training_summary.json
runs/AfterFullM2-B1/best_metrics.json
runs/AfterFullM2-B1/sealed_test_best_latest/metrics.csv
runs/AfterFullM2-B1/sealed_test_best_latest/summary.json
runs/AfterFullM2-B1/sealed_test_best_latest/predictions/
```
