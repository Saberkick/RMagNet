# C1-L20 损失函数详细定义

本文对应正式训练实现 `src/rmagnet/c1_l20_train.py`，并引用 `src/rmagnet/stage2_train.py` 与 `src/rmagnet/stage1_train.py` 中的基础损失。以下定义以当前分支实际代码为准。

## 1. 记号与数值范围

- `I`：待处理的 blended RGB 图。
- `T`：transmission GT。
- `T_hat`：Transmission LoRA 输出。
- `W_p`：像素权重图，形状 `384×512`。
- `W_q`：Q20 token 权重，形状 `24×32`，展平后共 768 个 token。
- `Q20(T_hat)`：预测结果经过冻结 Qwen 主干后，第 20 个 block 的特征。
- `Q20(T)`：训练准备阶段离线缓存的 GT 特征，形状 `768×3072`，BF16 保存。

模型图像张量原本位于 `[-1,1]`。所有像素损失先执行：

```text
P = clamp((T_hat + 1) / 2, 0, 1)
G = clamp((T     + 1) / 2, 0, 1)
X = clamp((I     + 1) / 2, 0, 1)
```

因此 `L_base`、局部 Charbonnier 和保持损失都在 `[0,1]` RGB 空间计算。

正式总损失为：

```text
L_total = L_base
        + 0.25 * L_weighted_charbonnier
        + 0.10 * L_low_response_keep
        + lambda_q * L_Q20
```

对应固定配置：

| 项目 | 数值 |
|---|---:|
| SSIM 系数 | 0.20 |
| Edge 系数 | 0.10 |
| 局部 Charbonnier 系数 | 0.25 |
| 低响应保持系数 | 0.10 |
| Charbonnier ε² | `1e-6` |
| `lambda_q` 范围 | `[0.02, 0.5]` |
| `lambda_q` EMA | 0.9 |
| 梯度比例测量间隔 | 20 次优化更新 |

## 2. 基础重建损失 L_base

```text
L_base = L1(P,G) + 0.20 * (1 - SSIM(P,G)) + 0.10 * L_edge(P,G)
```

### 2.1 全图 L1

```text
L1(P,G) = mean(|P-G|)
```

对 batch、RGB 通道和全部像素求平均。它提供稳定的全图颜色与亮度约束。

### 2.2 SSIM

实现使用：

- `11×11` 平均池化窗口；
- stride 1、padding 5；
- `C1=0.01²`；
- `C2=0.03²`；
- 最后对 batch、通道和空间位置求平均。

```text
L_ssim = 1 - SSIM(P,G)
```

这里使用的是代码内实现的 box-window SSIM，不是第三方库的 Gaussian-window 或多尺度 SSIM。

### 2.3 边缘 L1

分别计算水平、垂直一阶有限差分：

```text
dx(P) = P[..., x+1] - P[..., x]
dy(P) = P[..., y+1, :] - P[..., y, :]

L_edge = 0.5 * (
    L1(dx(P), dx(G))
  + L1(dy(P), dy(G))
)
```

该项约束局部梯度，减少边缘和细线结构被过度抹平。

## 3. 离线权重图

Qwen 第20层对输入和 GT 的 token 特征先做余弦差异：

```text
D_Q = cosine_distance(Q20(I), Q20(T))
```

每张图的原始 `D_Q` 使用 2%/98% 分位做稳健 min-max，并截断到 `[0,1]`。DoLP 直接使用 8 位灰度值除以255：

```text
D_DoLP ∈ [0,1]
S = D_Q * (0.7 + 0.3 * D_DoLP)
W_raw = clip(1 + 2*S, 1, 3)
W_q = W_raw / mean(W_raw)
```

随后把 `W_q` 从 `24×32` 双线性上采样为 `384×512`，并再次归一化：

```text
W_p = upsample(W_q)
W_p = W_p / mean(W_p)
```

注意：

- `[1,3]` 是 `W_raw` 的范围。
- 最终 `W_q`、`W_p` 的均值为1，因此低响应位置可以小于1。
- DoLP 只存在于离线权重缓存中；训练循环不会再把 DoLP 直接输入模型或单独计算损失。
- DoLP 只能在 Qwen 差异已经存在的位置将 `D_Q` 从70%增强到100%，不能在 `D_Q=0` 的位置独立制造高权重。

## 4. 加权局部 Charbonnier

先对每个像素的三个RGB通道求平均：

```text
C(h,w) = mean_c sqrt((P(c,h,w)-G(c,h,w))² + 1e-6)
```

再使用像素权重图：

```text
L_weighted_charbonnier
    = sum(W_p * C) / max(sum(W_p), 1e-6)
```

因为 `W_p` 逐图归一化到均值1，该项的总体量级不会仅因一张图的反射区域更大而任意增大；它主要改变不同位置对梯度的相对贡献。

Charbonnier 在误差接近0时比绝对值更平滑，`sqrt(1e-6)=0.001` 是它的数值平滑尺度。

## 5. 低响应区域保持损失

该项使用最终像素权重图在每张图内部重新做 min-max：

```text
A = (W_p - min(W_p)) / max(max(W_p)-min(W_p), 1e-6)
R_low = 1 - A
```

其中 `A` 是相对重要性，`R_low` 是低响应保持权重。随后约束输出接近原输入：

```text
E_keep(h,w) = mean_c |P(c,h,w)-X(c,h,w)|

L_low_response_keep
    = sum(R_low * E_keep) / max(sum(R_low), 1e-6)
```

作用是减少低 Qwen 差异、低加权区域的无依据改写。它约束的是“保持输入”，不是“接近GT”。

如果一张图的 `W_p` 完全为常数，代码中的分母保护会令 `A=0`、`R_low=1`，此时保持损失退化为整图输出与输入的RGB平均 L1。

## 6. Q20 特征损失

预测图 `T_hat` 使用确定性 VAE 编码，并在固定文本 embeddings、flow timestep 499 下进入冻结 Qwen。全部 LoRA 在该教师前向中关闭，并在第20个 block 后停止。

对每个 token 的3072维特征做 L2 归一化：

```text
q_hat_i = normalize(Q20(T_hat)_i)
q_gt_i  = normalize(Q20(T)_i)

d_i = 1 - dot(q_hat_i, q_gt_i)
```

使用离线 token 权重：

```text
L_Q20 = sum_i(W_q_i * d_i) / max(sum_i(W_q_i), 1e-6)
```

Qwen 主干、VAE和 GT 特征都不训练，但梯度可以从 `Q20(T_hat)` 经过预测图传回 Transmission LoRA。因此该项监督的是预测结果在 Qwen 第20层的表示，而不是训练一个新的 Qwen 模型。

## 7. lambda_q 的梯度比例控制

代码分别计算两个相对于预测图 `T_hat` 的输出梯度：

```text
g_base = ∂(L_base + 0.25*L_weighted_charbonnier
                   + 0.10*L_low_response_keep) / ∂T_hat

g_q = ∂L_Q20 / ∂T_hat
```

实际回传为：

```text
g_total = g_base + lambda_q * g_q
```

这在一阶梯度上等价于对 `L_total` 直接反向传播。代码分开求梯度，是为了及时释放冻结 Qwen 教师的计算图，控制24GB GPU显存。

目标梯度比例定义为：

```text
r_target = 0.15 * step_in_epoch / steps_per_epoch   # 第一个epoch线性升至15%
r_target = 0.20                                     # 后续epoch
```

在第1次优化更新以及之后每20次优化更新时测量：

```text
lambda_desired = r_target * ||g_base|| / ||g_q||
lambda_desired = clip(lambda_desired, 0.02, 0.5)

lambda_q = 0.9 * lambda_q + 0.1 * lambda_desired
lambda_q = clip(lambda_q, 0.02, 0.5)

r_actual = lambda_q * ||g_q|| / ||g_base||
```

这里的梯度范数是所有GPU汇总后的预测图输出梯度 L2 范数，不是 LoRA 参数梯度范数。未到测量步时继续使用上次的 `lambda_q`。

## 8. 本次10轮训练中的实际情况

正式运行 `c1_l20_e10` 中，`lambda_q` 始终停在下限 `0.02`。新测量点的实际 Q20/基础输出梯度比例为：

| Global step | 实际比例 |
|---:|---:|
| 1 | 1.945 |
| 20 | 2.597 |
| 40 | 0.929 |
| 60 | 1.607 |

因此当前实现没有达到设计的15%–20%，而是让加权后的 Q20 输出梯度约为基础梯度的0.93至2.60倍。原因是原始 `||g_q||/||g_base||` 很大，而 `lambda_q=0.02` 的下限仍然过高。

这意味着：

- 当前 Q20 监督确实进入训练并且很强。
- 10轮结果的提升不能只归因于空间权重，也可能来自强Q20特征约束。
- 若下一轮目标是严格控制在20%左右，需要允许 `lambda_q` 低于0.02，或重新定义梯度控制器。
- 日志中的 loss 标量比例不能代替梯度比例；两者量纲和局部导数不同。

## 9. 梯度累积与日志

- 四张GPU，每卡batch 1。
- 梯度累积为2，有效batch 8。
- 每个 microbatch 的 `g_total` 除以当前累积组实际大小后回传。
- LoRA参数梯度跨两个 microbatch 累积，再执行梯度裁剪和优化器更新。
- 训练日志分别记录 `l1`、`ssim_loss`、`edge`、`base_loss`、`local_loss`、`keep_loss`、`q20_loss`、`lambda_q` 和测得的梯度范数。
- early stopping 与 best checkpoint 使用验证集平均 PSNR，不直接使用训练总损失。

## 10. 源码对应位置

- 总损失与局部项：`src/rmagnet/c1_l20_train.py::c1_losses`
- Q20预测特征：`src/rmagnet/c1_l20_train.py::q20_prediction_features`
- 梯度比例控制：`src/rmagnet/c1_l20_train.py` 的正式训练循环
- 基础Transmission损失：`src/rmagnet/stage2_train.py::transmission_loss`
- SSIM与边缘损失：`src/rmagnet/stage1_train.py::ssim / edge_l1`
- 离线权重：`src/rmagnet/c1_l20_prepare.py`
