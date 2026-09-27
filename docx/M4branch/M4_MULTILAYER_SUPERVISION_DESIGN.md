# M4：Qwen 多层职责分离监督训练方案

## 1. 目标

M4 用多个 Qwen-Image-Edit 层承担不同职责，替换 M2/M3 中由单一 block 20 同时承担定位和语义监督的设计。

| 职责 | Qwen block | 使用方式 |
|---|---|---|
| 局部纹理保持 | 16、20 | 在线比较预测特征与缓存的输入/GT 特征 |
| 语义内容与关系 | 37、39、41 | 在线比较中心化特征与四邻域关系 |
| 反射位置门控 | 52、54、56 | 仅离线生成连续空间门控 |

模型只训练 Transmission LoRA。Qwen 主干、VAE 和文本条件全部冻结。

## 2. 设计依据

受控扰动实验显示：

- block 1–20 对局部模糊和文字笔画删除具有最高空间局部性；
- block 37–41 对对象替换响应最强，但响应会传播到更大范围；
- block 48–58 的热点重新集中；
- block 60 对多种扰动的绝对响应接近零；
- 中层对曝光和白平衡变化同样敏感，不能直接把高响应解释成纯语义。

因此 M4 把定位、局部保持和语义关系分开，不再寻找一个层完成所有任务。

## 3. 离线缓存

### 3.1 数据范围

只处理纠正标签后的 144 张 M2 训练样本：

- 输入：`blended/<id>.png`
- GT：`transmission_layer/<id>.png`

验证集和封存测试集不会进入缓存。

### 3.2 冻结教师口径

- Qwen-Image-Edit-2509 基础主干；
- 全部 LoRA 关闭；
- VAE posterior mode；
- timestep 499；
- 保留动态长宽比；
- token 网格为 `H/16 × W/16`。

### 3.3 每个样本的缓存

```text
q16_input, q16_gt
q20_input, q20_gt
q37_gt, q39_gt, q41_gt
late_gate
late_agreement
token_grid_hw
```

特征使用 BF16，门控使用 FP16。以平均 768 个 token 估算，完整缓存约 4.5 GiB。

训练阶段关闭随机水平翻转。缓存特征包含位置编码，直接翻转 token 排列并不等价于对翻转图像重新提取 Qwen 特征；第一版优先保证教师目标严格对齐。若后续需要翻转增强，应在缓存阶段同时生成真实翻转图像的特征。

### 3.4 末段门控

对 \(l\in\{52,54,56\}\)：

\[
D_l(x)=1-\cos(Q_l(I)_x,Q_l(GT)_x)
\]

每层在单张图内使用 p02/p98 robust min-max：

\[
\bar D_l(x)=\operatorname{clip}
\left(
\frac{D_l(x)-q_{.02}(D_l)}
{q_{.98}(D_l)-q_{.02}(D_l)},
0,1
\right)
\]

跨层平均：

\[
\mu_D(x)=
\frac{\bar D_{52}(x)+\bar D_{54}(x)+\bar D_{56}(x)}{3}
\]

层间一致性：

\[
A(x)=\operatorname{clip}
\left(
1-2\operatorname{std}
(\bar D_{52},\bar D_{54},\bar D_{56}),
0,1
\right)
\]

最终门控：

\[
G(x)=\mu_D(x)\left(0.75+0.25A(x)\right)
\]

单层异常不会完全决定位置；三层意见不一致时只做轻度降权。

M4 第一版不把 DoLP 混入门控，以便单独验证多层 Qwen 监督。DoLP 可在后续消融中作为独立增益项加入。

## 4. 在线训练

### 4.1 初始化

从干净的 Stage 2 最佳 Transmission LoRA 初始化：

```text
runs/stage2_transmission_r128/best_transmission_lora.safetensors
SHA-256:
f5737d4ffb89e86874a96a02bd58a074299ca12e00ec15cac438c403a342085a
```

不从 M3 权重开始，避免继承单层 Q20 监督的偏差。

### 4.2 基础重建

对普通输入 \(I\) 和反射增强输入 \(P90\)：

\[
L_{\mathrm{rec}}(X)=
L_1(F(X),GT)
+0.2(1-\operatorname{SSIM}(F(X),GT))
+0.1L_{\mathrm{edge}}(F(X),GT)
\]

基础目标：

\[
L_{\mathrm{base}}=
\frac{L_{\mathrm{rec}}(I)+L_{\mathrm{rec}}(P90)}{2}
+0.10L_{\mathrm{polar}}
\]

\(L_{\mathrm{polar}}\) 约束同一场景的两个输出一致，并使用晚层 gate 强调反射变化区域。P90 只用于训练。

### 4.3 末段空间损失

将 token gate 插值到像素尺寸：

\[
W(x)=
\frac{1+2G(x)}
{\operatorname{mean}(1+2G)}
\]

加权 GT 恢复：

\[
L_{\mathrm{weighted}}=
\operatorname{mean}
\left[
W(x)\rho(\hat T(x)-GT(x))
\right]
\]

低响应保持：

\[
L_{\mathrm{keep,pixel}}=
\frac{
\sum_x(1-G(x))\rho(\hat T(x)-I(x))
}{
\sum_x(1-G(x))
}
\]

空间组损失：

\[
L_{\mathrm{spatial}}
=
0.75L_{\mathrm{weighted}}
+0.25L_{\mathrm{keep,pixel}}
\]

### 4.4 前段纹理损失

定义门控可信度：

\[
C(x)=0.5+0.5A(x)
\]

定义保留与恢复权重：

\[
M_{\mathrm{keep}}=(1-G)C,\qquad
M_{\mathrm{restore}}=GC
\]

对 \(l\in\{16,20\}\)：

\[
L_{\mathrm{texture}}^l=
\frac{
\sum_x M_{\mathrm{keep}}d(Q_l(\hat T),Q_l(I))
+
M_{\mathrm{restore}}d(Q_l(\hat T),Q_l(GT))
}{
\sum_x(M_{\mathrm{keep}}+M_{\mathrm{restore}})
}
\]

其中 \(d=1-\mathrm{cosine}\)，最终对两个层取平均。

这不会要求全图复制输入：低 gate 区域保持输入纹理，高 gate 区域匹配 GT。

### 4.5 中层语义内容

中层对曝光敏感，因此先在 token 维度去掉每张图的全局均值，再做 L2 归一化：

\[
\tilde Q_l=
\operatorname{normalize}
\left(
Q_l-\operatorname{mean}_{token}(Q_l)
\right)
\]

对 \(l\in\{37,39,41\}\)：

\[
L_{\mathrm{content}}^l=
\frac{
\sum_x C(x)
\left[
1-\cos(\tilde Q_l(\hat T)_x,\tilde Q_l(GT)_x)
\right]
}{
\sum_x C(x)
}
\]

去中心化只能缓解全局光度偏置，不能证明这些层具有纯语义不变性。

### 4.6 中层邻域关系

在 token 网格上计算水平和垂直四邻域关系：

\[
R_l(x,y)=\cos(\tilde Q_l(x),\tilde Q_l(y))
\]

预测关系匹配 GT：

\[
L_{\mathrm{relation}}^l=
\operatorname{SmoothL1}
\left(
R_l(\hat T),R_l(GT)
\right)
\]

关系边使用相邻 token 的平均可信度加权。语义组损失：

\[
L_{\mathrm{semantic}}
=
0.7L_{\mathrm{content}}
+0.3L_{\mathrm{relation}}
\]

## 5. 梯度控制

M4 不直接依赖固定 loss 数值系数，因为不同层的特征尺度差异很大。分别计算对输出图的梯度：

\[
g_{\mathrm{base}},
g_{\mathrm{spatial}},
g_{\mathrm{texture}},
g_{\mathrm{semantic}}
\]

三个辅助目标的目标梯度比例均为 8%：

\[
r_s=r_t=r_m=0.08
\]

缩放系数：

\[
\alpha_k=
\operatorname{EMA}
\left[
\operatorname{clip}
\left(
r_k\frac{\|g_{\mathrm{base}}\|}
{\|g_k\|+\epsilon},
10^{-4},10
\right)
\right]
\]

前 36 次更新从 0 线性增加到目标比例。合并后施加硬上限：

\[
\left\|
\sum_k\alpha_kg_k
\right\|
\le 0.25\|g_{\mathrm{base}}\|
\]

最终输出梯度：

\[
g_{\mathrm{out}}
=
g_{\mathrm{base}}
+
\operatorname{cap}_{0.25}
\left(
\alpha_sg_s+\alpha_tg_t+\alpha_mg_m
\right)
\]

日志记录每组原始梯度、缩放系数、限幅系数和实际辅助/基础梯度比例。

## 6. 24GB 显存策略

在线只提取预测图的 block 16、20、37、39、41，并在 block 41 提前终止。block 52、54、56 只参与离线缓存。

单步顺序：

1. 无梯度生成 \(T_I\) 与 \(T_{90}\) 参考结果；
2. 将 \(T_I\) 作为独立叶子张量，运行冻结教师至 block 41；
3. 求各损失对 \(T_I\) 的 VJP；
4. 释放教师计算图；
5. 重新执行带梯度的 Transmission LoRA 前向；
6. 用保存的输出梯度回传 LoRA；
7. 处理 P90 分支；
8. 四卡同步 LoRA 梯度并更新。

该方案以额外计算换显存，避免生成计算图与 41 层教师图同时驻留。

## 7. 默认训练配置

| 参数 | 默认值 |
|---|---:|
| GPU | 4 张 |
| 每卡 batch | 1 |
| 有效 batch | 4 |
| 初始化 | Stage 2 best |
| 学习率 | 5e-6 |
| 优化器 | PagedAdamW8bit |
| Warmup | 20 updates |
| 梯度裁剪 | 1.0 |
| 辅助渐入 | 36 updates |
| 单组目标梯度比例 | 8% |
| 辅助梯度总上限 | 25% |
| 验证 | 每个 epoch |
| 保存 | 仅 best LoRA |
| 默认 Epoch | 10 |

推荐顺序：

| 训练量 | 目的 |
|---|---|
| 2 steps | 显存和反向链路 smoke |
| 2 Epoch | 检查训练方向 |
| 5 Epoch | 与既有方案比较 |
| 10 Epoch | 首轮完整实验 |
| 20 Epoch | 仅当 10 Epoch 仍持续改善 |

当前 M3 长训练在约第 18 Epoch 达到最佳后退化，所以 M4 不默认进行 100 Epoch。

## 8. 实现文件

| 文件 | 作用 |
|---|---|
| `src/rmagnet/m4_cache.py` | 四卡可恢复缓存与最终封存 |
| `src/rmagnet/m4_train.py` | 多层损失、VJP、四卡训练和验证 |
| `scripts/prepare_m4_cache.sh` | 四卡缓存生成 |
| `scripts/smoke_m4.sh` | 两步 smoke |
| `scripts/train_m4.sh` | 可选 Epoch 正式训练 |
| `docx/M4branch/RUNBOOK.md` | 命令和运行检查 |

## 9. 验收标准

缓存阶段：

- 恰好覆盖 144 个训练样本；
- 不包含验证集和封存测试集；
- 哈希、尺寸、token 网格全部通过；
- 特征、gate 与 agreement 均为有限值；
- gate 范围为 \([0,1]\)；
- 单卡峰值低于 24 GiB。

Smoke 阶段：

- 四卡均参与；
- 冻结 Qwen 和 VAE 没有梯度；
- 只有 Transmission LoRA 更新；
- 在线教师在 block 41 停止；
- 三个辅助梯度均非零；
- 合计辅助梯度不超过基础梯度 25%；
- 验证和 best LoRA 保存成功；
- 显存不随 step 持续增长。

正式比较统一使用纠正标签后的封存测试集，报告：

- L1、PSNR、SSIM、LPIPS；
- 低变化区域 L1；
- 高变化区域 L1；
- 文字与细纹理局部对比；
- Stage 2、M2-A、M2-B、M2-B1、M3、M4 同口径表格。

## 10. 风险

1. 中层仍可能响应光度变化；去中心化和关系损失只能缓解。
2. 在线教师运行到 block 41，单步耗时预计高于 M3，需要 smoke 实测。
3. 晚层 gate 使用训练期 GT，只能作为监督，推理时仍只输入待处理图。
4. M4 第一版不使用 DoLP，后续需单独检验 DoLP 的增益。
5. 当前受控扰动以一个主场景为主，正式训练前仍应检查五个场景的候选层稳定性。
