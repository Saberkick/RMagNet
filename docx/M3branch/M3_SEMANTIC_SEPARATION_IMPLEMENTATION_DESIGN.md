# M3：从位置权重扩展到语义分离表示——实现设计

> 状态：实现设计完成，尚未编写训练代码或启动实验
> 分支：`experiment/m3-semantic-separation`
> 日期：2026-09-27
> 上游：`experiment/m2-corrected-labels@bebf0dc`
> 目标：在不增加推理结构的前提下，把 Qwen block 20 从单一位置权重扩展为“干净内容关系 + 污染变化状态”，再利用同场景 `I / P90 / GT / DoLP` 训练 Transmission LoRA。

---

## 1. M3 要解决什么

M2 的 Q20 监督把

\[
D_Q(x)=1-\cos(Q20(I)_x,Q20(GT)_x)
\]

压缩成单个标量。它能表示“这个位置变化多大”，却丢失了：

- token 属于什么物体或表面；
- 哪些 token 在语义上应当一起保留；
- 输入相对 GT 的变化方向；
- 普通输入与反射增强输入是否表现出同一种污染；
- 变化位于区域内部、混合边界还是细纹理；
- 变化究竟更像反射、配准、曝光还是压缩误差。

纠正标签后的 M2 结果也说明，只提高 Q20 损失强度不足以带来稳定增益：

| 实验 | PSNR | SSIM | L1 | LPIPS |
|---|---:|---:|---:|---:|
| M2-A：无 Q20 | 24.10836 | 0.829485 | 0.049178 | 0.115681 |
| M2-B：旧 Q20 完整方案 | **24.16183** | 0.829621 | **0.048769** | **0.114754** |
| M2-B1：Q20 梯度严格 30% | 24.15496 | **0.829759** | 0.048828 | 0.115080 |

三组差异很小，而且把 Q20 梯度严格控制为 30% 没有产生相应收益。M3 因此不再继续放大同一个标量损失，而是改变监督表示。

M3 的核心定义是：

\[
C(x)=Q20(GT)_x
\]

\[
E_I(x)=Q20(I)_x-Q20(GT)_x
\]

\[
E_{90}(x)=Q20(P90)_x-Q20(GT)_x
\]

其中：

- `C` 表示干净场景中的内容、材质与局部关系；
- `E_I` 表示普通输入相对干净图的变化方向；
- `E_90` 表示反射增强观测相对干净图的变化方向；
- DoLP 只调节偏振证据的可信度，不单独决定“这里是反射”。

最终推理仍为：

\[
\hat T=F_{\theta}(I)
\]

只保留训练后的 Transmission LoRA。Qwen 教师缓存、P90、GT、DoLP、软聚类和关系图都只在训练期使用。

---

## 2. 设计原则

### 2.1 教师固定

教师始终是：

- 关闭全部 LoRA 的基础 Qwen-Image-Edit；
- block 20，代码索引 19；
- timestep 499；
- 固定提示词和确定性 VAE 编码；
- eval 模式，参数永久冻结。

不得使用当前学生 LoRA 重算 `C / E_I / E_90`。否则学生错误会进入教师目标，形成自我确认。

### 2.2 分开“这里是什么”与“这里发生了什么”

M3 生成两类互补表示：

1. **内容关系表示**：由 `C` 构造，描述 token 之间是否可能属于同一物体、表面或连续纹理。
2. **污染状态表示**：由 `E_I / E_90 / DoLP / RGB 差异 / 局部梯度` 构造，描述变化类型、强度和置信度。

两者不能再次压成一张硬 T/R 蒙版。

### 2.3 不向表示写入绝对位置

聚类特征中不加入 `x/y` 坐标，不以文件编号、画面位置或蒙版形状作为输入。空间连续性只通过邻域平滑和内容关系图表达，以降低模型只记住蒙版形状的风险。

### 2.4 P90 是第二个观测，不是 Reflection GT

`P90` 只能表示同一场景在另一偏振状态下的反射增强观测。它与反射层不等价，不允许直接作为 `R` 监督。

---

## 3. M3 总体结构

训练期数据流：

```text
                         Frozen base Qwen, LoRA off
GT  ------------------> C = Q20(GT) ------------------> semantic relation graph
I   ------------------> E_I = Q20(I) - C ------┐
P90 ------------------> E_90 = Q20(P90) - C ---┼--> soft pollution states
DoLP / RGB / gradients -------------------------┘

I   ----> shared Qwen DiT + LoRA_T ----> T_I  ----┐
P90 ----> shared Qwen DiT + LoRA_T ----> T_90 ----┼--> GT reconstruction
                                                   ├--> polarization consistency
Q20(T_I) + cached relation targets ----------------┘
```

推理期数据流：

```text
I ----> shared Qwen DiT + trained LoRA_T ----> T
```

M3 不添加 Interface Head、不保留聚类头、不增加第二个推理分支。

---

## 4. 阶段一：离线语义缓存

### 4.1 输入和身份校验

只使用 M2 的 144 张 train split 拟合 PCA、原型和归一化统计。validation/test 不参与缓存拟合。

必须检查：

- 数据版本为 `m2-variable-aspect-v2-corrected-labels`；
- `blended/` 是带反射输入 `I`；
- `transmission_layer/` 是干净 `GT`；
- `reflection_90/` 是 `P90`；
- `dolp/` 是单通道 DoLP；
- 四种图像尺寸、方向和编号完全一致；
- 数据 manifest SHA-256 与当前纠正后版本一致；
- 现有 `data_cache/m2a_q20` 的教师身份、block、timestep、GT 哈希和数据 manifest 哈希全部匹配。

现有 `Q20(GT)` 缓存约 680 MB，可以直接复用为 `C`，无需重新计算 GT。

### 4.2 特征提取方式

每次只把一张图送入单张 24 GB GPU：

1. 从现有缓存加载 `C=Q20(GT)`。
2. 提取 `Q20(I)`，立即计算 `E_I` 和统计量。
3. 释放 `Q20(I)`。
4. 提取 `Q20(P90)`，立即计算 `E_90` 和统计量。
5. 释放 `Q20(P90)`。
6. 把临时残差写入 scratch，继续下一张。

预计峰值显存与 M2a Q20 缓存相近，目标小于 18 GiB；不得同时保留三个 Qwen 前向图。

### 4.3 降维与特征分组

从训练 token 中按固定种子分层抽取最多 32768 个 token，使用 `torch.pca_lowrank` 拟合：

- `P_C(C)`：32 维；
- `P_E(E_I)`：32 维；
- `P_E(E_90)`：共享同一个残差投影，32 维。

使用同一个残差投影可以直接比较 `E_I` 与 `E_90` 的方向。PCA 均值、分量、解释方差和抽样 ID 写入 manifest。

每个特征组先独立做 robust normalization 和 L2 normalization，再按组缩放。污染状态特征为：

\[
z(x)=
[
0.25P_C(C),
P_E(E_I),
P_E(E_{90}),
d_I,
d_{90},
a_E,
d_{I,90},
d_{rgb}^{I},
d_{rgb}^{90},
d_{grad}^{I},
d_{grad}^{90},
D_{DoLP}
]
\]

其中：

\[
d_I=1-\cos(Q20(I),C),\quad
d_{90}=1-\cos(Q20(P90),C)
\]

\[
a_E=\max(\cos(E_I,E_{90}),0)
\]

\[
d_{I,90}=1-\cos(Q20(I),Q20(P90))
\]

`P_C(C)` 权重较低，防止聚类只按物体类别分组；`E_I/E_90` 和变化证据占主导。特征中不加入绝对坐标。

### 4.4 内容关系图

从 `C` 构建稀疏关系图，不构造 `N×N` 全矩阵。

每个 token 保存：

- 局部偏移：上下左右、四个对角、距离为 2 的水平和垂直邻居；
- 两个高相似非局部 token；
- 两个低相似非局部 token。

目标为：

\[
A_C(i,j)=
\sigma\left(
\frac{\cos(C_i,C_j)-\tau_{img}}{T_{rel}}
\right)
\]

其中 `τ_img` 是该图候选关系相似度的中位数，`T_rel` 初始设为 0.07。缓存保存：

- `edge_index[2,E]`：INT32；
- `edge_target[E]`：FP16；
- `edge_confidence[E]`：FP16；
- token grid 高宽。

这种表示保留“哪些位置属于一起”，并把存储量限制在线性规模。

### 4.5 软污染状态

使用 `K=4` 的全局原型，不给簇强行命名。拟合过程：

1. 对抽样 token 的 `z` 做 balanced Sinkhorn assignment；
2. 更新四个原型；
3. 迭代至原型变化收敛或最多 50 次；
4. 为全部 token 计算 soft posterior `P(x,k)`；
5. 使用由 `C` 相似度控制的 3 次邻域平滑；
6. 保存原始 posterior 与平滑 posterior 的差异统计。

输出：

\[
P(x,k),\quad k=1,2,3,4
\]

\[
U(x)=1-\frac{H(P(x))}{\log K}
\]

其中 `U` 是聚类置信度。训练使用连续 posterior，不做 argmax 硬分类。

容量门槛：

- 每个簇全局占比必须在 5% 到 45%；
- posterior 每个 token 求和误差小于 `1e-4`；
- 不能出现单簇吸收超过 60% token；
- 三个固定种子拟合后，Hungarian 对齐的平均 posterior 一致性应达到 0.80；
- 不满足门槛时停止，不生成“完成”manifest。

### 4.6 反射证据与边界图

额外构造连续反射证据：

\[
R(x)=\operatorname{robust\_unit}
\left(
\sqrt{d_I(x)d_{90}(x)}
\cdot a_E(x)
\cdot (0.7+0.3D_{DoLP}(x))
\right)
\]

这个公式要求普通输入和 P90 相对 GT 都发生 Q20 变化，并且变化方向一致。DoLP 最多增强 30%，不能单独产生高反射权重。

边界图：

\[
B(x)=\operatorname{robust\_unit}
\left(
0.5\frac{H(P(x))}{\log K}
+0.5\lVert\nabla R(x)\rVert
\right)
\]

高熵表示混合状态不确定，高 `∇R` 表示污染强度快速过渡。二者共同定义需要谨慎处理的软边界。

### 4.7 最终缓存结构

```text
data_cache/m3_semantic_v1/
├── manifest.json
├── pca/
│   ├── content_pca.safetensors
│   └── residual_pca.safetensors
├── prototypes/
│   ├── prototypes.safetensors
│   └── fit_report.json
├── samples/
│   └── <id>.safetensors
├── previews/
│   └── <id>_panel.png
└── audit/
    ├── cluster_occupancy.csv
    ├── seed_stability.json
    └── cache_check.json
```

每个 `<id>.safetensors` 保存：

- `posterior[K,N]`：FP16；
- `confidence[N]`：FP16；
- `reflection_evidence[N]`：FP16；
- `boundary[N]`：FP16；
- `edge_index[2,E]`：INT32；
- `edge_target[E]`：FP16；
- `edge_confidence[E]`：FP16。

完整 `E_I/E_90` 只作为缓存生成 scratch。最终缓存验证通过后删除 scratch，避免长期占用数 GB。

### 4.8 数据增强

继续使用与 M2-A 相同的固定种子水平翻转。翻转时必须同步：

- `I / P90 / GT / DoLP`；
- posterior、confidence、reflection evidence、boundary；
- 关系图的两个端点索引。

token 索引变换为：

\[
(r,c)\rightarrow(r,W_t-1-c)
\]

关系边是无向边，翻转后目标值不变。实现中必须用单元测试验证“翻转两次恢复原索引”。

---

## 5. 阶段二：训练目标

定义：

\[
T_I=F_\theta(I),\qquad T_{90}=F_\theta(P90)
\]

两个输入共用同一个 Transmission LoRA；没有 P90 专属参数。

### 5.1 基础重建

\[
L_{rec}
=\frac{1}{2}
\left[
L_{base}(T_I,GT)
+
L_{base}(T_{90},GT)
\right]
\]

`L_base` 继续使用 Stage 2 的 L1、SSIM 和 edge 组合。两个观测都必须被 GT 锚定，防止一致性损失让两路输出收敛到同一个错误结果。

### 5.2 软状态均衡重建

对每个软簇分别计算 Charbonnier，再对四个簇平均：

\[
L_{cluster}^{v}
=
\frac{1}{K}
\sum_{k=1}^{K}
\frac{
\sum_x P_k(x)\rho(T_v(x)-GT(x))
}{
\sum_x P_k(x)+\epsilon
}
\]

\[
L_{cluster}
=\frac{1}{2}
(L_{cluster}^{I}+L_{cluster}^{90})
\]

这样稳定区域、强变化区域、混合边界和高风险纹理不会因像素数量不同而被整图平均吞没。这里使用 soft posterior，不要求人为解释每个簇。

### 5.3 内容关系损失

只对部署时会使用的普通输入分支计算在线教师特征：

\[
\hat C=Q20(T_I)
\]

\[
L_{relation}
=
\frac{
\sum_{(i,j)}
w_{ij}
\operatorname{SmoothL1}
\left(
A_{\hat C}(i,j)-A_C(i,j)
\right)
}{
\sum_{(i,j)}w_{ij}+\epsilon
}
\]

Qwen 教师仍冻结并关闭 LoRA。关系损失匹配 token 间结构，不要求每个 token 的绝对向量与 GT 完全相同。

第一版不对 `T_90` 再跑一次 Q20，以控制单卡显存和训练耗时。P90 分支通过 GT 重建、软状态损失和偏振一致性接受监督。

### 5.4 偏振不变性

\[
L_{cons}
=
\frac{
\sum_x (1+\gamma R(x))
\rho(T_I(x)-T_{90}(x))
}{
\sum_x (1+\gamma R(x))+\epsilon
}
\]

初始 `γ=1`。完整偏振项已经包含在 `L_rec + L_cons` 中，不把 P90 当作反射层。

### 5.5 软边界损失

\[
L_{boundary}
=
\frac{1}{2}
\sum_{v\in\{I,90\}}
\frac{
\sum_x B(x)
\rho(\nabla T_v(x)-\nabla GT(x))
}{
\sum_x B(x)+\epsilon
}
\]

该项专门约束反射边缘的光晕、双边和过度抹除，不在区域内部施加强硬边界。

### 5.6 总损失

\[
L
=
L_{rec}
+\lambda_c L_{cluster}
+\lambda_r L_{relation}
+\lambda_p L_{cons}
+\lambda_b L_{boundary}
\]

M3 第一版不叠加 M2-B 的直接逐 token `L_Q20`、旧 weighted Charbonnier 和 low-response keep。对应能力分别由 relation、cluster-balanced reconstruction 和双路 GT 重建承担，避免同一信号重复计权。

---

## 6. 辅助梯度控制与教师撤除

M2-B1 已证明“把旧 Q20 标量梯度提至 30%”本身没有明显作用。M3 使用结构化辅助梯度，并限制总预算。

### 6.1 梯度预算

建议初始相对系数：

| 项 | 初始系数 |
|---|---:|
| cluster | 0.25 |
| relation | 0.10 |
| polar consistency | 0.10 |
| boundary | 0.05 |

每 10 次 optimizer update 测一次输出空间梯度：

\[
r_{aux}
=
\frac{\lVert g_{cluster}+g_{relation}+g_{cons}+g_{boundary}\rVert}
{\lVert g_{rec}\rVert+\epsilon}
\]

前 10 步目标为 0.20，之后目标为 0.25；允许所有辅助系数低于 0.02。使用 EMA 缩放辅助项，使组合辅助梯度不超过基础梯度的 30%。日志同时记录每一项的单独梯度范数和组合后的真实比例。

### 6.2 教师撤除

正式训练最后 20% 更新中，线性衰减：

- `λ_cluster → 0`；
- `λ_relation → 0`；
- `λ_boundary → 0`。

保留：

- `L_rec`；
- `L_cons`。

如果撤除后验证性能保持，说明语义分离能力已固化进 LoRA_T。若明显下降，说明模型仍依赖训练期教师目标。

---

## 7. 24 GB 单卡下的训练实现

同时保留 `T_I` 和 `T_90` 的完整 DiT 计算图容易超过 24 GB。M3 使用“冻结 latent + 输出梯度回放”。

### 7.1 单个 optimizer step

1. 冻结 VAE，分别编码 `I` 和 `P90`，得到 `z_I/z_90`。
2. 使用相同 latent 做无图前向，得到 `T_I/T_90` 的 detached 副本。
3. 把两个输出作为叶张量计算像素损失、软状态损失、偏振一致性和边界损失。
4. 只对 `T_I` 运行一次冻结 Qwen 到 block 20，计算 relation loss。
5. 得到损失对 `T_I/T_90` 的输出梯度 `g_I/g_90`。
6. 从保存的 `z_I` 重放学生前向，检查输出与 detached 副本最大误差小于 `2e-5`，执行 `backward(g_I)`。
7. 释放第一路计算图。
8. 从 `z_90` 重放学生前向，执行 `backward(g_90)`。
9. 四卡手动 all-reduce LoRA 梯度，裁剪后执行一次 optimizer step。
10. 清理显存并记录峰值。

这会增加计算时间，但任何时刻只保留一路学生计算图和一路教师图，适合 24 GB 3090。

### 7.2 必须增加的后端接口

在 `QwenSharedBackend` 增加：

```python
encode_latent_frozen(image) -> latent
forward_transmission_from_latent(latent) -> prediction
```

同一个 latent 必须可以重复得到相同预测，避免重新采样 VAE latent 后把旧输出梯度施加到另一个预测。

### 7.3 多 GPU

- 4 张 GPU；
- 每卡 batch 1；
- 有效 batch 4；
- 同一全局 step 内安排相同或相近 token grid 的样本；
- 每张卡独立处理一组 `I/P90/GT`；
- 两路 backward 后只同步一次 LoRA 梯度；
- 主干、VAE、文本条件和教师 Qwen 全部冻结；
- 只允许 LoRA_T 参数有梯度。

---

## 8. 计划新增的代码和脚本

```text
src/rmagnet/
├── m3_cache.py              # C/E_I/E_90、PCA、原型、posterior、关系图
├── m3_cache_check.py        # 哈希、形状、占比、稳定性和可视化审计
├── m3_dataset.py            # I/P90/GT 与所有 token 图同步增强
├── m3_losses.py             # cluster/relation/consistency/boundary
├── m3_train.py              # 双路输出梯度回放和四卡训练
└── m3_eval.py               # 只输入 I 的 validation/test 评估

scripts/
├── prepare_m3_cache.sh
├── check_m3_cache.sh
├── smoke_m3.sh
├── train_m3.sh
└── eval_m3.sh
```

默认输出：

```text
runs/m3_semantic_v1/<run_name>/
├── run_config.json
├── metrics.jsonl
├── checkpoints/
├── validation/
├── gradient_diagnostics.csv
└── training_summary.md
```

---

## 9. 实验顺序

所有主对照都从 Stage 2 最佳权重重新初始化，使用相同样本顺序和 70 次 optimizer update。

| 顺序 | 名称 | 启用项 | 回答的问题 |
|---:|---|---|---|
| 0 | M2-S0 | 零步评估 | 新分布起点 |
| 1 | M2-A | 仅基础重建 | 数据增量基线 |
| 2 | M3-P | 双路 GT + polar consistency | P90 多观测本身是否有效 |
| 3 | M3-S | cluster + relation | 语义分离表示是否优于位置权重 |
| 4 | M3-F | P + S + boundary + withdrawal | 完整 M3 是否稳定提升 |
| 5 | M3-F-shuffle | posterior 数值不变、位置置乱 | 收益是否来自正确语义位置 |

置乱对照按同尺寸样本对 token posterior 和关系目标做固定种子置换，保持数值分布和训练预算不变。只有真实位置稳定优于置乱位置，才能说明模型使用了语义分离信息。

---

## 10. 验收门槛

### 10.1 缓存验收

- 144 个 train ID 完整，validation/test 未参与拟合；
- 所有源文件哈希与纠正后 manifest 一致；
- posterior 有限且每 token 和为 1；
- 四个簇没有容量坍缩；
- 三种随机种子原型对齐后稳定性达到 0.80；
- DoLP 不能在 Qwen 变化接近零时单独制造高 `R`；
- 关系边索引全部在合法 token 范围；
- 水平翻转两次能精确恢复 token 图和边；
- scratch 删除后最终缓存仍可完整校验。

### 10.2 Smoke 验收

- 4 张 GPU 均参与；
- 单卡峰值目标小于 22.5 GiB；
- 显存没有随 step 持续增长；
- latent 重放误差小于 `2e-5`；
- 只有 LoRA_T 有非零梯度；
- `g_I/g_90` 都能改变 LoRA_T；
- 所有损失和梯度比例有限；
- checkpoint 可恢复并继续一步；
- 预测 PNG 可保存并计算 PSNR/SSIM；
- smoke 产物验证后删除，正式训练重新从 Stage 2 初始化。

### 10.3 效果验收

不能只看整图 PSNR/SSIM。至少同时报告：

- PSNR、SSIM、L1、LPIPS；
- 七个长宽比分桶；
- 四个软状态的均衡误差；
- 高反射证据区域残留；
- 低变化区域颜色与纹理保持；
- 高边界区域光晕和双边误差；
- 文字和细纹理固定样本；
- 真实语义位置与等数值置乱对照；
- 教师撤除前后指标。

test split 继续封存，只有模型与超参数在 validation 上定稿后运行一次。

---

## 11. 当前清理记录

2026-09-27 已清理以下已完成且可由脚本重建的 M2 权重：

- `runs/m2_corrected_a_data70_noq20/` 下的 best 与 step 70 权重；
- `runs/m2_corrected_b1_q20grad30/` 下的 best 与 step 70 权重；
- `runs/m2_corrected_b_q20full70/` 下的 best、step 35 与 step 70 权重。

保留：

- 所有 M2 配置、日志、指标、验证预测和三组对比报告；
- 所有复现实验脚本；
- `runs/stage2_transmission_r128/best_transmission_lora.safetensors`；
- 纠正后的 M2 数据集；
- 完整 M2a Q20 GT 特征缓存。

七个权重路径中部分是硬链接，实际释放约 12.8 GiB。清理后个人盘所在文件系统可用空间约 407 GB。

---

## 12. 实施边界

本分支当前只完成 M3 实现设计，不启动缓存生成、Smoke 或正式训练。下一步按以下顺序实现：

1. `m3_cache.py` 和缓存审计；
2. 先为 7 个长宽比桶各选 1 张生成 gate cache；
3. 人工检查 `C` 关系、四类 posterior、`R` 和 `B` 可视化；
4. gate 通过后生成 144 张完整缓存；
5. 实现损失与输出梯度回放；
6. 运行 5 步四卡 Smoke；
7. 依次运行 M3-P、M3-S、M3-F 和置乱对照。
