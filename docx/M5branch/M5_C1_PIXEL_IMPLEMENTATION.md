# M5-R：固定 C1-best 后的像素细节恢复

2026-10-09；分支 `experiment/m5-c1-pixel`。已实现、通过短流程检查、已后台启动首轮训练；尚未完成效果评估。本页是当前实施说明，其他 M5 文档是历史设计。

## 1. 这次实际训练什么

按本次要求，固定 `runs/sma_c1_e10/best_sma.safetensors`，即 C1 epoch 2 / step 102 的联合权重：Transmission LoRA + ConditionedSMA。VAE、Qwen、该 LoRA 和语义条件模块全部保持原状。训练期间直接读取 C1 输出缓存，不加载这些模型，因此没有对 C1 续训，也没有改变其历史偏振监督配置。

先前 C1-best 在当前 26 张测试图上的 PSNR/SSIM 为 24.1925 / 0.798306，M4-best 为 24.2355 / 0.799657；real20 分别为 25.2884 / 0.817175 和 25.5068 / 0.819492。C1 尚未证明优于 M4。本轮选择 C1 是为了按指定上游检验独立像素恢复是否有用，不能预先声称它会超过 M4。

$$
I\xrightarrow{\text{固定 C1-best}}T_0
\xrightarrow{\text{RGB8 PNG量化接口}}T_0^{8bit}
$$

$$
[I,T_0^{8bit},I-T_0^{8bit}]\xrightarrow{\text{M5-R}}\Delta,
\qquad T_1=\operatorname{clip}(T_0^{8bit}+\Delta,0,1)
$$

测试输入仍只有普通图 I。T0 是上游内部结果，GT 只用于监督/评测；本版 M5 不读取 P90、DoLP、Qwen特征缓存。语义由固定 C1 提供，新增网络承担可观测残余的像素修正。还没有加入收益 gate、SVM、OCR、RAG 或生成式补字。

## 2. 数据与缓存

- 复用纠正标签后的 `datasets/rmagnet_sma_dataset2`：204 train / 26 validation / 26 test；按现有拍摄场景划分，不重划分。
- 配对角色以已审核 manifest 为准：`blended` 为 I，`transmission_layer` 为 GT；不再按曾反标的原始文件名猜角色。
- 完整保留当前处理尺寸和长宽比，不裁剪、不再缩图。当前约 20 万像素，不能外推为 4K 文字恢复验证。
- `data_cache/m5_c1_rgb8_v1/`：256 张 RGB PNG + manifest，总约 45 MB。训练图重新推理；验证/测试图仅在 C1 权重和数据 manifest 哈希相同且既有 PNG 存在时复用，否则重新推理。
- 固定后验 `mode`、条件 `learned`、种子 2026。保存采用 clamp → 乘255 → torch.round → uint8；训练和推理都读回同一 RGB8 接口。
- manifest 记录 C1/data/input/GT/预测 SHA-256、尺寸、split 和缓存来源；训练入口重验数据与缓存文件哈希。C1 生成 T0 时从不接收 GT。

缓存脚本支持同身份断点补齐，拒绝来源改变或已记录缓存损坏。缓存已完成，后台下载或新的大模型权重均不需要。

## 3. 恢复器结构

实现：`src/rmagnet/m5_pixel.py`；width32，**87,052 个可训练参数**，FP32。采用轻量原型检验残差是否可学习，参数量比此前约百万级目标更小。

三次下采样形成四个尺度。下采样前使用 separable 5×5 binomial 高斯滤波、reflect padding，stride2；上采样为 bilinear / align_corners=False。每尺度两个 NAF 风格块，包含通道归一化、depthwise convolution、SimpleGate、全局通道注意力和残差缩放。粗尺度特征逐级融合到细尺度，粗尺度平均池化上下文共享给三个细尺度。

$$
G_{l+1}(X)=\downarrow_2(K*G_l(X)),\qquad
H_l(X)=G_l(X)-\operatorname{up}(G_{l+1}(X))
$$

四个零初始化输出头预测有正负方向的残差；三个细节带幅度上限分别为 0.03 / 0.05 / 0.08，低频为 0.12。重建后用平滑限制约束总修正：

$$
\delta_l=a_l\tanh(h_l),\qquad
\Delta=0.20\tanh\left(\frac{\operatorname{PyramidReconstruct}(\delta_0,\delta_1,\delta_2,\delta_3)}{0.20}\right)
$$

因此初始输出逐像素等于 T0；整体修正幅度严格小于 0.20。记录绝对修正 ≥0.19 的像素比例。网络不直接拷贝 I 的高频，避免把反射文字/车辆边缘重新贴回。

## 4. 损失的确切定义

$$
L=L_{Charb}+0.2L_{SSIM}+0.1L_{edge}+0.1L_{band}+0.05L_{keep}
+0.05\mathbf{1}_{identity}L_{identity}
$$

$$
L_{Charb}=\operatorname{mean}\sqrt{(T_1-GT)^2+10^{-6}},\qquad
L_{SSIM}=1-SSIM(T_1,GT)
$$

edge 是横/纵相邻像素差分与 GT 差分之间 L1 的均值。band 为四尺度平均 L1：前三尺度预测带对齐 GT 与 T0 的拉普拉斯带差，第四尺度对齐低频差。

$$
L_{band}=\frac14\left[\sum_{l=0}^{2}\|\delta_l-(H_l(GT)-H_l(T_0))\|_1
+\|\delta_3-(G_3(GT)-G_3(T_0))\|_1\right]
$$

$$
M(x)=\mathbf{1}\left[\operatorname{mean}_{RGB}|T_0(x)-GT(x)|<0.02\right],\qquad
L_{keep}=\frac{\sum_{x,c}M(x)|\Delta_c(x)|}{\max(1,3\sum_xM(x))}
$$

每 epoch 第 0、10、20…个训练样本额外执行 `refiner(GT,GT)`，identity 项是其新增修正绝对值均值。这个 GT identity 分支仅在训练期出现，用来保护真实高光、文字、已正确的结构。I/T0/GT 同步随机水平翻转，identity 使用同方向 GT。

SSIM 沿用工程历史口径：11×11 均匀窗口、stride1、zero padding5，常数 0.01² / 0.03²；不是 Gaussian-window SSIM。损失在浮点输出上求梯度，验证在落盘 RGB8 上计算。现有系数是首轮初值，不是已证明最优的组合；不加新的 Qwen 语义项以便观察像素模块的独立作用。

## 5. 已完成的流程/显存检查

`runs/m5_c1_pixel_e30/preflight.json`：

- 检查四张图，包括最大面积和长宽比两端。
- 零初始化最大误差 **0.0**。
- 损失/梯度有限且非零，168 个 state tensor 更新。
- safetensors 重载预测完全一致；PNG 保存、PSNR/SSIM 可计算。
- 检查峰值 allocated **1.568 GiB** / reserved **1.633 GiB**；临时检查权重/PNG 已删除，仅保留检查报告。
- 正式训练重新随机初始化零输出模型，从不使用检查更新后的权重。初期正式峰值约 **2.97 GiB**，GPU0 上已完成至少20次更新，显存满足24GB限制。

本版用单张空闲 GPU0 就足够，不需要 DDP 复制 Qwen。小网络并不等于训练已有效；性能结论等待验证和之后的测试。

## 6. 正式预算、早停与保留策略

| 项目 | 配置 |
|---|---|
| 最大 epoch | 30，可通过环境变量/位置参数调整 |
| 完整 epoch | 204 张训练图各使用一次 |
| batch / accumulation | 1 / 4，有效 batch4，每 epoch51更新 |
| optimizer | AdamW，LR 2e-4，weight decay1e-4 |
| scheduler | 一整个 epoch 线性 warmup，之后 cosine 至0.1×LR |
| precision / clip | FP32 / 1.0 |
| validation | 每 epoch，26 张；训练过程中不运行测试集 |
| best criterion | 保存后 PNG 的逐图宏平均 L1，改善幅度 >1e-6 |
| early stopping | 连续4次验证无改善，至少完成5epoch |
| retention | 只有 best.safetensors / latest.safetensors；不存 optimizer |

额外保留 epoch0 的零修正基线作为 best 候选。如果训练始终不改善，best_epoch=0 会如实记录，不能把无变化输出称为训练成功提升。latest 为实际最近 epoch。固定覆盖 best/latest 验证 PNG，不积累每 epoch 权重或输出；权重每份约0.4 MB。

输出：`runs/m5_c1_pixel_e30/`，包含 config.json、preflight.json、train.jsonl、epochs.jsonl、baseline_validation、best/latest 验证、best/latest_metrics.json，结束后写 training_summary.json/md。训练过程中日志展示未完成状态，尚无最终报告。

启动记录：`runs/m5_launch/train.pid`、`train.log`、`train.exit_code`（结束后生成）。使用 nohup，退出 SSH 不影响训练。不要在同名非空输出目录重复启动；新实验改 RUN_NAME。

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=30 RUN_NAME=m5_c1_pixel_e30 \
OMP_NUM_THREADS=1 bash scripts/train_m5_c1.sh

# 新预算用新目录，例如
CUDA_VISIBLE_DEVICES=0 RUN_NAME=m5_c1_pixel_e10 bash scripts/train_m5_c1.sh 10

# 训练结束后手动测试，默认best；首次执行后不允许覆盖非空评测目录
CUDA_VISIBLE_DEVICES=0 RUN_NAME=m5_c1_pixel_e30 bash scripts/eval_m5_c1.sh
CUDA_VISIBLE_DEVICES=0 RUN_NAME=m5_c1_pixel_e30 CHOICE=latest bash scripts/eval_m5_c1.sh
```

## 7. 单图与评估边界

`src/rmagnet/m5_apply.py` 可由普通图生成固定 C1 中间 PNG，再执行 M5；也支持提供已生成的 C1-best PNG 以节省大模型推理。后者的来源由调用者保证。首版限现有处理尺度（≤30万像素、轴长16倍数），不静默裁剪或压缩；4K 需要另建带整图上下文的分块流程。

```bash
# 在 scripts/sma_env.sh 提供的 uvpython 环境中
uvpython -m src.rmagnet.m5_apply --input /absolute/path/I.png \
  --t0 /absolute/path/C1_output.png \
  --checkpoint runs/m5_c1_pixel_e30/best.safetensors \
  --output /absolute/path/M5_output.png
# 去掉 --t0 会执行C1；模型/数据身份仍固定。
```

正式比较需同时保留 C1、M5 的逐图 L1/PSNR/SSIM、边缘误差、低变化保持误差、高变化恢复误差；训练不依据测试指标选择设置。当前测试集和 real20 已多次用于研究，且历史 Stage2 有旧场景重叠，仍属回顾性比较，不能声称全历史盲测。先确认候选修正本身有收益，再考虑独立校准的收益 gate；本次不训练该 gate。原生4K纹理、固定文字 ROI 和新场景泛化评测是后续工作，不把锐化或补造纹理当作真实恢复。
