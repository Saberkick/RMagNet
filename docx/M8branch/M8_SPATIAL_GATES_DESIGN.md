# M8：冻结 M4，在前、中、后层学习空间门控

## 目标

从 M4 final best 出发，在 Qwen 第14、37、54层（索引13、36、53）新增空间门控。冻结原 DiT、M4 LoRA、VAE Encoder/Decoder，只训练三组小型门控头。首轮5 epoch，无早停，只评估25张验证图，不运行封存测试或 real20。

M7 在封存24张中仅改善 PSNR 约0.026 dB，同时 SSIM/L1 略退化；尾部补救尚未带来整体改善。M8 把控制位置提前，检验是否能在生成过程中减少过度修改。前/中/后层的纹理、语义、编辑强度角色仍是实验假设，不把差异热图当作语义证明。

## 来源与隔离

- 分支：`experiment/m8-m4-spatial-gates`。
- 目录：`/share/linmingheng-local/xuke/RMagNet-M8`。
- 基于当前 M7 分支代码创建，只复用工程工具，不使用 M7 恢复模块或 M7权重。
- M4权重：`RMagNet/runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors`。
- SHA-256：`897282b1bb9cfe61f96530df72edcf8a44a066bb819a3663e9100862aefdb2a3`。
- 基础 Qwen 和 WindowSeat 文本条件沿用 `check_snapshots` 的已有模型锁，加载过程全离线，无新增大模型下载。
- 依赖复用 uv管理的 Python3.12环境 `envs/windowseat-py312`，不修改系统包、驱动或其他任务。

## 结构

原始 Qwen 的时间条件门控保持不变，新增图像内容驱动的空间门控，仅作用于图像流的两次残差更新。文本流仍按原算法执行。

$$
H'_l=H_l+G^{attn}_l\odot\Delta H^{attn}_l
$$

$$
H_{l+1}=H'_l+G^{mlp}_l\odot\Delta H^{mlp}_l
$$

残差增量包含 Qwen 原有时间条件的通道门控；新增门控不替换它。对每个选定block，从其输入H预测两个空间门控，分别调制Attention和MLP更新，通道广播。两张门控都由该block输入产生。

```text
H [1,N,3072]
 -> LayerNorm -> Linear(3072,64) -> SiLU
 -> 恢复为 H/16 × W/16 空间网格
 -> depthwise 3×3 Conv -> SiLU -> 1×1 Conv(64,2)
 -> sigmoid -> Attention / MLP 两张连续门控
```

$$
G_l=0.75+0.5\operatorname{sigmoid}(P_{\phi,l}(H_l))
$$

输出卷积零初始化，因此初始门控均为1，与原M4行为一致。先限制范围为[0.75,1.25]，抑制/增强幅度最多25%，不清零隐藏特征或token。原始背景与反射混合在同一token，直接清零存在一同删除真实内容的风险。

门控计算和参数保持FP32；残差按原BF16乘法先计算，再用FP32空间门控调制，最后回到BF16。这样避免很小的门控偏移直接先被量化为1。初始化阶段用实际大尺寸训练样本检查启用/关闭门控后的latent必须完全相等。

## 梯度与缓存

冻结权重不等于关闭梯度：第14层之后需要允许梯度通过冻结DiT传回门控，Decoder也需允许对输入latent求导。

- VAE编码、输入投影与前13个block在 `no_grad` 下执行。
- 第14至60个block使用non-reentrant gradient checkpointing。
- Decoder使用M7已验证的函数级checkpoint；不调用当前版本不支持的VAE内置接口。
- 原DiT、原LoRA及VAE的参数必须全部 `requires_grad=False`，每更新检查无梯度。
- 门控输出参数第一步必须变化，三层门控每更新都须有有限非零梯度。
- 没有复用M7的后段隐藏特征缓存，没有新建Q20等语义缓存。前段门控改变下游特征，下游必须实时重算。

## 数据

复用 `datasets/rmagnet_sma_dataset3` 的纠正标签、现有动态尺寸与按拍摄组划分：204 train、25 validation、24封存test。

- 普通 I 是唯一模型输入。
- GT仅用于损失、验证和文件审计。
- P90与DoLP不参与本轮模型输入或损失。
- 不重新缩图、不改划分、不翻转缓存特征、不使用错配标签。
- 本轮无几何增广，以简化门控网格和数值对照。

## 损失

$$
\rho(e)=\sqrt{e^2+10^{-6}}
$$

$$
L_{rec}=\operatorname{mean}\rho(\hat T-GT)
$$

$$
L_{edge}=\frac12\left[\operatorname{mean}\rho(\nabla_x\hat T-\nabla_xGT)+\operatorname{mean}\rho(\nabla_y\hat T-\nabla_yGT)\right]
$$

$$
L_{gate}=\frac16\sum_{l\in\{14,37,54\}}\sum_{b\in\{attn,mlp\}}\operatorname{mean}(G^b_l-1)^2
$$

$$
L=L_{rec}+0.10L_{edge}+0.01L_{gate}
$$

重建项提供整体GT目标；边缘项监督笔画/边缘；门控正则鼓励必要位置才改变原M4。首轮不加入额外语义教师损失、不强制稀疏蒙版、不要求三层蒙版相同。系数是预先规定的起点，不依据封存测试调整。

## 训练预算与多卡方式

采用一份冻结模型分布在GPU1、2、3的**模型并行**；不是3份模型的DDP。原block0–19、20–39、40–59分别放到三个逻辑GPU，VAE与输入投影在逻辑0，输出头在逻辑2。

| 项目 | 首轮 |
|---|---|
| GPU | 物理1、2、3；脚本支持2–4卡模型并行 |
| 每次前向 | 1张，保持动态尺寸 |
| 梯度累积 | 6张 |
| 有效batch | 6 |
| epoch | 5，完整遍历204张训练图 |
| 每epoch更新 | 34 |
| 总更新 | 170 |
| 学习率 | 1e-4，所有门控相同 |
| Optimizer | AdamW，weight decay 1e-4 |
| 调度 | 10步warmup，cosine至0.1倍LR |
| 精度 | 原DiT NF4；激活BF16；门控/optimizer FP32 |
| 梯度裁剪 | 1.0 |
| Early stopping | 无 |
| 验证 | 每epoch一次，RGB8保存结果 |
| best选择 | 验证RGB8平均L1，min_delta=1e-8；允许epoch0基线 |
| 保留 | best门控；latest门控+optimizer/scheduler |

当前GPU0–3上还有其他项目，本实验不终止它们。启动检查每张选择GPU剩余显存至少18000MiB，使用模型并行和checkpoint降低额外显存。启动后实际峰值、速度以日志为准，不把参数量少误认为反向传播显存必然很小。

## 日志与结果

```text
runs/m8_m4_gates_14_37_54_e5/
  config.json identity_check.json
  logs/train.log train.jsonl metrics.jsonl
  m4_baseline_validation/predictions/ metrics.csv metrics.json
  best/gates.safetensors state.json
  latest/gates.safetensors trainer.pt state.json
  best_validation/ latest_validation/
    predictions/ metrics.csv metrics.json
    gates/  # 固定3张验证图，3层×2分支
  best_metrics.json latest_metrics.json
  training_summary.json pipeline.exit_code
```

记录rec、edge、gate正则、总loss、三层梯度范数、各层门控范围/均值/空间标准差/触边比例，各GPU峰值、样本ID和实际更新数。门控预览固定灰度标度：0.75黑、1灰、1.25白，表示修改强度而非反射概率；不是每张图独立拉伸。

指标基于保存后的RGB8 PNG宏平均，SSIM沿用项目uniform11×11、zero padding5。M4基线在同一执行方式下重新生成，后续验证使用固定baseline文件；不混用历史不同推理协议的结果。

## 运行和续训

```bash
cd /share/linmingheng-local/xuke/RMagNet-M8
CUDA_VISIBLE_DEVICES=1,2,3 EPOCHS=5 OMP_NUM_THREADS=1 \
  bash scripts/run_m8_spatial_gates.sh
```

首次检查通过后使用专用tmux会话后台运行，并断开SSH；不继续监控完整训练。

```bash
RESUME=1 EPOCHS=10 CUDA_VISIBLE_DEVICES=1,2,3 \
  bash scripts/run_m8_spatial_gates.sh
```

续训要求相同GPU数量、累积次数、数据和M4身份，并检查latest门控cursor与trainer状态一致。只保存门控权重，推理必须同时加载固定M4与本门控代码；没有复制第二份完整DiT。

## 验收与判断

首先验收初始化等价、三层门控非零梯度、参数变化、冻结参数无梯度、有限loss、显存正常和latest保存。随后看验证集是否同时改善整体与局部指标；若门控接近常数1，要区分真实无需修改、信号弱或训练不足。若长期触及范围边界，也不立即放宽，应先判断是否在破坏原M4结果。

只有验证集结果明确后再授权封存测试。本实验不保证恢复VAE已丢失或原图不可见的真实纹理，也不将门控可视化自动解释成物体语义识别。
