# 官方 WindowSeat 起点的 C1 + M5：过滤数据上的重新训练

日期：2026-10-09。代码分支：`experiment/windowseat-c1-m5-filtered`。

## 1. 实验目标与模型身份

沿用既有 C1 语义条件联合编辑 + M5 像素残差方案，**重新从官方 WindowSeat 初始化**。本轮不续接 Stage 2、M4-best 或历史 C1-best 的训练权重。原始 Qwen NF4 主干和 VAE 始终冻结；官方 Transmission LoRA 是学生的可训练起点，也是固定损失教师的初始权重。

官方 LoRA：

`/share/linmingheng-local/xuke/.cache/huggingface/hub/models--huawei-bayerlab--windowseat-reflection-removal-v1-0/snapshots/c1f59ca02bff68535c976e55e17147b3d9323309e/transformer_lora/pytorch_lora_weights.safetensors`

SHA-256：`2f04225c5be1cac69a5d76faf4c95515d5503210cfee73c0bb1e79309f5a74e9`。

官方 checkpoint 含 1446 张量；本项目完整 LoRA 含 1448 张量。差别仅是官方未保存的 `proj_out.lora_A.default.weight` 与 `proj_out.lora_B.default.weight`。加载器仅对以上指定官方路径允许这两个缺失项，并验证新建 B 为全零，维持官方的零残差行为；其他 checkpoint 仍要求完整对应。所有新 C1 checkpoint 保存完整学生 LoRA 和语义模块。

旧内部字段 `base_m4_sha256`、缓存格式名 `m4-best-lora-multilayer-v1` 为兼容保留，不能据此认为本轮教师是 M4；真实身份以 `base_initial_sha256`、`base_family`、教师路径及哈希为准。

## 2. 过滤后的数据

数据根：`/share/linmingheng-local/xuke/datasets/rmagnet_sma_dataset2/`。

| 划分 | 样本数 | 使用方式 |
| --- | ---: | --- |
| train | 168 | 离线语义缓存、C1 和 M5 优化 |
| validation | 20 | 每 epoch 评价，按宏平均 L1 选择 best |
| test | 19 | 封存；训练期间不评价、不选权重 |

按本地 `D:\Develop\PhotoManager\data_set` 文件名保留名单过滤，拍摄编号上限为 162，之后的组保留；原划分不重排。保留纠正后的服务器角色和哈希，**不按历史 JPEG 后缀重新交换标签**。`blended` 是输入 I，`transmission_layer` 是 GT，`reflection_90` 是 P90，`dolp` 是 DoLP。使用现有动态尺寸和长宽比分桶，不重复缩图或裁剪。缓存与训练都禁用改变空间位置的数据增强。

## 3. 自动阶段和预算

入口：`scripts/run_windowseat_c1_m5.sh`。默认预算为 **C1 50 epoch + M5 50 epoch**，两段分别完整遍历训练集；可分别设置 `C1_EPOCHS` 和 `M5_EPOCHS`。这是两段训练，并非 100 epoch 联合端到端训练。缓存、5 epoch 的小型语义记忆预拟合不计入生成器训练 epoch。

| 阶段 | 设备 | 内容 | 产物 |
| --- | --- | --- | --- |
| 数据与模型检查 | CPU | 配对、尺寸、哈希、划分、官方权重身份、剩余空间 | preflight_data.json |
| 官方教师缓存 | 可选 1–4 张 GPU | 168 个训练样本，按 CACHE_GPU_DEVICES 数量分互斥片 | 层特征、热区门控、manifest |
| 语义记忆准备 | 第一个缓存 GPU | Q37 压缩与 CleanMemory 训练，默认 5 个特征 epoch | memory.safetensors + report.json |
| C1 联合训练 | GPU 0–3 | 学生 Transmission LoRA 和 4 个语义条件读取器训练 | best_sma/latest_sma |
| C1-best RGB 缓存 | GPU 0 | 用本轮 C1-best 对 207 个输入单图推理并保存 RGB8 | 每图 PNG 与 manifest |
| M5 训练 | GPU 0 | 固定 C1-best，训练小型像素恢复器 | best/latest |

RGB 缓存可覆盖验证和测试输入，但不把它们用于拟合；M5 训练只加载 train，验证只加载 validation，训练流程不读取 test GT 算指标。GT 哈希仅用于完整性审计。

C1：每卡 batch 1、累积 1，有效 batch 4，每 epoch **42 次更新**；50 epoch 共 **2100 次更新**。M5：单卡 batch 1、累积 4，同为 42 次更新/epoch 和 2100 次更新。两者无早停，无额外 MAX_STEPS 截断。

## 4. 需要的 Qwen 层缓存

官方 WindowSeat LoRA 开启，教师冻结，VAE posterior mode，固定 timestep 499，沿用官方缓存文本 embedding。

* 前段：Q16/Q20，保存 I 与 GT 特征，用于保留与恢复纹理。
* 中段：Q37/Q39/Q41，保存 GT 特征；额外保存 Q37(I)、Q37(P90)，用于条件记忆和关系监督。
* 后段：Q52/Q54/Q56，比较 I 与 GT，形成反射变化门控；仅保留门控及一致程度，不保留这三层完整特征。

离线教师不随学生更新。训练时为当前预测计算可微教师特征；教师权重分阶段从 CPU 暂存切入，在教师反传结束后恢复学生 LoRA，再执行学生反传。禁止误用历史 M4 层缓存。

缓存格式 BF16 完整特征、FP16 门控；每条记录包含图像来源哈希、教师哈希、尺寸、token 网格、缓存哈希、显存峰值。最终 manifest 完整检查 168 条且与训练划分一致。

## 5. C1 网络与损失

Q37 图像 token 经冻结的 PCA 到 256 维、冻结 CleanMemory，得到可读取空间条件。4 个读取器挂在 Q39、Q41 图像 FFN 的入口和出口 LoRA-A 输出上，条件影响学生可训练 LoRA 的编辑行为；条件记忆与主干不更新。

$$
z=Ah,\qquad y_{LoRA}=B\left((1+\gamma\,g(C)\tanh U(C))\odot z\right),\quad \gamma=0.25\tanh\gamma_{raw}
$$

gamma 从 0 初始化。学生 LoRA LR `5e-6`；条件读取器 LR `1e-4`；gamma LR `5e-3`。Warmup 20 更新，之后 Cosine，梯度裁剪 1.0。量化主干冻结，计算 BF16，条件模块/梯度统计 FP32。学生采用 paged 8-bit AdamW，优化器状态在更新间移到 CPU。

基础重建与偏振一致性：

$$
L_{rec}(X)=\|F(X)-GT\|_1+0.2(1-SSIM(F(X),GT))+0.1L_{edge}(F(X),GT)
$$

$$
L_{base}=\tfrac12[L_{rec}(I)+L_{rec}(P90)]+0.1L_{polar}
$$

辅助项沿用 C1 方案：后段热区上的空间恢复/低响应保持；Q16/Q20 特征保持与恢复；Q37/Q39/Q41 中心化内容余弦距离和四邻域关系差异，其中语义内部为 `0.7 content + 0.3 relation`。核对当前 C1 实际代码后确认：空间权重来自 Qwen 后段差异和层间一致性，**DoLP 文件只检查配对和哈希，不直接进入当前 C1 loss**；P90 与偏振一致性项继续使用。

$$
L=L_{base}+\lambda_s L_{spatial}+\lambda_t L_{texture}+\lambda_m L_{semantic}
$$

三个辅助项**在预测图像梯度空间**控制强度，目标各为基础输入分支梯度的 8%，前 36 更新渐入，EMA 0.9，合计梯度范数上限 25%。这不是参数梯度严格 8%，也不是三个固定标量 loss 系数。实际比例写入逐步日志。

第 2 epoch 起每 8 更新，执行当前学生同权重的条件 off/on 反事实比较，根据相对于 GT 的局部改善训练可靠性门控；此校准梯度仅进入 gate，DDP 平均后范数不超过该 gate 主梯度的 10%。测试时只需 I，不需要 GT、P90 或 DoLP。

为 24GB GPU 开启梯度 checkpoint、分阶段反传和保存激活的 CPU 卸载。CPU 内存不足或 GPU 报错时流程退出并记录失败，不自动换到他人 GPU。

## 6. M5 像素恢复器

从本轮 C1-best 生成 RGB8 T0，固定上游。四尺度 width32 NAF 风格恢复器约 87,052 参数，输入 I、T0、I−T0；零初始化输出头，保留 C1 起点，预测有界有符号残差。各尺度幅度上限 0.03/0.05/0.08/0.12，最终总残差压至 ±0.20。

$$
T_{final}=clip(T_0+\Delta(I,T_0),0,1)
$$

$$
L_{M5}=L_{Charbonnier}+0.2L_{SSIM}+0.1L_{edge}+0.1L_{band}+0.05L_{keep}+0.05L_{identity}
$$

identity 每 10 个样本执行一次；keep 针对 T0 与 GT 差异小于 0.02 的区域。FP32 AdamW、LR 2e-4、weight decay 1e-4，1 epoch Warmup 后 Cosine 到 0.1 倍 LR。内部短探针使用独立初始化，结束删除探针 PNG/checkpoint，正式训练重新初始化。验证按保存后的 8 位 PNG 计算指标；零残差 baseline 也有资格成为 best，避免宣称一定改善。

## 7. 路径、空间与运行

默认组名：`ws_c1_m5_filtered_e50`。

```text
RMagNet/data_cache/ws_c1_m5_filtered_e50_layers/
RMagNet/data_cache/ws_c1_m5_filtered_e50_rgb8/
RMagNet/runs/ws_c1_m5_filtered_e50/
  console.log
  status.json
  pipeline.pid / pipeline.exit_code
  preflight_data.json
  logs/{memory,c1_preflight,c1,rgb_cache,m5}.log
  memory_prepare/memory.safetensors
  c1/{best_sma,latest_sma}.safetensors
  m5/{best,latest}.safetensors
```

C1 两个模型文件合计约 6.4GiB，原子写临时文件还需约 3.2GiB；层缓存约 6–7GiB（以生成 manifest 为准），RGB/预览/日志另占小量空间。启动闸门要求至少 20GiB 可用；未新增大模型下载，不使用 Docker，不修改系统环境。只保留每阶段 best/latest 模型、固定记忆组件和实验记录，不保存 optimizer，不累积 epoch 权重，训练无法精确恢复优化器状态。

```bash
cd /share/linmingheng-local/xuke/RMagNet
C1_EPOCHS=50 M5_EPOCHS=50 CUDA_VISIBLE_DEVICES=0,1,2,3 CACHE_GPU_DEVICES=2 OMP_NUM_THREADS=1 \
  bash scripts/run_windowseat_c1_m5.sh
```

后台启动使用 tmux，日志、phase 状态及 exit_code 判断完成。启动时 0/1/3 被其他进程占用，因此本次缓存选择 GPU 2。C1 启动前要求 0–3 每卡至少 22000MiB 空闲；不满足时进入 waiting_c1_gpus，每 300 秒只读检查显存，满足条件才加载模型。不会终止其他进程。C1 若失败不启动 M5；cache 若缺失不启动训练。已结束组拒绝重复覆盖，需明确处理失败原因并另设 RUN_GROUP。

已验证官方重载兼容性：训练样本 `100_2531_1536` 在官方后端初始化后、经本轮加载器重载后的八个缓存层特征逐元素最大绝对差均为 0，缺失输出投影 B 的非零元素数为 0，GPU 2 峰值约 16.084GiB；记录 `tmp/windowseat_c1_m5_20261009/official_reload_parity.json`。这证明本轮加载策略未改变该样本的选层输出，不代替长训练稳定性验证。

历史 C1 仅训练到 466 更新附近出现显存不足，M5 的历史收益很小且 real20 有退步；本轮 50 epoch 的结果未知，验证只选模型，不预设一定优于官方或 M4。最终测试需另行明确执行，不在本脚本训练期间自动运行。
