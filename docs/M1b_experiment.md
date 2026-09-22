# M1b：DoLP 区域语义监督最小实验

## 目的和实验边界

从 Stage 2 的 `best_transmission_lora.safetensors` 分叉，检验冻结 DINOv2 ViT-S/14 的 patch 特征损失是否比单纯继续训练更能改善高 DoLP 区域。WindowSeat 仍只接收 45° RGB 输入；0° GT 和 DoLP 仅参与训练损失。完全不使用 90° 反射加强图。

三组必须使用同一批 50 张训练图、同一初始 LoRA、随机种子、学习率和更新步数：

| 组 | 损失 | 检验什么 |
|---|---|---|
| A `base` | `L1 + 0.2(1−SSIM) + 0.1 edge` | 继续训练本身的效果 |
| B `dolp` | A + `λ × L_DINO(真实 DoLP 蒙版)` | 选中区域的语义监督是否有效 |
| C `shuffle` | A + `λ × L_DINO(等面积像素置乱蒙版)` | 收益是否仅来自多一项损失或蒙版覆盖率 |

先把**完整预测图和完整 GT**缩到 448×336，分别经冻结 DINO 提取第 6 层的 32×24 个 patch token，然后才用下采样蒙版加权逐 patch cosine 距离。预测图的 DINO 路径保留输入梯度，GT 路径不计算梯度。DINO 参数全冻结，位于第二张 GPU。基础损失覆盖全图。C 对每张图作固定随机像素置乱，保留该图精确相同的蒙版像素数。验证和推理均不用 DoLP 作为模型输入。

`DoLP >= 64/255` 是第一版固定阈值，不是反射真值；亮度编码与不同照片的分布需要结合 `m1b_dolp_manifest.json` 的每图覆盖率审查。DoLP 中第 22 张为 8192×6144，其余为 4096×3072，均按完整 4:3 画幅 Lanczos 缩至 512×384。训练和验证沿用原有索引，验证集 `11,12,17`。

## 准备和校准

在 `/share/linmingheng-local/xuke/RMagNet` 执行：

```bash
bash scripts/prepare_m1b.sh
CUDA_VISIBLE_DEVICES=4 bash scripts/diagnose_m1b.sh
CUDA_VISIBLE_DEVICES=2,3 bash scripts/calibrate_m1b.sh
```

校准使用固定训练图 `13`，不更新参数。分别求基础损失和真实蒙版 DINO 损失对预测 RGB 的梯度范数，令语义项的初始梯度约为基础项的 10%。结果写到 `runs/m1b/sem_calibration.json`；B 和 C 读同一个系数。若改 DoLP 阈值、DINO 层、初始权重或输入预处理，须重新校准。A 不读取 DINO。

零更新诊断输出在 `runs/m1b/diagnostics/`：比较 DINO 第 6 层中 `I↔GT` 在蒙版内外的 patch 差异，以及 `blur(GT)↔GT` 的差异。前者检查蒙版是否倾向选中受干扰区域，后者检查这一层是否对涂抹敏感。诊断结果是训练前的可行性信号，不应靠验证图调阈值或挑层。

项目、数据、模型缓存、临时文件与结果都在 `/share/linmingheng-local/xuke`。B/C 的 `CUDA_VISIBLE_DEVICES=2,3` 是一进程双卡：可见卡 0 放 Qwen/LoRA，可见卡 1 放 DINO；这不是把 batch 分到两卡的 DDP。运行前按实际空闲卡调整编号。

## 由你手动运行的实验表

以下每行都是**从相同 Stage 2 best 重新初始化**的独立实验，不续跑上一行。每组每 epoch 为 50 次更新（batch 1）；每个 epoch 自动评估一次 11/12/17，最终保存一个约 3.2 GiB LoRA。三组同一时长结束后再比较。先做短实验，再决定是否值得运行长实验。

| 时长 | 每组更新 | 要测试的问题 | 建议执行条件 |
|---|---:|---|---|
| 1 epoch | 50 | 损失、梯度、显存和验证指标能否完整跑通 | 工程试跑；不据此宣称效果 |
| 2 epochs | 100 | B 是否比同预算 A 改善蒙版内且不伤非蒙版区；C 是否跟着改善 | 第一轮因果探针 |
| 4 epochs | 200 | 2 epoch 的方向是否持续，还是仅有短暂波动 | 仅在前三组都正常时运行 |
| 8 epochs | 400 | 小数据上是否出现过拟合、文字涂抹或幻觉 | 仅在 4 epoch 仍有清晰信号时运行 |

例如 2 epoch 的三组：

```bash
EPOCHS=2 RUN_GROUP=probe_e2 CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_m1b.sh base
EPOCHS=2 RUN_GROUP=probe_e2 CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_m1b.sh dolp
EPOCHS=2 RUN_GROUP=probe_e2 CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_m1b.sh shuffle
RUN_GROUP=probe_e2 CUDA_VISIBLE_DEVICES=2 bash scripts/eval_m1b.sh
```

将 `2` 换成 `4` 或 `8`，并相应改变 `RUN_GROUP`。三组可顺序运行，避免 GPU 显存互相争抢。`run_m1b.sh` 不自动后台运行，不会在本次交付后自行开始多 epoch 训练。想只做一次前后向，可使用 `EPOCHS=1 RUN_GROUP=smoke_step CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_m1b.sh dolp --max-steps 1 --no-save-final`；此时仍会运行 step 0 和 step 1 的三图验证。

## 评价和判定

`eval_m1b.sh` 以固定随机种子推理 Stage 2 起点及 A/B/C，保存 8 位 PNG，再计算逐图和平均的全图 PSNR/SSIM/L1、蒙版内 PSNR/L1、非蒙版区 L1。输出在 `runs/m1b/<RUN_GROUP>/evaluation/`，含 `metrics.csv`、`evaluation.json`、`REPORT.md` 与每张图的 PNG。蒙版 PSNR 通过蒙版内 MSE 求得；没有把蒙版外设零后套用整图 PSNR/SSIM。

判断 B 是否有效，至少要求相较 A：蒙版内误差在 11/12/17 的多数图下降、全图 PSNR/SSIM 无明显恶化、非蒙版区误改无明显增加；再人工并排检查窗框、细纹理、文字和新增物体。C 若与 B 改善接近，说明真实 DoLP 的位置选择尚未得到支持。三张验证图不足以证明跨场景泛化；DINO 特征相似也不等于真实纹理、可读文字或无幻觉。

论文的“约一天”对应作者自己的数据与设备。其主实验约 11,000 更新，涉及 25,000 张 PBR 加真实数据；本实验独立训练图只有 50 张，2/4/8 epoch 分别为 100/200/400 更新，目标是可行性和消融，不是复现论文训练量级。

## 已完成的零更新检查和单步试跑（2026-09-22）

- 53 组 DoLP 已缩图并按阈值 64 生成蒙版；训练图覆盖率范围 0.0055–0.4663，中位数 0.1704。每张源图及派生图的 SHA-256、覆盖率见 `datasets/rmagnet_stage1_512x384/m1b_dolp_manifest.json`。
- DINO 第 6 层诊断：训练集 `I↔GT` patch cosine 距离在蒙版内/外平均为 **0.1603/0.0731**；`blur(GT)↔GT` 为 **0.2282/0.2136**。第 6 层对 9×9 模糊有响应，真实蒙版也更偏向输入与 GT 差异区。逐图值见 `runs/m1b/diagnostics/patch_distances.csv`。这只支持继续做 A/B/C 对照，不构成效果结论。
- 固定训练图 13 的零更新梯度校准得到 `λ=0.0029368706`，对应基础损失梯度范数 0.001514、语义梯度范数 0.051554，目标梯度比例 10%。B/C 共用这一值。
- `runs/m1b/smoke_step/dolp` 完成真实蒙版组 **1 次**更新，训练样本 ID 65；损失有限，LoRA 梯度范数 1.519，Qwen 峰值 18.25 GiB，DINO 峰值 0.24 GiB，更新本身约 7.5 秒。该试跑没有保存 LoRA，也没有继续跑第 2 步。
- 三图浮点验证均值从起点的 23.1422 dB / 0.82193 SSIM 变成单步后的 23.1353 dB / 0.82180；单步波动不能解读为模型优劣。正式对比请运行三组同更新量，再用保存后 8 位 PNG 的评估脚本。
