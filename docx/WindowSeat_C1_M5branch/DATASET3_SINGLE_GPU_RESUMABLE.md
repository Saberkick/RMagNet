# data_set3 合并后的单卡 C1 + M5：20 epoch，可续训

2026-10-10；分支保持 `experiment/windowseat-c1-m5-filtered`。入口 `scripts/run_windowseat_c1_m5_single.sh`。原入口在选择单张 GPU 时转到此脚本。模型来历、语义条件结构和各项损失沿用 [官方起点设计](IMPLEMENTATION_AND_TRAINING.md)，本文件覆盖其旧四卡运行方式、旧 168/20/19 计数和不保存优化器的策略。

## 数据与上一轮状态

上一轮完整生成了 168 张层缓存和 5 epoch 的 CleanMemory，但生成器训练尚未启动，流程最后停留在 `waiting_c1_gpus`。当时 0/1/3 显存不足，不是已完成 C1 训练后失效。等待日志和配置归档后清理旧运行目录；完整旧缓存只在新缓存检查通过后清理，已经复用的张量用 hardlink 保存。

新增 `D:\Develop\PhotoManager\data_set3`：46 组、184 文件、25,723,175 字节；与旧 207 个完整样本 ID 无重叠。用户已确认：无后缀 `.jpg` 是 I，`_GT.jpg` 是 GT，`_90.jpg` 是 P90，`_DoLP.png` 是 DoLP。旧数据保留纠正后的 manifest 角色，不能按历史文件后缀重新解释。

合并目录：`/share/linmingheng-local/xuke/datasets/rmagnet_sma_dataset3/`。

| 划分 | 原样本 | 新增 | 合计 |
| --- | ---: | ---: | ---: |
| train | 168 | 36 | 204 |
| validation | 20 | 5 | 25 |
| test | 19 | 5 | 24 |

按拍摄编号整组划分，原组继承原来的划分，新组约 80/10/10，长宽比分桶用于排列训练顺序。扫描源文件与 PNG 哈希，拒绝跨组字节重复和跨划分相同 GT；检查 RGB/DoLP 模式与配对尺寸。保留全部旧划分，不把旧测试图移入训练集。新增测试集与历史测试集不同，后续均值比较需说明集合。

图片目标面积约 512×384，长宽比误差上限 3%，两边为 16 的倍数，无裁剪、无 padding、无放大，RGB Lanczos、DoLP BOX。旧 PNG 用 hardlink 引用，不重复缩图。新增尺寸不固定，如 320×192、400×496、512×384、608×336。训练不改变位置，以免破坏缓存对应。

## 官方层缓存与语义记忆

原生 Qwen + 官方 WindowSeat rank128 LoRA 为固定教师，SHA-256 `2f04225c5be1cac69a5d76faf4c95515d5503210cfee73c0bb1e79309f5a74e9`。不使用历史 M4/C1 权重初始化。

缓存 Q16/Q20 的 I/GT、Q37/Q39/Q41 的 GT，并保存 Q37(I/P90)；Q52/Q54/Q56 的 I/GT 差异形成门控和一致程度。固定 timestep 499，VAE posterior mode。旧 168 份通过源图/教师/缓存哈希检查复用，补 36 份；新 manifest 对应完整 204 张 train，总体积 **7.746GiB**。验证和测试不参与层缓存、PCA 或记忆拟合。

CleanMemory 重新在 204 张 train 上拟合 5 个小型特征 epoch，属于准备阶段，不计入 C1 的 20 epoch；旧 PCA/记忆不直接继承。当前 C1 的 DoLP 仅参与配对与哈希审计，实际空间权重来自 Qwen 差异；P90 重建及偏振一致性继续使用。

## 单卡训练及预算

当前只使用物理 **GPU 2**，进程内映射为 `cuda:0`，不初始化 DDP。batch 1、累积 4，4 张图可以各自有不同尺寸，逐张前向，不将不同尺寸堆成一个张量。

$$
B_{effective}=1\times 4=4,\qquad U_{epoch}=204/4=51
$$

默认依照两段流程：**C1 20 epoch，再固定本轮 C1-best 训练 M5 20 epoch**，各 1020 次优化更新，无早停。`C1_EPOCHS`、`M5_EPOCHS` 可独立调整；`STAGES=c1` 可只执行生成器阶段。5 epoch 的特征准备单独记录。

### C1 保持的配置

冻结 NF4 主干、VAE、PCA、CleanMemory；联合训练学生 Transmission LoRA 和 Q39/Q41 四个条件读取器。LoRA LR 5e-6，读取器 LR 1e-4，gamma LR 5e-3，20 更新 Warmup，然后 Cosine，裁剪 1.0。

$$
L_{base}=\tfrac12[L_{rec}(I)+L_{rec}(P90)]+0.1L_{polar}
$$

$$
L_{rec}(X)=L_1(F(X),GT)+0.2(1-SSIM(F(X),GT))+0.1L_{edge}(F(X),GT)
$$

三个辅助项为空间恢复/保持、Q16/Q20 纹理、Q37/Q39/Q41 内容与关系。预测图像梯度目标各 8%，前 36 更新渐入，EMA 0.9，组合上限 25%。每组四张分别计算输出梯度控制；使用更新开始时相同 EMA 状态，组结束取平均 EMA，避免把一个优化更新误算成四次 EMA 更新。有效 batch 与目标相同，单卡的执行顺序和 EMA 聚合不保证与旧四卡逐位相同。

第 2 epoch 起每 8 更新做条件 off/on 反事实校准；四张图分别生成校准信号，再平均加入 gate。每张以同一组平均主梯度为参照控制校准增量至 10%，不多次叠加放大上限。

显存策略：NF4 + BF16、梯度 checkpoint、可微激活 pinned CPU 卸载、教师/学生分阶段切换。每个微步结束把参数梯度累计到 CPU 并清空 GPU 梯度，组结束才将平均梯度送回 GPU 更新；8-bit AdamW 状态只在更新时到 GPU。可避免后三张图前向时一直额外占用完整 LoRA 梯度。

### M5 保持的配置

用本轮 C1-best 离线生成 RGB8 T0，只以普通 I 为输入；再固定上游训练 87,052 参数的四尺度像素残差恢复器。输入 I、T0、I−T0，输出头零初始化，总残差限制 ±0.20。

$$
L_{M5}=L_{Charbonnier}+0.2L_{SSIM}+0.1L_{edge}+0.1L_{band}+0.05L_{keep}+0.05L_{identity}
$$

identity 每 10 样本执行；FP32 AdamW，LR 2e-4，weight decay 1e-4，首个 epoch Warmup，Cosine 到 0.1 倍 LR。零残差 baseline 有资格成为 best，不能预先声称像素模块会改善。

## best/latest 与恢复保证

```text
runs/ws_c1_m5_dataset3_s1_e20/
  status.json / pipeline.pid / pipeline.exit_code
  logs/{prepare,c1,rgb_cache,m5}.log
  memory_prepare/{memory.safetensors,report.json}
  c1/
    best/{model.safetensors,manifest.json}
    latest/{model.safetensors,trainer_state.pt,manifest.json}
    run_config.json / metrics.jsonl / training_summary.json
    best_validation/ / latest_validation/
  m5/
    best/{model.safetensors,manifest.json}
    latest/{model.safetensors,trainer_state.pt,manifest.json}
    config.json / train.jsonl / metrics.jsonl / training_summary.json
```

长期仅两个模型版本。best 按保存后 RGB8 的验证宏平均 L1 选择，latest 保存训练权重及优化器、调度器、Python/NumPy/Torch CPU/CUDA RNG、epoch/group/update 游标。C1 还保存梯度控制 EMA 和可靠性 tau。第一更新保存，然后默认每 10 更新及每 epoch 保存；不会保存 epoch_N 权重。遇到 SIGINT/SIGTERM 在当前优化更新完成后保存并停止。硬终止最多丢失从上次保存后的更新。

checkpoint 为目录 bundle：先完整写临时目录，记录 model/state SHA-256，再切换；前一个完整 latest 在切换完成前保留。恢复检查模型、缓存、数据、记忆和训练 recipe 身份，拒绝串用旧数据/旧 C1。崩溃留下 `.previous_latest` 时可自动恢复上一个完整版本；临时副本不作为历史模型长期保留。

**从 latest 续训保留优化器；best 是推理/选择权重，没有优化器状态。** epoch 数是累计目标，不能将目标设到已完成进度以下。中途同预算重启恢复原调度状态；延长目标时保持已有 optimizer/scheduler step，之后 Cosine 按新总步数延展，可能有学习率调整，不等同于一开始就用更长预算。

空间约：C1 best 模型 3.2GiB、latest 模型+8-bit 状态约 4.9GiB；原子替换再临时需要约 4.9GiB，连同层缓存 7.746GiB 约 21GiB 峰值，不随 epoch 线性增长。入口要求至少 18GiB 额外可用；实际文件大小以运行记录为准。

## 启动与继续

```bash
cd /share/linmingheng-local/xuke/RMagNet
CUDA_VISIBLE_DEVICES=2 EPOCHS=20 OMP_NUM_THREADS=1 \
  bash scripts/run_windowseat_c1_m5_single.sh
```

同一命令重启同一 RUN_GROUP 会自动从已有 latest 接续；再次使用 tmux 前确认旧任务已经停止（有 flock 防止重复启动）。后台使用 `ws_c1_m5_d3_s1` 会话，关闭 SSH 不停止任务。

延长 C1 的累计目标，只运行 C1：

```bash
CUDA_VISIBLE_DEVICES=2 C1_EPOCHS=30 STAGES=c1 OMP_NUM_THREADS=1 \
  bash scripts/run_windowseat_c1_m5_single.sh
```

仅继续 M5，保持它原来的 C1-best 固定：

```bash
CUDA_VISIBLE_DEVICES=2 STAGES=m5 M5_EPOCHS=30 OMP_NUM_THREADS=1 \
  bash scripts/run_windowseat_c1_m5_single.sh
```

若 C1-best 改变，不可直接把旧 M5 latest 配到新的 RGB 缓存；身份检查会拒绝。应为新的 C1-best 另建 M5 实验，避免将上游变化误当作同一次像素续训。

验证只用 25 张 validation，训练不执行 24 张封存 test 或 real20。单卡比四卡慢；本轮只确认启动与恢复流程，不持续监控 20 epoch 完成。

## 实际启动前校验（2026-10-10）

- C1 在物理 GPU 2 完成一个真实优化更新，然后退出、载入 latest 并完成第二个更新。每个更新包含四个样本的 I/P90 前向。
- 恢复了 1,442 个优化器参数状态及 RNG；恢复后的游标为 epoch 0 / group 2 / update 2。主干、VAE、冻结记忆无梯度；LoRA/条件模块发生更新。
- 实测峰值 allocated 19.397 GiB；reserved 约 19.96 GiB。验证产物仅用于流程检查，正式训练仍从官方 WindowSeat 初始化。
- M5 使用合成小图验证优化器、调度器与 RNG 恢复后下一更新，与不中断的参考更新参数完全一致（最大差值 0）。这是续训状态测试，不是数据集效果实验。
- GPU 由物理编号 2 解析为 UUID `GPU-627649fe-d91c-b14e-1aa4-a615277dfca1` 再交给 CUDA，避免编号映射改变。
- 重启时完整缓存逐文件核对 SHA-256 后直接复用，清单保持不变；不会重新 finalize 破坏记忆/续训身份。
- 验证日志与状态报告保存在 `results_archive/WindowSeat_C1_M5/dataset3_single_verification_20261010/`；验证权重与临时预测图删除，正式训练不使用验证权重。
