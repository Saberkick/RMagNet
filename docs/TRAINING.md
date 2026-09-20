# 训练接入顺序

> M1a 将最终融合改为第三个 `LoRA_Fuse`。下列旧顺序保留为历史基线；实际冻结策略和 24 GB 显存限制以 [M1a 最终方案](M1A_DESIGN.md) 为准。

1. **前向等价与资源测量。** 新 Qwen backend 加载本机已缓存的官方固定权重。在所有新模块关闭时，与既有 WindowSeat 推理的 latent/PNG 对齐。确认 VAE 标准化、token 打包、速度场符号、固定文本 embedding 和 tile 流程。然后只做一对样本的反向梯度与显存测量。
2. **数据清单。** 写入真实图、四偏振角度、曝光、GT、配准分数、场景 split。合成数据保留可解释的 R 和 mask。质量差区域降权，训练集以场景为单位划分。
3. **界面层。** 合成真 mask 训练 `M_interface/M_edit`，真实 DoLP 只给软提示；审查高 DoLP 负例。独立记录界面和编辑 mask 的效果。
4. **分离训练。** 冻结一份 4-bit 预训练底座，分别训练 `LoRA_T` 与 `LoRA_R`。T 用合成/真实可靠 GT，R 主要用合成反射真值。先单支小 crop、batch 1 和梯度累积；rank、分辨率由实测决定。
5. **融合与交替微调。** 先冻结 T/R，缓存候选训练小融合器；再交替更新 T+融合与 R+融合。若最终要求两路同时端到端反传，必须实测两条 autograd 图的峰值显存，顺序前向本身不节省这部分内存。
6. **消融。** 同训练集和预算比较 T-only、+界面、+R 简单融合、+R 学习融合、完整模型。统一保存 8-bit PNG，评估整图 PSNR/SSIM、反射区效果、未反射区误改、文字与物体幻觉、运行成本。

训练代码下一步必须提供可重启 checkpoint、数据 split 哈希、随机种子、环境版本和 adapter 权重，不保存重复的 12B 底座。正式训练前检查共享机器的可用 GPU 与个人目录容量。

## Stage 1 已实现入口

Reflection LoRA 的真实八卡训练、验证、最佳权重和精确断点恢复已落在 `src/rmagnet/stage1_train.py` 与 `scripts/run_stage1.sh`。当前 45°→90° 监督的含义、默认超参数和短程实测见 [Stage 1 训练报告](STAGE1_TRAINING_REPORT.md)。

