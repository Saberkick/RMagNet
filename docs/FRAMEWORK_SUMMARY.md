# RMagNet 框架总结

> 更新：2026-09-18。本文记录目前服务器目录中的真实实现状态，避免把骨架检查误认为模型已训练。

## 目标与核心判断

RMagNet 的当前目标是**单张 RGB 去反射**。采集阶段拥有四张偏振照片和一张反射较少的候选 GT；训练时利用这些额外信息，部署时只输入单张 RGB。主创新由三部分组成：

1. **反射界面识别**：区分“可能发生反射的界面”与“这张图确实需要改动的区域”。输出 `M_interface`、`M_edit` 和保留原图的置信度 `C_keep`。
2. **T/R 分离训练**：一份冻结的预训练 DiT 底座，分别激活 `LoRA_T`、`LoRA_R` 做两次前向。T 分支预测透射候选，R 分支预测反射/干扰候选。两条分支使用不同的监督目标，不是把两个 LoRA 权重相加。
3. **最终融合**：轻量网络读取原图、T/R 候选与界面图，产生去反射输出及不确定性。融合器必须通过消融证明自己实际利用了 R；高置信无反射区域尽量保留原像素。

```text
RGB I ──► 界面头 ───────────────┐
   │                           │
   ├──► 共享 DiT + LoRA_T ─► T候选 ─┐
   └──► 共享 DiT + LoRA_R ─► R候选 ─┼─► 融合器 ─► T_final, U
       原图像素 ────────────────────┘
```

选择一份共享底座，是为了避免复制两套约 12B 级模型权重。首版融合不使用第三个 DiT LoRA：它会要求另一次完整前向和新的多输入编码，且增加生成改写的风险。未来如果两次 DiT 前向过慢，再考虑将已验证的双分支系统蒸馏成单次前向学生。

## 当前代码与真实状态

| 文件 | 已经提供的能力 | 当前边界 |
| --- | --- | --- |
| `src/rmagnet/interface.py` | 可训练的三图界面头和张量契约 | 目前是小 CNN，未接入预训练视觉语义特征，也未受真实 mask 训练 |
| `src/rmagnet/backend.py` | 一个底座切换 T/R adapter 的接口；独立的玩具共享网络 | 尚未接入 Qwen/WindowSeat 的 VAE、DiT、LoRA 与一步 flow 路径 |
| `src/rmagnet/fusion.py` | 融合输入、残差输出、不确定性图 | 只验证了形状/梯度；尚未学习真实融合策略 |
| `src/rmagnet/system.py` | 界面 → T 前向 → R 前向 → 融合的调用顺序 | 真实双支训练的 adapter 梯度行为仍需单独验证 |
| `src/rmagnet/losses.py` | 合成监督的损失原型 | 权重只是烟雾检查值，不是经过实验选择的训练配方 |
| `src/rmagnet/manifest.py` | JSONL 结构、重复项和场景跨 split 检查 | 尚未读取真实图片或做配准、文件/标签质量检查 |
| `src/rmagnet/smoke.py` | CPU 输出尺寸、T/R 路由及四部分梯度检查 | 玩具网络结果没有去反射意义，也不预测真实显存 |

项目根目录还有 `configs/default.toml`、`pyproject.toml`、数据清单示例、[架构契约](ARCHITECTURE.md)、[训练接入顺序](TRAINING.md)和[实施记录](IMPLEMENTATION_LOG.md)。现有 WindowSeat 仓库和 uv 隔离环境在同一用户空间；RMagNet 没有复制其约 42.8 GB 模型权重或建立第二套大环境。

## 监督关系与最重要的限制

- `T*` 是主恢复目标，但“偏振后反射最少的一帧”可能仍有残影、色偏或错位，需要登记可靠区域。
- DoLP 是反射线索，不是反射像素真值；高 DoLP 的真实物体和无明显反射的玻璃都应进入训练对照。
- R 分支优先使用**合成数据中真实保存的反射贡献**监督。真实照片的 `I−T*` 因非线性成像、曝光和错位只能作为低权重弱线索。若只有最终 `T_final` 损失，R 分支可能塌缩为零或被融合器忽略。
- 最终图像的 PSNR/SSIM 与“字形、灯具、物体是否被虚构”要分开评价。11、12、17 是已知失效的固定回归样例，不足以作泛化结论。

## 已验证与未验证

已通过：CPU 前后向烟雾检查，`M_interface/M_edit/C_keep` 与 T/R/融合的张量形状，四个模块均有有限且非零梯度，伪 adapter 按 T→R 分别切换，示例 manifest 校验和 Python 编译。

尚未验证：真实 Qwen DiT 前向与 WindowSeat 是否等价、真实 PEFT adapter 切换后两路梯度、训练显存、真实数据清洗、界面标签质量、T/R 独立性及最终指标。**目前没有进行真实图片推理或模型训练。**

运行检查（只使用已存在的 uv 环境）：

```bash
cd /share/linmingheng-local/xuke/RMagNet
PYTHONPATH=src /share/linmingheng-local/xuke/envs/windowseat-py312/bin/python -m rmagnet.smoke
PYTHONPATH=src /share/linmingheng-local/xuke/envs/windowseat-py312/bin/python -m rmagnet.manifest data/manifest.example.jsonl
```

下一步以[实施路线](NEXT_STEPS.md)为准。项目、未来数据和训练缓存均放在 `/share/linmingheng-local/xuke`；不使用 Docker。

