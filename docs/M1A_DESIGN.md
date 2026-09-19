# RMagNet M1a 最终方案

日期：2026-09-19。M1a 在已验证的 M1（真实 Qwen/WindowSeat T 前向和 T/R 命名 adapter）上增加第三个融合角色与可选界面条件，但不把尚未训练的模块描述成有效模型。

## 结论

采用一个冻结的 Qwen-Image DiT、一套冻结 VAE 和三套互相独立的 LoRA：`LoRA_T`、`LoRA_R`、`LoRA_Fuse`。三者是**三次串行前向**，不是一次前向中叠加三个 adapter。

```text
                         optional Interface Head
                                  │
I ── frozen VAE/DiT + LoRA_T ──► T│
│                                 ├─► Fusion Condition Mixer
└── frozen VAE/DiT + LoRA_R ──► R│     anchored at T
                                  │            │
                                  └────────────┘
                                               ▼
                               frozen VAE/DiT + LoRA_Fuse
                                               │
                                               ▼
                                            T_final
```

第一版关闭 Interface Head。Qwen 本身已有很强先验，先证明 `T/R/Fuse` 三个角色带来可测收益，再决定界面分支是否值得额外训练。关闭时输出中性条件：interface/edit 为 0，keep 为 1。

## 冻结 VAE 与 Interface Head 的关系

使用冻结 VAE，但它不能替代 Interface Head。VAE 负责把 `I/T/R` 无损耗尽量小地映射到 Qwen 的固定 latent 形状，并负责最终解码；它没有“哪里是玻璃、哪里应编辑”的明确监督。Interface Head 是可选的空间先验模块，只有在 PBR mask、人工 mask 或可靠偏振弱标签足够时才启用。

真实融合不直接把 RGB 通道拼给 Qwen。Qwen 的 latent/patch 输入宽度固定。M1b 采用以下路径：

1. 冻结 VAE 分别编码 `I/T/R` 为 `z_I/z_T/z_R`。
2. 将可选界面图降采样、投影为 latent 特征。
3. 小型 Fusion Condition Mixer 读取这些张量，输出与 `z_T` 同形状的残差条件：`z_fuse = z_T + Δ(z_I,z_T,z_R,F_interface)`；末层零初始化，使初始条件严格锚定 T。
4. 激活 `LoRA_Fuse`，做第三次一步 flow，冻结 VAE 解码得到 `T_final`。

仓库当前的 `FusionConditionMixer` 先在 RGB 上实现同一契约，便于 CPU 检查；真实 Qwen latent mixer、第三 adapter 与显存验证属于 M1b。

## 分阶段训练

### Stage 1：Reflection

- 冻结 DiT、VAE、`LoRA_T`、`LoRA_Fuse` 和融合 mixer。
- 默认关闭 Interface Head，只训练 `LoRA_R`。
- R 的主监督必须来自 PBR/合成的真实反射通道；真实照片的 `I-GT` 只能作为经过线性化、配准和未饱和 mask 约束的弱标签。
- 先用 256/512 crop、batch 1、梯度检查点。R rank 先用 8 或 16。

### Stage 2：Transmission

- 冻结 DiT、VAE、`LoRA_R`、`LoRA_Fuse`。
- `LoRA_T` 从 WindowSeat rank 128 权重开始，监督目标为 clean transmission。
- 24 GB 3090 上，M1 的 512 单次反向已经预留 21.10 GiB，且没有优化器状态。因此先把 WindowSeat T 作为冻结基线；确需微调时从 256 crop 与 8-bit/paged optimizer 开始，或者将其蒸馏/分解到较低 rank。没有显存实测前不启动 rank 128 的完整 AdamW 训练。

### Stage 3：Fusion

- 冻结 DiT、VAE、`LoRA_T`、`LoRA_R` 和 Interface Head。
- 先离线缓存 T/R 候选，训练 Fusion Condition Mixer 与 `LoRA_Fuse`，避免保留三条反向计算图。
- 用 transmission GT 监督 `T_final`；同时加入未反射区保持、边缘/文字保真和低频颜色一致性损失。R 是反参考，不要求满足简单的 `I=T+R`，除非数据在线性光域且成像模型成立。
- 必须做 `T-only`、移除 R、置乱 R、移除 interface 四组消融；若置乱 R 不影响结果，说明 Fuse 忽略了 R。

### 可选 Stage 0：Interface

只有获得可审查标签后才启用。先独立训练 interface/edit/keep，随后在 Stage 1/2 冻结它。若还需要可学习的 T/R condition mixer，应作为单独消融，不混入“三个 LoRA”的首个结论。

## M1a 与 M1b 边界

M1a 已定义可选 interface、T/R 输入条件、T 锚定的融合条件、第三 FusionBackend 接口和严格分阶段冻结的 CPU 检查。M1 已验证的 WindowSeat T 输出、数据哈希和显存记录全部保留。

M1b 才接通真实 Qwen `LoRA_Fuse` 与 frozen-VAE latent mixer，并完成：T 回归不变；第三 adapter 的 T→R→Fuse→T 路由；256/512 单次反向；缓存 T/R 的 Stage 3 最小优化器步骤；磁盘与显存报告。通过这些检查后才进入数据训练。

## 验收指标

- 全图：保存后 8-bit PNG 的 PSNR、SSIM、LPIPS。
- 区域：反射区改善、非反射区误改、文字/细线保真。
- 真实性：物体新增/删除和文字涂改案例数；强遮挡区单独标不确定性。
- 模块有效性：R 与 interface 的置乱/删除消融。
- 工程：每阶段 trainable 参数清单、非零有限梯度、峰值显存、吞吐、checkpoint 可恢复。
