# RMagNet M1 分支阶段总结

日期：2026-09-21。

## 目标与结构

M1 验证一个冻结 Qwen-Image DiT/VAE、三套独立 LoRA 的去反射路径：

```text
I ── LoRA_T ──► T ─┐
I ── LoRA_R ──► R ─┼─► LatentFusionMixer ─► LoRA_Fuse ─► T_final
I ─────────────────┘
```

Stage 1 训练 `LoRA_R`，Stage 2 训练由 WindowSeat 初始化的 `LoRA_T`，Stage 3 冻结两者并训练 latent mixer 与 `LoRA_Fuse`。首版没有启用 Interface Head。

## 数据与索引

数据根目录：`/share/linmingheng-local/xuke/datasets/rmagnet_stage1_512x384`，共 53 组、每张 512×384：

- `blended/{id}.png`：45° 原始反射图 I；
- `transmission_layer/{id}.png`：0° 图，作为 T/GT；
- `reflection_layer/{id}.png`：90° 反射加强图，作为 R 监督。

训练索引（50 张）：

```text
13,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,
34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
54,55,56,57,58,59,61,64,65,66
```

固定验证/测试索引：`11,12,17`。三个正式阶段使用相同划分，避免样本泄漏。Stage 3 的冻结候选缓存位于 `cache/stage3_candidates`，含 53 张 T 和 53 张 R。

## 阶段结果

| 阶段 | 关键权重 | 结论 |
|---|---|---|
| Stage 1 / R | `runs/stage1_reflection_r8/best_reflection_lora.safetensors`，step 500 | 比复制输入略好，但低于逐图仿射基线；主要学到色调变化，尚不能证明是可靠 R expert。 |
| Stage 2 / T | `runs/stage2_transmission_r128/best_transmission_lora.safetensors` | 11/12/17 保存后 PNG：23.140 dB、0.8216 SSIM；比官方 WindowSeat adapter 高 0.564 dB、0.0071 SSIM。 |
| Stage 3 / Fuse | `runs/stage3_fusion_r8/best_{fusion_lora,latent_mixer}.safetensors`，step 400 | 23.035 dB、0.8157 SSIM，低于 Stage 2 T 的 23.142 dB、0.8215；没有带来平均收益。 |

Stage 3 正式训练完成 50 epochs、650 steps，单卡峰值约 18.46 GiB allocated。best 出现在 step 400，final 略差，说明后半程没有继续改善验证集。

## 关键消融与判断

Stage 3 best 的 R 置零为 23.061 dB / 0.8155，R 跨样本置乱为 23.102 dB / 0.8159；两者均未稳定劣于正常 R。当前 Fuse 没有可靠使用 Reflection 分支，因此 M1 的工程链路成立，但“R 反参考有效”这一模型假设尚未被实验支持。

视觉上未见明显新增大型物体，但 Stage 3 没有改善文字与细线；12 号图出现最明显退化。下一轮应先加强 R 的区域监督或门控约束并增加验证数据，再决定是否继续三 LoRA 主线。

## 复现实验入口

```text
scripts/run_stage1.sh
scripts/eval_stage1.sh
scripts/run_stage2.sh
scripts/eval_stage2.sh
scripts/prepare_stage3_cache.sh
scripts/run_stage3.sh
scripts/eval_stage3.sh
```

正式 Stage 3 测试产物：`runs/stage3_fusion_r8/evaluation`。
