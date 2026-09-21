# M1：真实共享 Qwen 后端验证

日期：2026-09-18。所有路径相对 `/share/linmingheng-local/xuke/RMagNet`，模型、uv 环境、输入和输出均在 `/share/linmingheng-local/xuke`。本轮复用已有快照，没有下载大权重或使用 Docker。

## 固定资源与实现

- WindowSeat 官方仓库提交 `e5ccbebd583ba53f385092ff5cb02898f1645709e`。Qwen-Image-Edit-2509 快照 `d3968ef930e841f4c73640fb8afa3b306a78167e`；WindowSeat LoRA 快照 `c1f59ca02bff68535c976e5e17147b3d9323309e`。加载前按 `../configs/windowseat_download_manifest.json` 检查文件大小，强制 Hugging Face 离线模式。
- 既有 uv 环境 `../envs/windowseat-py312`：Python 3.12.11、torch 2.8.0+cu126、diffusers 0.35.1、peft 0.17.1、bitsandbytes 0.47.0。使用物理 GPU 1（一块 RTX 3090，23.69 GiB），运行前该卡约 24.3 GB 空闲。共享 GPU 状态随时会变。
- `src/rmagnet/qwen_backend.py` 加载**一个** NF4 Qwen Transformer 与冻结 VAE，沿用官方 `encode → flow_step → decode`、固定文本 embedding、timestep 与 latent 规则。T 为预训练 WindowSeat rank 128 LoRA（`default`），R 为**随机初始化、未训练**的独立 rank 8 LoRA（`reflection`）。两者串行 `set_adapter`，单次仅激活/训练一支；R 输出不具备反射层语义。
- 11、12、17 的输入、GT 和既有 WindowSeat PNG 的 SHA256 在 `runs/m1_inputs.sha256`。输入仍是 `../datasets/own3/processed_1024/blended` 中的 1024×768 图；GT 在相邻的 `transmission_layer`。没有重新缩图。

## 前向等价与路由

运行：

```bash
cd /share/linmingheng-local/xuke/RMagNet
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 PYTHONPATH=src ../envs/windowseat-py312/bin/python -m rmagnet.m1_validate --output runs/m1_parity
```

同一图块、同一 VAE 随机状态，T 路与官方调用的最大绝对浮点差为 **0**。T→R→T 后，恢复的 T 输出最大差为 **0**；T 与未训练 R 的最大绝对差为 **1.96875**，只说明命名路由生效。两支可训练参数集合分别为 T 852,180,992 个、R 53,261,312 个，均只含相应 LoRA。

| 图片 | 图块数 | 输出尺寸 | 与先前 WindowSeat 保存后 PNG 不同的通道值数 | 8 位最大绝对差 |
| --- | ---: | --- | ---: | ---: |
| 11 | 2 | 1024×768 | 0 | 0 |
| 12 | 2 | 1024×768 | 0 | 0 |
| 17 | 2 | 1024×768 | 0 | 0 |

逐图结果和新生成的 PNG 在 `runs/m1_parity/`；对比基线在 `../results/windowseat_own3_1024/`。验证沿用官方图块拼接、Lanczos 和 8 位 PNG 保存流程，随机种子 2026。这里的 T 是既有 WindowSeat 模型，因此没有重新宣称质量提升。

## 单支梯度与显存

命令示例（`--branch` 为 `transmission` 或 `reflection`，`--crop` 为 256 或 512）：

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 PYTHONPATH=src ../envs/windowseat-py312/bin/python -m rmagnet.m1_backward --branch reflection --crop 512 --checkpointing --output runs/m1_backward_R_512.json
```

中心裁块来自图片 11；T 的 L1 目标为其配对 GT，R 的零张量目标**仅用于检查反向传播**，不是训练标签。VAE 冻结，Transformer 底座冻结，只选中当前分支 LoRA；启用梯度检查点。各次为新进程、batch 1，且只做一次 forward/backward；未创建优化器或执行更新。

| 分支 | crop | LoRA 参数 | 有梯度张量/可训练张量 | 耗时 s | 峰值 allocated GiB | 峰值 reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R rank 8 | 256 | 53,261,312 | 1442/1448 | 4.78 | 15.92 | 16.25 |
| R rank 8 | 512 | 53,261,312 | 1442/1448 | 3.25 | 19.26 | 19.87 |
| T rank 128 | 256 | 852,180,992 | 1442/1448 | 2.56 | 18.19 | 19.46 |
| T rank 128 | 512 | 852,180,992 | 1442/1448 | 3.25 | 19.27 | 21.10 |

上述四次所有已生成梯度均有限且梯度绝对值总和大于零。少数 LoRA 张量本轮梯度为 `None`，与当前计算路径未使用该层有关；正式训练前仍需检查其覆盖范围。计时不包含模型从磁盘加载，不能视为稳定训练吞吐。T 512 结束时约剩 2.27 GiB，因此 **768 crop、优化器状态、梯度累积、保存/恢复训练均未验证**，在共享 24 GB 卡上可能需要缩小 T rank/目标模块、优化器分片或更多显存。运行记录分别为 `runs/m1_backward_{T,R}_{256,512}.json`。

## M1 结论与下一阶段

M1 的真实单底座、双命名 adapter、T 前向精确回归以及两支单次梯度检查完成。这里没有训练出 R，也没有融合、界面识别或质量对比结论。下一步按 `docs/NEXT_STEPS.md` 的 M2 先做真实数据 manifest、四偏振帧配准/曝光及 GT 质量审计和场景隔离；之后再选合理的 T LoRA 规模与 R 真值，实施可重启的单支优化器训练并检查 loss 下降。
