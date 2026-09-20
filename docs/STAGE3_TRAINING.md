# Stage 3：Fusion 训练流程

## 目标与冻结策略

Stage 3 固定 Stage 1/2 的输出能力，只训练 `LoRA_Fuse`（默认 rank 8）和 `LatentFusionMixer`（默认宽度 64）。共享 Qwen DiT、VAE、`LoRA_T`、`LoRA_R` 均冻结。第一版继续关闭 Interface Head，避免在没有界面真值的情况下把额外变量混入主实验。

## 先生成候选缓存

```bash
cd /share/linmingheng-local/xuke/RMagNet
CUDA_VISIBLE_DEVICES=0 bash scripts/prepare_stage3_cache.sh
```

默认读取 Stage 2 best T、Stage 1 best R 以及原始 I，输出到 `cache/stage3_candidates/{transmission,reflection}`。`manifest.json` 记录两个权重的路径和 SHA256。缓存采用 8 位 PNG，使训练输入可复现并与图片评价口径一致。

## 融合路径

```text
I,T,R -- frozen VAE encode --> z_I,z_T,z_R
                              │
                              ▼
                  LatentFusionMixer (trainable)
                  z_fuse = z_T + delta(I,T,R)
                              │
                              ▼
                 frozen DiT + LoRA_Fuse (trainable)
                              │
                              ▼
                    frozen VAE decode --> T_final
```

Mixer 末层零初始化，因此初始 latent 严格锚定 `z_T`。R 作为反参考进入 mixer，但不强制 `I=T+R`。损失为 L1、SSIM 和边缘 L1 的加权和，监督目标是 transmission GT。

## 正式训练

默认使用 4 张 GPU，每个进程 batch 1：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 bash scripts/run_stage3.sh
```

可调整 GPU 与 DataLoader CPU 数量：

```bash
CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 OMP_NUM_THREADS=4 \
  bash scripts/run_stage3.sh --num-workers 4
```

checkpoint 保存 Fuse LoRA、latent mixer、scheduler、epoch 和 batch 位置。PagedAdamW8bit 状态不保存，恢复时动量重新初始化。

## 最小流程测试

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/smoke_stage3.sh
```

只缓存 ID 13/11，运行一个 optimizer step、一次验证和一次 checkpoint 保存。确认流程后立即结束，不属于正式监督训练。

## 产物与后续验收

正式产物位于 `runs/stage3_fusion_r8`，包含 `fusion_lora.safetensors`、`latent_mixer.safetensors`、best 权重、指标、配置和日志。

训练结束后统一以保存后的 8 位 PNG 比较 WindowSeat、Stage 2 T 与 Stage 3 final。还要完成 `T-only`、R 置零、R 置乱消融；若 R 置乱没有明显改变结果，不能声称 R 分支参与了有效融合。

## 2026-09-20 单步实测

- GPU：2 × 24 GiB；训练 ID 13，验证 ID 11。
- 完成 1 个 optimizer step、验证、best 保存和 `checkpoint-0000001` 保存后正常退出。
- trainable 参数：53,335,184；有效梯度张量：1,448。
- step 1：loss 0.213918，L1 0.131580，SSIM loss 0.386149，edge 0.051074。
- 验证：L1 0.168599，PSNR 12.7815 dB，SSIM 0.461159。
- 单卡峰值：allocated 18.332 GiB，reserved 18.904 GiB。

这些数值只证明计算图、跨卡梯度同步、验证和保存链路可执行，不代表模型质量。
