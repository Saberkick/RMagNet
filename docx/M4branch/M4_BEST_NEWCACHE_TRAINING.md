# M4-best 新缓存继续训练方案

## 目标

从正式 M4-best Transmission LoRA 初始化，使用启用该 LoRA 重新提取的多层缓存继续训练。保留偏振一致性损失，不使用 DoLP。

## 固定配置

- 初始化：`runs/m4_e30_p4/best_transmission_lora.safetensors`
- 初始化 SHA-256：`5725d32b04e1271d51a33f7512174f1035ff0acf5e5427bdd3b179e98e1a13eb`
- 缓存：`data_cache/m4_best_multilayer_v1/`
- GPU：4 张；每卡 batch 1；有效 batch 4
- 学习率：`5e-6`，cosine，warmup 20 updates
- 最大 epoch：20；每 epoch 36 updates，最多 720 updates
- 验证：每个 epoch 一次
- 早停：验证 L1 连续 4 个 epoch 没有改善即停止
- `L_polar` 系数：`0.10`
- DoLP：不读取、不进入 loss
- checkpoint：只保留 `best` 与 `latest`

## 损失

基础生成损失继续对普通输入 I 和反射增强输入 P90 分别计算，并保留：

`L_base = mean(L_rec(I), L_rec(P90)) + 0.10 * L_polar`

其中 `L_rec = L1 + 0.2*(1-SSIM) + 0.1*L_edge`。多层 Qwen 辅助监督沿用 M4：空间、早层纹理与中层关系的目标梯度比例均为 0.08，总辅助梯度不超过基础梯度的 0.25。

## 运行

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
  bash scripts/train_m4_best_newcache.sh
```

输出目录默认为：`runs/m4_best_newcache_e20_p4/`。
