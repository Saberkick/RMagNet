# M4 运行手册

## 0. 当前状态

代码和脚本已实现，但尚未生成 M4 缓存，也没有启动 M4 smoke 或正式训练。当前 M3 100 Epoch 训练不会被这些新增文件打断。

## 1. 生成离线缓存

必须等待所选 GPU 空闲。四张卡并行：

```bash
cd /share/linmingheng-local/xuke/RMagNet

GPUS=0,1,2,3 \
OMP_NUM_THREADS=1 \
bash scripts/prepare_m4_cache.sh
```

如果 0–3 正在使用，可改为：

```bash
GPUS=4,5,6,7 bash scripts/prepare_m4_cache.sh
```

缓存支持断点续做。输出：

```text
data_cache/m4_multilayer_v1/
├── samples/*.safetensors
├── records/*.json
├── logs/shard_*.log
└── manifest.json
```

只有四个 shard 都成功后才生成 `manifest.json`。

M4 训练默认不做水平翻转，因为离线 Qwen 特征与原始 token 位置绑定。

## 2. 两步 Smoke

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OMP_NUM_THREADS=1 \
bash scripts/smoke_m4.sh
```

Smoke 输出：

```text
runs/m4_smoke_2steps/
```

这是唯一需要重点验证显存的阶段。训练在线教师运行到 block 41，比 M3 的 block 20 更深。

## 3. 正式训练

两 Epoch：

```bash
EPOCHS=2 \
RUN_NAME=m4_e2 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OMP_NUM_THREADS=1 \
bash scripts/train_m4.sh
```

五 Epoch：

```bash
EPOCHS=5 RUN_NAME=m4_e5 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_m4.sh
```

十 Epoch：

```bash
EPOCHS=10 RUN_NAME=m4_e10 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_m4.sh
```

位置参数同样可用：

```bash
RUN_NAME=m4_e10 CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_m4.sh 10
```

限制更新次数：

```bash
EPOCHS=10 MAX_STEPS=70 RUN_NAME=m4_70steps \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_m4.sh
```

训练在 `EPOCHS × 36` 和 `MAX_STEPS` 中较早达到的条件结束。


## 3.1 最多 30 Epoch、连续 4 Epoch 无改善早停

```bash
EPOCHS=30 \
EARLY_STOPPING_PATIENCE=4 \
RUN_NAME=m4_e30_p4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OMP_NUM_THREADS=1 \
bash scripts/train_m4.sh
```

早停指标为验证集保存后 8 位 PNG 的宏平均 L1。每个 Epoch（36 次更新）验证一次；只有严格降低 L1 才重置计数，连续 4 个 Epoch 未刷新 best 时停止。最多运行 30 Epoch（1080 次更新）。训练从干净的 Stage 2 best 初始化，只保留：

- `best_transmission_lora.safetensors`：历史最小验证 L1；
- `latest_transmission_lora.safetensors`：最近一个完整 Epoch；
- `best_metrics.json` 与 `latest_metrics.json`。

不保存优化器状态和逐 Epoch checkpoint，因此 latest 用于推理比较，不用于精确恢复优化器训练。

## 4. 后台运行

```bash
tmux new-session -d -s m4_e10 \
  "cd /share/linmingheng-local/xuke/RMagNet && \
   EPOCHS=10 RUN_NAME=m4_e10 CUDA_VISIBLE_DEVICES=0,1,2,3 \
   bash scripts/train_m4.sh > runs/m4_e10.console.log 2>&1"
```

查看：

```bash
tail -f runs/m4_e10.console.log
```

退出 tail 使用 `Ctrl+C`。查看 tmux 时，用 `Ctrl+B` 后按 `D` 安全脱离，不要在训练窗口按 `Ctrl+C`。

## 5. 输出

```text
runs/<RUN_NAME>/
├── run_config.json
├── metrics.jsonl
├── best_metrics.json
├── best_transmission_lora.safetensors
└── training_summary.json
```

只长期保留最佳 LoRA，不保存逐 Epoch 的恢复检查点。

## 6. 关键日志字段

- `actual_aux_base_ratio`：辅助梯度与基础梯度实际比例，应不超过 0.25。
- `spatial_scale`、`texture_scale`、`semantic_scale`：三组自适应缩放。
- `texture_q16_q20`：前段纹理损失。
- `semantic_q37_q39_q41`：中层语义总损失。
- `semantic_content`、`semantic_relation`：中层两个子项。
- `peak_allocated_gib`、`peak_reserved_gib`：显存峰值。
- `best_updated`：本次验证是否刷新最佳 LoRA。

## 7. 停止条件

出现以下任一情况应停止正式训练并检查：

- 任意 loss、梯度或 gate 出现 NaN/Inf；
- 辅助/基础梯度比例超过 0.25；
- 冻结主干或 VAE 获得梯度；
- 显存随 step 持续增加；
- 验证 L1 连续多个 Epoch 恶化；
- 文字和低变化区域明显被重绘。
