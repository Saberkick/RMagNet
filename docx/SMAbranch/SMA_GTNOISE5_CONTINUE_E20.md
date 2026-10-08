# GTnoise5：从 4 epoch 的 latest 继续到累计 20 epoch

## 续训边界

父实验 `runs/sma_gtnoise5_e4/` 正常完成 4 epoch、144 次更新。使用其中 `latest_sma.safetensors` 继续，不使用 best，也不重新训练 PCA 或内容记忆。

旧实验只保留模型权重，未保留优化器、调度器或梯度控制 EMA 状态，因此这是**从权重继续训练**，不是恢复原训练进程。此前删除的初始化 memory 文件不影响续训：best/latest 内已包含 PCA、内容记忆和读取模块的完整状态。

| 项目 | 配置 |
|---|---|
| 开始位置 | 完成第 4 epoch / 全局 step 144 后的 latest |
| 本次训练 | 第 5–20 epoch，新增 16 epoch / 576 次更新 |
| 累计结束位置 | 20 epoch / 全局 step 720 |
| GPU / batch | GPU 0–3；每卡 1、累积 1，有效 batch 4 |
| 错配 GT | 保持原固定 7/144，不重抽样、不改验证/测试标签 |
| 缓存与教师 | 原噪声缓存及固定 M4-best，校验 manifest 与权重 SHA-256 |
| 训练参数 | 只训练 SMA Q39/Q41 readers；M4、VAE、PCA、内容记忆冻结 |
| 优化器 | 新建 AdamW，LR peak 1e-4、weight decay 0.01、clip 1.0 |
| 调度器 | 新建 576-step cosine，前 20 次新增更新 warmup |
| 辅助梯度 | 保留原三个 8% 目标与合并 25% 上限；EMA 重建，全局步数已超过 36-step 渐入期 |
| 验证/权重 | 每 epoch 保存后 PNG 评价；只保存 best/latest，按验证宏平均 L1 选 best |
| 早停 | 关闭，按用户要求继续到累计 20 epoch |

采样器从 epoch index 4 开始，不从 epoch index 0 重放。日志的 step 与 epoch 为累计编号；训练摘要同时记录 `optimizer_updates_this_run=576` 与累计 `optimizer_updates=720`。

## 继承 best 与保护旧实验

新 run 单独放在 `runs/sma_gtnoise5_e20_continue/`。父实验权重、指标、测试图片保持原状。新 run 先继承父实验的 best/metrics/validation，避免把续训后的劣化权重自动当作 best；每轮按同一验证标准更新。

父实验的 best 当前为 step 0 初始化。因此，新 run 的 best 也可能长期保留 step 0；应读取 `best_metrics.json` 的真实 step，不把它称作训练 20 epoch 的模型。新 run 的 latest 才代表累计训练位置。

启动审核：父 run 必须完整结束于全 epoch；latest 必须为最终 step；SMA 架构、M4 SHA、数据与 cache manifest SHA、四卡 batch 计数必须一致。首次 GPU 验证应复现父 latest 的验证 PSNR/SSIM，加载后读模块必须非零，第一步参数变化与冻结检查沿用原 SMA 验收。

本次延长训练由用户在看过封存测试后发起，因此该封存集已参与研究决策；后续测试应视为重复使用的研究测试，独立泛化结论仍需要新的未见数据。best 的选择继续只使用验证集。本次启动阶段不自动再跑测试集。

## 运行

```bash
./bin/xuke
cd /share/linmingheng-local/xuke/RMagNet

# 后台续训
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
  bash scripts/background_continue_sma_gtnoise.sh

# 前台入口，不要和后台重复执行
bash scripts/continue_sma_gtnoise.sh

# 只读查看
tmux ls
tail -n 3 runs/sma_gtnoise5_e20_continue/metrics.jsonl
tail -n 20 runs/sma_launch/sma_gtnoise5_e20_continue.console.log
```

tmux session：`sma_gtnoise5_e20_continue`。退出码在任务结束后写入 `runs/sma_launch/sma_gtnoise5_e20_continue.exitcode`。

正式续训稳定若干步后退出 SSH，不等待剩余 16 epoch 完成。启动记录见 `SMA_GTNOISE5_CONTINUE_LAUNCH.md`。
