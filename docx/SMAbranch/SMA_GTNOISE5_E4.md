# SMA：约 5% GT 错配、重新训练 4 epoch

## 实验定义

用户指定：把约 5% 训练样本的 GT 换成其他样本的 GT；不从上一轮 best/latest 继续，以 SMA 相同配置重新训练 4 epoch。

这是**标签噪声压力测试**。错配 GT 不应被表述为更正确的监督，也不预设其能提升泛化。原 SMA 的第 4 epoch 是在验证集上选出的 best，适合作为相同 4-epoch 读模块预算的历史参考。旧实验包含一次 20-epoch cosine schedule，而新实验为 4-epoch schedule，两者的前 144 步学习率轨迹不同；因此单次结果不能把差异归因于标签噪声。严格消融还需同为 4-epoch schedule 的干净标签对照，本次不额外启动该实验。

## 标签映射与数据保护

- 原始纠正标签的数据不改动：`/share/linmingheng-local/xuke/datasets/rmagnet_m2_aspect`。
- 独立 overlay：`/share/linmingheng-local/xuke/datasets/rmagnet_sma_gtnoise5`。
- 144 张训练图中固定选 **7 张**：实际比例 **7/144 = 4.8611%**，不是精确 5%。
- 标签噪声种子 `20261008`；模型、采样和训练种子仍为原配置的 `2026`。
- 接收样本在有可选 donor 的训练样本中均匀抽样；donor 必须来自**另一拍摄组**、训练集、完全相同的 width/height。避免为错配监督引入额外 resize/crop。
- donor 可以重复；接收样本 ID 不重复。只替换接收样本 GT，I/P90/DoLP、拍摄组、分桶、全部划分不变。
- 7 个映射固定用于全部 4 epoch，不逐 epoch 重新随机。
- 验证集 18 张、封存测试集 17 张保持正确配对，维持 `108_421_1388` 排除规则。
- `noise_mapping.json` 与 overlay 的 `manifest.json` 记录 mapping、来源 SHA-256、可抽样数量和实际比例。训练前审核所有 GT 文件与声明 donor 的哈希。

对映射为 `i -> j` 的训练样本：

$$
(I_i,P90_i,DoLP_i,GT_i) \longrightarrow (I_i,P90_i,DoLP_i,GT_j),\quad j\in\mathcal D_{train}.
$$

GT overlay 使用只读用途的符号链接；代码不写入链接目标。未变化的 137 份特征缓存通过硬链接复用，文件仅供读取；7 份有变化的缓存重新提取到独立文件，不覆盖旧缓存。

## 全部 GT 监督一致

不只交换像素损失的 GT。对于这 7 张图：

1. 重建目标统一换为 donor GT，普通输入与 P90 两路都使用它。
2. Q16/Q20/Q37/Q39/Q41 GT 特征重新提取。
3. Q52/Q54/Q56 的 I/GT 差异、空间 gate、agreement 全部按新配对重算。
4. I/P90 特征仍来自原接收样本。
5. 使用新缓存重新拟合 PCA、从相同随机种子初始化内容记忆并预训练 5 个 feature epoch；避免冻结记忆已经提前见过那 7 张的正确 GT。
6. 正式读模块从零输出初始化，只训练 4 个 image epoch，不加载上一轮 SMA best/latest。

固定底座仍为已训练过的最终 M4-best，这与原 SMA 配置一致。因此本实验测的是**M4 固定先验下的新 SMA 训练对噪声的敏感性**，不是整个系统从未见过干净标签的学习实验。

## 配置

| 配置项 | 数值 |
|---|---|
| 固定 M4 | `runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors` |
| SHA-256 | `897282b1bb9cfe61f96530df72edcf8a44a066bb819a3663e9100862aefdb2a3` |
| 架构 | `sma-rms-v1`，Q37 内容记忆，Q39/Q41 RMS-scaled reads |
| 记忆预训练 | 新初始化、5 epoch、144 updates/epoch，与错配目标一致 |
| 正式训练 | 4 epoch、36 updates/epoch、共 144 updates |
| GPU | 0–3，四张 24GB RTX3090 |
| batch / 累积 | 每卡 1 / 1，有效 batch 4 |
| 读模块 LR / warmup | 1e-4 / 20 updates |
| 调度 / weight decay / clip | cosine / 0.01 / 1.0 |
| 精度 | 冻结底座 NF4 + BF16，新模块 FP32 |
| 辅助梯度 | spatial/texture/semantic 各目标 8%，36 steps 渐入，合并上限 25% |
| 冻结 | M4 全部参数、VAE、正式训练期的 PCA 与内容记忆 |
| 验证与选择 | 每 epoch，保存后 8-bit PNG；最小宏平均验证 L1 选择 best |
| 早停 | 关闭，按用户要求完成 4 epoch |
| 权重保留 | best/latest SMA，不保存优化器或中间 epoch 权重 |

Loss 与上一轮一致：普通输入/P90 重建均值 + 0.10 偏振一致性，spatial、texture、semantic 仍由原梯度控制器加入。**不取消 L_polar，不改变语义层选择，不增加新 loss。** 见 `SMA_IMPLEMENTATION_AND_TRAINING.md`。

## 脚本与产物

```bash
./bin/xuke
cd /share/linmingheng-local/xuke/RMagNet

# 一次后台完成：映射 -> 7 份缓存 -> 新内容记忆 -> 4 epoch 正式训练
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
  bash scripts/background_sma_gtnoise.sh

# 分阶段（仅用于手动执行，已有后台任务时勿重复启动）
bash scripts/run_sma_gtnoise.sh prepare
bash scripts/run_sma_gtnoise.sh train

# 只读查看
tail -n 3 runs/sma_gtnoise5_e4/metrics.jsonl
tail -n 20 runs/sma_launch/sma_gtnoise5_e4.console.log
tmux ls
```

- 分支：`experiment/sma-gtnoise5-e4`。
- 新 cache：`data_cache/sma_gtnoise5_v1/`，仅 7 份数组占额外空间，其余共享旧缓存存储。
- 内容记忆初始化：`runs/sma_gtnoise5_memory_pretrain/`。
- 正式 run：`runs/sma_gtnoise5_e4/`。
- tmux：`sma_gtnoise5_e4`。
- 退出码：`runs/sma_launch/sma_gtnoise5_e4.exitcode`，仅后台任务结束后出现。

正式训练稳定后退出 SSH，不等待 4 epoch 完成；不启动封存测试。缓存及原 SMA 的 best/latest 权重、验证/测试数据和指标保留。Smoke、临时传输包及临时日志已清理。内容记忆初始化权重在所有训练进程完成加载后可删除，只保留报告与 best/latest；若之后要重跑 fresh 实验，需要重新执行记忆预训练，不支持优化器恢复。

## 检查要点

固定映射必须为 7 个不同接收样本；无训练以外 donor；尺寸一致；验证/测试 GT 不变；全部 GT 特征和 gate 来源与 overlay GT 哈希一致；137 份复用数组与旧文件共享 inode；新记忆读模块零输出初始化；四卡实际参与，损失有限、冻结参数无梯度、读取梯度非零。启动实测另记于 `SMA_GTNOISE5_LAUNCH_RECORD.md`。
