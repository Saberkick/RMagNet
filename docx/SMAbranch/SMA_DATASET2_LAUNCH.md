# SMA data_set2 合并训练启动记录

- UTC 2026-10-08T14:52:42.247598+00:00：后台训练已稳定，tmux `sma_dataset2_e50`。
- GPU0/1/2/3，4个训练rank；204训练/26验证/26测试，旧划分不变。
- 204个缓存校验完成（旧144复用，新60提取），新memory仅用训练集拟合5个feature epoch。
- 正式50完整epoch，每epoch51更新，共2550更新，无早停；固定最终M4，reader从零初始化。
- 已检查35次更新：所有损失有限，reader梯度和参数更新正常，M4/VAE/memory冻结，没有错配GT。
- 已观察显存峰值17.891GiB。
- 仅保留best/latest完整SMA权重，不保留优化器和中间epoch权重。初始化memory权重在所有rank加载后已删除，保留报告和SHA。
- best按保存PNG的验证macro L1最小选择；latest在第1个epoch验证完成后首次保存，之后每epoch更新。

## 服务器路径

- 训练结果：`/share/linmingheng-local/xuke/RMagNet/runs/sma_dataset2_e50`
- 总日志：`/share/linmingheng-local/xuke/RMagNet/runs/sma_launch/dataset2_e50.console.log`
- 结束状态：同目录 `dataset2_e50.exit_code`，0表示全部流程成功。
- 数据：`/share/linmingheng-local/xuke/datasets/rmagnet_sma_dataset2`
- 缓存：`/share/linmingheng-local/xuke/RMagNet/data_cache/sma_dataset2_v1`
- 逐图索引：`/share/linmingheng-local/xuke/RMagNet/docx/SMAbranch/materials/dataset2_split_index.csv`
- 错配实验记录：`/share/linmingheng-local/xuke/RMagNet/results_archive/SMA_gtnoise5`

```bash
tmux attach -t sma_dataset2_e50
# Ctrl-b，再按d：脱离tmux，训练继续。
```

启动正常；本轮封存测试集未运行推理评估。详情见同目录materials中的启动审计JSON。
