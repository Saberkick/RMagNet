# C1 后台启动记录

状态：运行中，尚未完成10epoch，也尚无正式测试结论。

已观察连续 10 次更新，学生LoRA及条件模块均有非零梯度、四处条件hooks工作，主干/VAE/记忆冻结检查通过。主rank峰值 21.379 GiB。

- tmux：`sma_c1_e10`
- run：`/share/linmingheng-local/xuke/RMagNet/runs/sma_c1_e10`
- console：`runs/sma_launch/c1_e10.console.log`
- 整体退出码：`runs/sma_launch/c1_e10.exit_code`，包含训练、四项测试及两项验证集条件干预。
- 正式可靠性校准从第2epoch起每8更新；短跑已覆盖校准和checkpoint精确复现，短跑权重/图片已删除。
- 训练代码提交：`0936cc72eb20c34afe47316e2477989ce424cb27`。

只保留best/latest。结束后自动生成REPORT.md和C1_RESULTS.md。未启动C0、C2或M5训练。
