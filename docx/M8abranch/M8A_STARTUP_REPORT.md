# M8a 启动检查

记录时间：2026-10-12T05:49:48.826689+00:00。已启动后台训练，不是最终效果报告。

- 物理GPU 1、2、3，tmux会话 `m8a-late-strong-gates`。
- 冻结M4 final best，重新初始化第54、58层门控；计划2epoch、68次更新。
- 最大训练图 `130_5411_2729` 上，禁用门控及κ=1/4零初始化门控相对M4的latent最大误差均为0。
- 同口径25张M4验证基线：PSNR 24.0578 dB / SSIM 0.875925。
- 已检查优化更新 23 次；两层梯度非零、有限，第一步权重变化断言通过，冻结参数无梯度。
- 显存峰值（逻辑0/1/2，对应物理1/2/3）：{'0': 9.340239524841309, '1': 4.843430519104004, '2': 5.085613250732422} GiB。
- best/latest已保存；latest有优化器状态，κ随checkpoint保存。门控强度及质量改善需等待epoch验证后判断。
- 运行目录：`/share/linmingheng-local/xuke/RMagNet-M8a/runs/m8a_m4_late_strong_gates_e2`。
- 日志：`logs/train.log`；每步记录：`train.jsonl`；每epoch评价：`metrics.jsonl`。
- 训练完成标志：`training_summary.json`的status为complete且cursor.epoch为2、step为68；脚本正常退出码0。

详细设计见 `M8A_LATE_STRONG_GATES.md`。本次没有运行封存测试或real20。完成启动检查后退出SSH，不继续监控整个训练。
