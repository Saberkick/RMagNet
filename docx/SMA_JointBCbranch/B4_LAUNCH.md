# B4 正式启动记录

## 当前状态

2026-10-09（Asia/Shanghai）启动。已通过一步联合训练、26张验证、权重重载逐PNG哈希一致性检查；试跑权重和图片已清理，记录封存于 `results_archive/SMA_joint_b_smoke/`。

正式训练从M4-best和fresh SMA readers重新开始，GPU0–3，4 epoch /204更新。稳定检查记录见 `materials/B4_LAUNCH.json`，训练尚未完成时不能将本文件当作结果报告。

## 配置

- 全部学生Transmission LoRA可训练，852,180,992参数，LR=5e-6。
- SMA读取模块可训练，4,610,050参数，LR=1e-4。
- 固定Qwen主干、VAE、语义记忆与监督教师；教师LoRA驻CPU，与学生状态分离。
- 数据204/26/26，变量长宽比和现有缓存不变；保留L_polar及既有辅助损失。
- 只保留best/latest联合权重，不保存优化器。
- C仅设计，未实现或训练。

## 自动流程

```bash
bash scripts/run_sma_joint_b.sh 4
```

已有正式目录时拒绝覆盖，不应重复执行上述命令启动第二份当前实验。

后台tmux会依次执行：

1. 四卡B4训练，每epoch验证并保存best/latest。
2. best/latest分别评估26张封存测试，旧17/新9分开汇总。
3. best/latest分别评估real20，统一posterior mode和官方短边分块。
4. 与已有同口径M4/SMA50-best结果对比，保存逐图CSV、报告与预先按seed2026抽样的三张封存测试对比图，另保存real20的22/47对比图。

## 查看位置

- tmux会话：`sma_joint_b_e4`。
- 日志：`runs/sma_launch/joint_b_e4.console.log`。
- 最终退出码：`runs/sma_launch/joint_b_e4.exit_code`，全流程成功为0；尚不存在表示流程未结束。
- 正式目录：`runs/sma_joint_b_e4/`。
- 训练结束标志：`training_summary.json`。
- 评价完成标志：`REPORT.md`、`comparison.json`。
- 项目文档：`docx/SMA_JointBCbranch/B4_RESULTS.md`，由全流程最后一步生成。

严格S4/A4尚未运行。历史SMA50-best只作参考，不是相同4epoch cosine轨迹对照，因此报告不能直接宣称解冻LoRA具有因果收益。
