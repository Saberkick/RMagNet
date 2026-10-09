# C1中断后评估说明

2026-10-09：原计划10epoch/510更新，已完成9个完整epoch并记录466次更新。第10epoch在rank2的P90 VAE解码处OOM，训练流程退出码1。错误日志显示另有1.02GiB显存占用，未触碰该进程。

此次不补训练，仅评估已保存的best（epoch2/step102）和latest（epoch9/step459）。step460–466没有持久化权重，不能把latest标成step466或epoch10。

脚本 `scripts/eval_sma_c_saved.sh` 使用GPU0、1、3两批完成：test26 best/latest、real20 best/latest，以及同一best模型验证集的语义条件off/on。正常门控验证结果复用best保存时的验证记录；同一checkpoint重载复现已在smoke验证。

GPU2仍有占用，避开；没有训练或M5实验。报告入口允许显式评估中断权重，完整训练的检查仍保留，不伪造training_summary.json。

最终结果在 `runs/sma_c1_e10/REPORT.md`、`comparison.json`、六个评估子目录及 `docx/SMA_Cbranch/C1_RESULTS.md`。评估是否完整看 `runs/sma_launch/c1_saved_eval.exit_code`，原训练退出码继续保留为1。
