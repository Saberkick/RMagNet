# M1b 可复现实验代码索引

执行代码保持在项目原位置，避免归档副本与实际运行入口分叉：

| 功能 | Shell 入口 | Python 实现 |
|---|---|---|
| 数据准备 | `scripts/prepare_m1b.sh` | `src/rmagnet/m1b_prepare.py` |
| 校准 | `scripts/calibrate_m1b.sh` | `src/rmagnet/m1b_calibrate.py` |
| 训练 A/B/C | `scripts/run_m1b.sh` | `src/rmagnet/m1b_train.py` |
| 诊断 | `scripts/diagnose_m1b.sh` | `src/rmagnet/m1b_diagnose.py` |
| 评估 | `scripts/eval_m1b.sh` | `src/rmagnet/m1b_eval.py` |
| 30% 配置 | `scripts/m1b_strength30.sh` | `src/rmagnet/m1b_strength30.py` |

10% 实验代码版本：`2bc4229`。30% 实验代码版本：`fe4e255`。当前归档提交只移动文档、复制元数据并清理 smoke 产物，不改变训练代码。各组实际参数、日志和完成记录见 `materials/run_records/{probe_e2,strength30_e2}/{base,dolp,shuffle}/`。
