# RMagNet

当前目标：**反射界面识别 + 共享预训练 DiT 的 T/R 独立 LoRA + 最终学习融合**。部署输入为单张 RGB；四张偏振帧与 DoLP 只在训练阶段提供监督或构建先验。

## 当前状态

M1 已完成真实 Qwen/WindowSeat T 前向回归和 T/R 单支梯度测量。M1a 在此基础上固定最终结构：同一个冻结 DiT/VAE 串行激活 `LoRA_T`、`LoRA_R`、`LoRA_Fuse`，并增加以 T 为零初始化锚点的 Fusion Condition Mixer；Interface Head 为可选模块，首版默认关闭。Stage 1 的训练工程链路已经通过并完成 100 epochs；语义验收显示 R LoRA 仅小幅超过 identity、未超过 oracle affine，因此进入 Stage 1b 的变化区域加权修正。

## 结构

```text
configs/default.toml              路径与候选超参数
data/manifest.example.jsonl       数据清单格式示例
src/rmagnet/contracts.py           模块间张量契约
src/rmagnet/interface.py           可训练界面/编辑/保留置信度头
src/rmagnet/conditioning.py        可选界面条件与 T 锚定融合条件
src/rmagnet/m1a.py                三次 LoRA 前向的 M1a 系统契约
src/rmagnet/m1a_smoke.py          分阶段冻结和梯度检查
src/rmagnet/backend.py             共享底座的分支路由接口与玩具实现
src/rmagnet/qwen_backend.py        M1 真实共享 Qwen 后端和独立 T/R LoRA
src/rmagnet/m1_validate.py         M1 WindowSeat PNG 前向回归与路由检查
src/rmagnet/m1_backward.py         M1 单分支梯度与显存检查
src/rmagnet/stage1_train.py         Stage 1 多卡 Reflection LoRA 训练器
src/rmagnet/stage1_eval.py          Stage 1 identity/affine/最佳/最终验收
scripts/run_stage1.sh               可完成 Stage 1 的八卡训练脚本
scripts/smoke_stage1.sh             两卡短程训练检查
scripts/eval_stage1.sh              单卡 Stage 1 验收脚本
src/rmagnet/fusion.py              轻量融合网络
src/rmagnet/system.py              两次 T/R 前向及融合
src/rmagnet/losses.py              合成数据损失原型
src/rmagnet/manifest.py            清单与场景隔离校验
src/rmagnet/smoke.py               CPU 前后向检查
docs/ARCHITECTURE.md               设计和监督关系
docs/TRAINING.md                   分阶段接入计划
docs/IMPLEMENTATION_LOG.md         实施记录与验证结果
docs/M1_REPORT.md                 M1 实测、复现命令与边界
docs/M1A_DESIGN.md                三 LoRA 最终方案与 M1b 边界
docs/STAGE1_TRAINING_REPORT.md      Stage 1 流程、实测和执行方法
docs/STAGE1_ACCEPTANCE.md           Stage 1 量化、视觉验收与结论
```

## 运行骨架检查

沿用已用 uv 创建的隔离环境，暂不新装一套 PyTorch 或复制大权重：

```bash
cd /share/linmingheng-local/xuke/RMagNet
PYTHONPATH=src /share/linmingheng-local/xuke/envs/windowseat-py312/bin/python -m rmagnet.smoke
PYTHONPATH=src /share/linmingheng-local/xuke/envs/windowseat-py312/bin/python -m rmagnet.manifest data/manifest.example.jsonl
```

后续若需修改依赖，使用个人目录中的 uv 缓存创建独立 `.venv` 并记录 lock；先测剩余空间。基础权重从已有个人 HF 缓存读取，不复制到项目目录。项目、缓存、数据与运行结果只放 `/share/linmingheng-local/xuke`；不使用 Docker。

## 文档入口

- [M1a 最终方案](docs/M1A_DESIGN.md)：三套 LoRA、冻结 VAE、可选 Interface 和训练顺序。
- [Stage 1 训练报告](docs/STAGE1_TRAINING_REPORT.md)：数据、损失、多卡同步、checkpoint、命令与实测。
- [Stage 1 验收报告](docs/STAGE1_ACCEPTANCE.md)：identity/affine 基线、逐图结果、视觉判断和 Stage 1b 建议。
- [框架总结](docs/FRAMEWORK_SUMMARY.md)：当前设计、代码职责与已验证边界。
- [下一步路线](docs/NEXT_STEPS.md)：真实 DiT 接入、数据、训练和验收闸门。
- [架构契约](docs/ARCHITECTURE.md)、[训练接入顺序](docs/TRAINING.md)、[实施记录](docs/IMPLEMENTATION_LOG.md)。

## M1 真实后端（已验证）

`src/rmagnet/qwen_backend.py` 复用 WindowSeat 的 NF4 Qwen、VAE 和 T LoRA，并在同一个 Transformer 注册独立的未训练 R LoRA。`m1_validate.py` 已在 11、12、17 上逐像素复现基线 PNG；`m1_backward.py` 已完成两支 256/512 crop 的单次梯度与显存检查。使用方法、数据、实测值和限制详见 [M1 验证报告](docs/M1_REPORT.md)。界面头、融合器和真实 R 训练仍是骨架阶段。
