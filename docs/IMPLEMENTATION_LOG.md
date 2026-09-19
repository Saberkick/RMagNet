# 实施记录

## 2026-09-18：项目骨架

- 已在 `/share/linmingheng-local/xuke/RMagNet` 建立项目目录；工作范围是个人目录。
- 服务器为 Ubuntu 22.04；uv 为 0.12.15；现有独立 WindowSeat 环境为 Python 3.12.11、torch 2.8.0+cu126、diffusers 0.35.1、peft 0.17.1。
- 现有 WindowSeat 仓库固定在提交 `e5ccbebd583ba53f385092ff5cb02898f1645709e`；骨架未克隆第二份仓库或模型。
- 检查时个人数据盘所在文件系统约有 1.2 TB 可用；8 块 RTX 3090 中 1–7 号各约有 24.3 GB 空闲，0 号约有 14.5 GB。共享机器状态会变化，正式训练前重查。
- 本轮只创建源代码、配置和文档。`toy` 共享网络用于接口和梯度烟雾检查，不代表 DiT 已接入；无真实数据训练、无新大文件下载。

## 验证

- `PYTHONPATH=src ... -m rmagnet.smoke`：CPU 输出形状检查通过；界面、T、R、融合模块的梯度均有限且非零；伪 adapter 路由按 T→R 分别激活并产生不同输出。
- `PYTHONPATH=src ... -m rmagnet.manifest data/manifest.example.jsonl`：`manifest_ok records=1`。这是占位清单的结构校验，没有检查图片文件存在。
- `python -m compileall -q src`：通过。
- 下一阶段的首个真实验证是官方前向等价和单支反向显存，而不是直接跑完整训练。

## 2026-09-18：框架总结与路线文档

- 新增 FRAMEWORK_SUMMARY.md：明确当前三个主模块、代码职责和玩具网络检查的边界。
- 新增 NEXT_STEPS.md：以真实 Qwen 后端前向等价为 M1，随后依次准备数据、训练界面/T/R、训练融合和独立评测。
- README 增加文档入口。本文档工作没有启动模型下载或训练。

## 2026-09-18：M1 真实后端

- 接入单个本地 NF4 Qwen 底座、冻结 VAE，分别注册 WindowSeat T LoRA 和随机初始化 R LoRA；不复制或下载大权重。
- 11、12、17 保存后的 8 位 PNG 与原 WindowSeat 结果完全相同；T→R→T 后 T 输出恢复一致。
- T/R 两支在 256/512 crop 上各做一次有限非零梯度检查；512 的 T 分支峰值 reserved 21.10 GiB，优化器与 768 crop 未验证。
- 详细资源版本、运行命令、输入哈希、结果和下一步见 [M1 报告](M1_REPORT.md)。


## 2026-09-19：M1a 三 LoRA 架构

- 最终主线调整为同一冻结 DiT/VAE 的 T、R、Fuse 三次串行 adapter 前向；Interface Head 首版可关闭。
- 新增中性 Interface、零初始化 T/R 条件器、T 锚定 Fusion Condition Mixer、FusionBackend 接口和 M1a 系统契约。
- `python -m rmagnet.m1a_smoke` 通过：融合条件初始化严格等于 T，三个阶段各自参数选择与有限梯度通过。
- 真实 Qwen `LoRA_Fuse`、frozen-VAE latent mixer、优化器与显存测量留给 M1b；详细方案见 [M1A_DESIGN.md](M1A_DESIGN.md)。
