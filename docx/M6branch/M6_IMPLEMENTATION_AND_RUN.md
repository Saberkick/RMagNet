# M6-B 实施与五个 epoch 运行说明

## 本轮配置

- 起点：M4 initial best，step 612，SHA `5725d32b04e1271d51a33f7512174f1035ff0acf5e5427bdd3b179e98e1a13eb`。
- 数据：当前 `rmagnet_sma_dataset3`，204 train / 25 validation / 24 test；不改标签或划分，不读取 DoLP。
- 新项：Q37/39/41 中 P90-I45 方向，由 I45-GT 检验后，对预测超过 GT 的正投影残留做平方惩罚。
- 保留 M4 的 Lrec、空间、纹理、正向语义及 Lpolar。语义预算正向6%、负方向2%，合并8%，辅助总上限25%，36更新渐入。
- 5个 epoch，无早停。GPU 0/1/3，每卡batch1，有效batch3，每个epoch68次更新，总计340次。
- BF16、NF4冻结DiT、冻结VAE、只更新rank128 LoRA_T、PagedAdamW8bit，学习率5e-6，warmup20，cosine，clip1。
- 只运行主实验 M6-B；本轮未自动运行A/W/S消融，也不自动运行测试集或real20。

当前用户另一个任务 `ws_c1_m5_d3_s1` 使用GPU2。M6使用独立worktree、LoRA变量、梯度、优化器及输出目录。基础模型文件只读取，两个进程的模型对象独立，训练不会相互修改磁盘上的基础权重。共享磁盘文件不意味着共享可变GPU参数。

三个M6进程使用个人目录的文件锁依次加载模型，错开CPU加载峰值；加载后并行计算。权重保持原路径，不复制大型基础模型，不修改共享软件、驱动或其他用户的任务。

## 缓存

关闭全部LoRA的教师与M4 initial保持同一表示空间。**不能复用正在训练的C1缓存**，它启用了官方WindowSeat LoRA。

可复用旧M4缓存中角色、源图哈希及教师设置匹配的108张参考；实际复用数量以最终manifest为准。缺失参考与新增三视图方向重新计算。新缓存保存七份M4前/中层参考、三层U、三层置信度及晚层门控，CPU侧BF16特征、FP16权重，差分和损失使用FP32。

缓存目录：`data_cache/m6_polar_negative_v1/`。三进程按ID分片，不重复采样；可按哈希和公式身份接续缓存。对P90和GT不用DoLP。

原有配准异常已从manifest排除。区域质量控制包括差分范数、方向一致性、至少两层通过，以及饱和像素占比；生成训练图预览供审阅。未增加自动光流、未声称已证明每个局部都精确配准。

## 启动

在服务器执行 `./bin/xuke` 后：

```bash
cd /share/linmingheng-local/xuke/RMagNet-M6
CUDA_VISIBLE_DEVICES=0,1,3 EPOCHS=5 OMP_NUM_THREADS=1 \
  bash scripts/run_m6_pipeline.sh
```

环境由原uv 0.12.15与已有Python3.12管理。三个GPU均需至少22000MiB空闲，磁盘保留空间检查未通过就退出，不抢占现有任务。

后台使用专用tmux `m6_b_e5`，日志：`runs/m6_pipeline_e5/console.log`，阶段：`phase.txt`，结束码：`pipeline.exit_code`。

训练目录：`runs/m6_b_e5/`；只保留 `best_transmission_lora.safetensors` 和 `latest_transmission_lora.safetensors`，不保存逐epoch权重或优化器副本。best按25张验证图保存PNG的宏平均L1选择，step0允许best仍为原始M4起点。

验证PNG只保留best/latest两套。每epoch的CSV/JSON和完整loss/梯度日志保留，以便画曲线。只保存权重意味着可从latest重新开始下一轮微调，但不是优化器和RNG的精确断点恢复。

## 梯度与验证检查

每次更新记录基础及辅助loss、真实梯度比例、负方向与基础/正向梯度夹角、有效残留支持、LoRA梯度范数和GPU峰值。每次检查冻结主干、VAE与反射adapter没有参数梯度。首次若干步记录LoRA参数切片变化。

M4的验证入口沿用既有随机VAE采样与固定seed2026，所有epoch使用同一协议，计分为保存后RGB8宏平均。训练和教师编码仍使用确定性posterior mode。不要把训练代理loss误当包含全部辅助项的总标量。

## 历史初始化的数据暴露限制

本轮补查Stage2的`run_config.json`：其训练编号与当前validation的25/36/38/46/66组、test的21/43/55组重合。旧源manifest不在，无法逐文件核验这些编号是否对应完全相同的图，但应保守标为继承的潜在训练暴露。

M4第一轮144张训练清单与当前validation/test没有ID或拍摄组交叉。本实验按用户要求保留M4 initial与当前划分，日志保存这项历史限制；结果用于同初始化的回顾性继续训练比较，不能称为完全未见场景的独立泛化验证。若需要严格独立评价，应另留未进入任何初始化训练的拍摄组，不能只在M6阶段重新随机划分。

## 下一次检查

本轮启动并稳定后即可断开SSH，由tmux继续。完成后读取 `training_summary.json`、`best_metrics.json`、`latest_metrics.json` 与验证history，比较同一清单的step0/各epoch；测试集需另行运行，不因训练结束自动使用测试图选参数。
