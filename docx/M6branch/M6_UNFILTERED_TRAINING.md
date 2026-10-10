# M6：不筛方向的五 epoch 实验

日期：2026-10-10。本文件为当前执行方案，替代原始strict方向筛选与未采用的GT校准候选方案。

## 1. 用户指定的修改

不要求P90-I45与I45-GT同向；不翻转候选轴，不按方向角度剔除样本或位置。去掉两层共识、整图1%置信度门禁、差分幅度门槛及饱和筛选。仅保留数值有限性检查和零向量的除零保护。

新增项使用原始P90-I45差异轴和GT锚点，**正负投影误差均计算平方**。原设计的正半轴ReLU截断也取消，避免方向相反时又将惩罚关闭。GT用于恢复目标，P90-I45用于指定额外约束的候选轴。这项损失不是纯反射层真值，可能与已有GT语义loss冗余；本轮先实测。

## 2. 新 loss

教师为冻结且关闭全部LoRA的Qwen；每图token均值中心化后按通道归一化。中层为Q37/39/41，固定timestep499、文本与确定性VAE posterior mode。

$$
Z_l(X)_x=\operatorname{normalize}\left(Q_l(X)_x-\operatorname{mean}_u Q_l(X)_u\right)
$$

$$
U_l(x)=\operatorname{normalize}\left(Z_l(P90)_x-Z_l(I45)_x\right)
$$

$$
r_l(x)=\left\langle Z_l(F_\theta(I45))_x-Z_l(GT)_x,U_l(x)\right\rangle
$$

$$
L_{axis}=\operatorname{mean}_{l\in\{37,39,41\}}
\frac{\sum_x V_l(x)r_l(x)^2}{\sum_x V_l(x)+\epsilon}
$$

V只表示有限且非零的数学方向；其余token等权，不乘晚层G、置信度或DoLP。数值不有限直接报错，差异恰为零时该轴不产生梯度。日志仍沿用`negative`命名，但其当前数学定义是双侧投影平方。`negative_positive_residual_fraction`只记录投影正号比例，不用于筛选或loss截断。

## 3. 保留 M4 生成能力和监督

初始化M4 initial best，step612：

`/share/linmingheng-local/xuke/RMagNet/runs/m4_e30_p4/best_transmission_lora.safetensors`

SHA-256：`5725d32b04e1271d51a33f7512174f1035ff0acf5e5427bdd3b179e98e1a13eb`。

冻结DiT主干和VAE，只训练rank128 LoRA_T。生成启用LoRA；预测特征教师及全部VJP完成期间关闭LoRA，再恢复生成。

$$
L_{rec}=L_1+0.2(1-SSIM)+0.1L_{edge}
$$

$$
L_{base}=\tfrac12L_{rec}(F(I45),GT)+\tfrac12L_{rec}(F(P90),GT)+0.10L_{polar}
$$

**Lpolar保留**，沿用M4双向stop-gradient一致性距离。空间项继续使用晚层G，纹理项继续使用Q16/20，正向内容与邻接关系项继续使用Q37/39/41。取消G筛选只针对新增轴loss，未删除原有空间监督。

辅助项通过输出图梯度控制加入：空间8%、纹理8%、正向语义6%、新增轴2%；语义合并上限8%，全部辅助上限25%。36更新渐入、EMA0.9、当前梯度硬限幅；百分比是梯度预算，不是固定loss系数，也不保证每步实际比例恰好到达目标。

## 4. 数据与成功缓存复用

数据`rmagnet_sma_dataset3`，204 train / 25 validation / 24 test；按原manifest读取I/GT/P90，未改标签或划分，不读取DoLP。保留尺寸与长宽比分桶，禁用对缓存不一致的空间增强。

旧strict预检已正确提取全部204张的教师参考及原始U，只是筛选权重全为零。成功特征无需重新计算。

将原目录转换为`data_cache/m6_unfiltered_v3/`，更新规则、记录与manifest。原safetensors包保持原字节与SHA，含原始早层、GT中层和U；旧零权重字段占用很小，作为原包的一部分保留但**训练绝不读取它来加权**，加载器按新规则生成等权V。新manifest明确记录这种物理存储与逻辑权重的区别，旧strict目录和失败预览已移除。

完整文件哈希校验由正式训练rank0执行一次，其他rank在初始化/barrier等待。所有rank检查文件大小、教师与规则身份，每次取样检查特征有限性。原特征写出时已有逐文件SHA，现场核验的新鲜GT特征与旧教师特征一致。

## 5. 训练预算与运行入口

| 项目 | 配置 |
|---|---|
| GPU | 物理1、3，启动时转换为UUID；保留GPU2任务 |
| Epoch | 5，完整遍历训练集，无早停 |
| Batch | 每卡1，有效batch2，梯度累积1 |
| 优化更新 | 102/epoch，共510 |
| 优化器 | PagedAdamW8bit，学习率5e-6，weight decay0.01 |
| 调度 | warmup20，cosine，梯度裁剪1.0 |
| 精度 | BF16，冻结DiT为NF4 |
| 验证 | 初始及每epoch，25张保存后8位PNG，宏平均PSNR/SSIM/L1/LPIPS |
| Best | 验证集宏平均L1最低；允许仍为初始化 |
| 保存 | 仅best/latest LoRA；无epoch权重、无优化器副本 |

```bash
./bin/xuke
cd /share/linmingheng-local/xuke/RMagNet-M6
M6_DIRECTION_MODE=unfiltered CUDA_VISIBLE_DEVICES=1,3 \
EPOCHS=5 OMP_NUM_THREADS=1 bash scripts/run_m6_pipeline.sh
```

后台tmux为`m6_b_e5`。日志`runs/m6_pipeline_e5/console.log`，阶段`phase.txt`，最终退出码`pipeline.exit_code`。训练输出`runs/m6_b_e5/`，包含`metrics.jsonl`、`status.json`、`validation/best`、`validation/latest`、每epoch指标history及best/latest权重。

验证沿用M4固定seed随机VAE采样；训练与教师采用确定性posterior mode。历史Stage2可能接触过部分当前验证/测试拍摄组，具体编号见run_config的lineage审计；本轮评价属于同初始化的回顾性比较。未自动使用封存测试集或real20选参数。

## 6. 清理范围

已删除闲置缓存：`m4_multilayer_v1`、`m4_best_multilayer_v1`、`m2a_q20`、`sma_m4final_v1`、`sma_gtnoise5_v1`、`sma_dataset2_v1`，以及M6失败run临时目录。保留JSON/CSV等元数据与小日志在`results_archive/M6/cleanup_20261010/`，M4历史训练manifest另保留在本docx目录，审计不再依赖已删除的张量缓存。

GPU2任务正在使用的`ws_c1_m5_dataset3_s1_e20_layers`、其memory文件与训练权重保留。M4初始化、各成功模型权重、原始/处理后数据、基础模型和uv环境保留。独立前置模型项目未在本轮清理名单中。

清理后的文件系统剩余约47GiB。清理报告的逻辑文件量包含硬链接重复计数，不能把它直接当作实际释放空间。

## 7. 稳定性验收

数值检查包括GT零点、双侧残留非零、正确下降方向、反向候选轴也参与、零向量保护和梯度上限。正式启动需确认真实loss与输出梯度非零、LoRA参数发生变化、冻结主干/VAE无梯度、两卡都更新且无OOM，然后断开SSH；不持续监督五epoch完成。
