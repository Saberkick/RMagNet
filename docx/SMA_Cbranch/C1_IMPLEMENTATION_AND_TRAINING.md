# C1：语义条件编辑与收益门控

日期：2026-10-09。分支 `experiment/sma-conditioned-c`。本文是实现与运行规范；结果由训练完成后的报告给出。

## 1. 为什么实施 C

B4 已证实全量 Transmission LoRA 与 SMA readers 均参与训练，但第2 epoch后验证收益回落。自有26张的 best 相对M4只有 +0.0235 dB，real20为 -0.2221 dB。因此不能把“解冻LoRA”本身当成足够的改进。

C1 改变语义与编辑的连接方式：不再向主干叠加原SMA残差，而是以完整空间记忆调节LoRA内部低秩通道，并对“条件是否真的改善编辑”增加训练期监督。C1没有错配GT，也没有条件扰动；错误条件扰动C2留待C1有效后单独验证。

## 2. 结构与初始化

输入普通图 I，经冻结VAE及带学生LoRA的Qwen，在第37块得到空间特征。冻结的PCA投影和预训练CleanMemory生成256维逐token记忆。记忆不反传；这保留了原方案的资源边界，也仍存在学生Q37表示漂移的限制。

Q39/Q41图像FFN的 `img_mlp.net.0.proj` 与 `img_mlp.net.2` 共4处安装LoRA-A输出钩子。文本、attention Q/K及其他LoRA位置不加空间调节，但全部Transmission LoRA继续训练。

$$
z_l=A_lh_l,\quad
\Delta h_l=B_l\left[\left(1+\gamma_l g_l(C)\tanh U_l(C)\right)\odot z_l\right]
$$

$$
\gamma_l=0.25\tanh(a_l),\qquad g_l(C)=\sigma(H_l(\operatorname{stopgrad}(C)))
$$

- A/B从M4-best初始化，rank128，原LoRA alpha128、dropout0。
- U：LayerNorm256→Linear256→GELU→Linear128，非零小随机初始化。
- H：LayerNorm256→Linear64→GELU→Linear1，末层零初始化，初始gate=0.5。
- 原始a从0初始化，因此完整模型初始严格复现M4；gamma和U没有同时全零。
- a的LR为5e-3（短跑用1e-3，正式提高其预设学习速度），其他条件/门控参数1e-4。gamma幅度最多0.25，LoRA低秩通道倍率位于[0.75,1.25]；这是稳定性限制，不是最优性保证。
- 初始化不使用SMA50/B4已学习的readers，不从smoke继续。

可训练：完整学生Transmission LoRA、4个条件投影和4个gate头。冻结：量化原始DiT、VAE、PCA/记忆、独立M4监督教师。新条件/门控模块464,904参数；学生LoRA852,180,992参数，总计852,645,896个可训练参数，写入run_config.json。

## 3. 主任务损失

沿用B4的基础、空间、纹理和语义监督，保留L_polar：

$$
L_{rec}(Y,G)=L_1(Y,G)+0.2(1-SSIM(Y,G))+0.1L_{edge}(Y,G)
$$

$$
L_{base}=\frac12\left[L_{rec}(F(I),GT)+L_{rec}(F(P90),GT)\right]+0.10L_{polar}
$$

普通I额外计算空间项、Q16/Q20纹理项、Q37/Q39/Q41语义项，P90保持基础监督。三项各自目标输出梯度为基础项8%，36步渐入，EMA0.9，系数范围[1e-4,10]，合成辅助梯度上限25%。

这些是输出空间梯度比例，不能等同于LoRA、门控或语义项各自的参数更新占比。沿用固定M4教师与 `data_cache/sma_dataset2_v1`；预测特征在线计算，GT特征读取缓存。教师LoRA驻CPU，分阶段换入计算VJP再恢复学生，任何活跃计算图不得跨越权重切换。

## 4. 编辑收益校准

正式训练第1 epoch不校准；从第2 epoch起每8次更新执行一次。先完成当前普通I/P90主任务反传，尚未更新参数时，用相同学生、同一普通I latent做两次no-grad前向：语义强制关闭得到T0，强制完全开启得到T1。

Q37位于全部干预之前，因此两次记忆张量必须逐元素相同；代码强制检查。这两个输出都不是旧M4输出，T1也不依赖待学习gate。

$$
e(Y,G)=\operatorname{AvgPool}_{9\times9}\left(\operatorname{Mean}_{RGB}|Y-G|+0.1\big[\operatorname{Mean}_{RGB}|\nabla_xY-\nabla_xG|+\operatorname{Mean}_{RGB}|\nabla_yY-\nabla_yG|\big]\right)
$$

误差在模型训练张量[-1,1]域中计算，池化边界不把额外零值计入平均；随后自适应平均到token网格。

$$
\delta=\operatorname{stopgrad}\left(e(T_0,GT)-e(T_1,GT)\right),\quad
y=\sigma(\delta/\tau),\quad w=\min(1,|\delta|/\tau)
$$

$$
L_{reliability}=\frac14\sum_{l=1}^{4}\frac{\sum_x w(x)BCEWithLogits(H_l(C(x)),y(x))}{\sum_xw(x)+10^{-8}}
$$

tau由训练样本T0局部误差均值给出，截断[0.002,0.1]，四rank平均后EMA0.9；不读取验证/测试误差调tau。标签和记忆detached，仅对4个gate头计算此项梯度；不能通过改变候选输出操纵自己的标签。

对DDP平均后的主任务gate梯度与校准gate梯度分别测量范数，取：

$$
\lambda_{rel}=\min\left(1,\frac{0.1\|g_{gate,main}\|}{\|g_{gate,rel}\|+10^{-12}}\right)
$$

主任务gate梯度为0时，校准也不得单独推动gate。记录实测比例、gate均值、gamma、条件开启/关闭的输出L1差、收益为正的token比例。合并后仍执行全局梯度裁剪1.0。

局部收益标签来自“4处同时开启”的候选，所以不能解释为每处独立因果贡献；所有gate使用同一个收益目标。这是最小实现的明确限制。没有独立验证集校准前，gate不是可信语义正确概率。所有gate关闭、gamma接近0或开启条件不改变图像，都应报告为条件路径未被有效利用。

## 5. 正式训练预算与产物

- 数据204训练/26验证/26测试，不改变原划分与长宽比。4GPU，每卡batch1，累积1，每epoch51更新。
- 本轮固定10epoch=510更新，无早停，观察第2epoch后是否继续退化；只保留验证L1最优best与最终latest，不保留epoch权重或优化器状态。
- 学生LoRA LR5e-6、PagedAdamW8bit、CPU优化器状态卸载；条件模块FP32 AdamW；warmup20后cosine，weight_decay0.01，gamma无weight_decay。
- GPUs0–3，最多4张24GB；不触碰4–7。
- 默认目录 `runs/sma_c1_e10/`，tmux `sma_c1_e10`，总日志 `runs/sma_launch/c1_e10.console.log`，完成状态 `c1_e10.exit_code`。
- `best_sma.safetensors` / `latest_sma.safetensors`是完整学生LoRA+条件模块+记忆联合包，约3.2GiB各；architecture=`sma-conditioned-c1-v1`，不能按旧SMA权重载入。
- 每epoch验证；结束后自动评估best/latest的test26与real20，保存PNG、逐图CSV及REPORT.md。另在同一best模型、同一26张验证集上评估条件强制off/on，与正常门控验证结果比较，检验语义路径是否实际有益；这不是另行训练的C0。训练结束不等于总流程结束，要看exit_code与四项evaluation.json。

```bash
EPOCHS=10 CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 bash scripts/run_sma_c.sh
```

旧测试部分场景与Stage2模型来历存在重叠；“封存”仅指当前划分，不能称为完整训练历史从未见过。详见M5branch历史审计。real20和历史测试已用于方案讨论，本轮只能作为回顾性泛化诊断，不是全新盲测。

## 6. 交付前验收及结论边界

短跑必须覆盖至少一次可靠性校准；验证初始化复现M4、4个hooks实际工作、LoRA与条件参数更新、冻结模块无梯度、校准不超过10%及checkpoint重载复现PNG。短跑两步的原始gamma约1e-3，开启/关闭条件尚不改变最终BF16图像；临时将raw gamma设0.2的无更新诊断得到输出L1差0.001602，并得到非零可靠性梯度、约10%cap，因此确认路径有效。正式将raw gamma LR从1e-3提高为5e-3，仍零初始化和0.25有界调节，以缩短条件路径起效时间；这不是已验证的最佳学习率。短跑记录归档后删权重和预测；正式训练fresh启动。

C0（相同结构但gate恒1）及同预算LoRA-only尚未运行。10epoch与B4的4epoch预算不同，不能把最终变化单独归因于门控。后续训练选择使用验证集；封存测试不用于每epoch选择最佳模型。

相关代码：`sma_conditioned.py`、独立 `sma_c_train.py`、`sma_c_report.py`、兼容C的两个评估入口；B4训练代码保留。C2扰动训练和M5像素恢复本轮均不启动。
