# M6 方向预检与 GT 校准修订

日期：2026-10-10。**严格单向模式已完成204张训练图缓存，但尚未训练；GT校准模式仅实现代码，待用户选择。**

## 1. 预检结论

204张训练图，没有任何一张通过原新增负项的整图可靠性条件。最大的置信度质量为0.00070538，而设计阈值为0.01。三个中层的每图平均方向余弦在训练集上的中位数分别为Q37=-0.28595、Q39=-0.29347、Q41=-0.28657。

原可靠性判断要求下面两个向量同向：

$$
\Delta_p=Z(P90)-Z(I45),\qquad \Delta_g=Z(I45)-Z(GT)
$$

$$
a=\cos(\Delta_p,\Delta_g)>0.2
$$

此假设未通过当前训练数据检查。负夹角不等于图像标签又反了，也不能据此认定P90一定是干净图：Qwen表示有非线性、全局上下文和归一化作用，且两个差分在I45上符号相反。它们并非可直接按物理反射强度相加的线性分量。

已经在训练图101_2292_1137、103_831_2184重新提取GT，中层特征与旧M4缓存相对L2误差均为0；两个分片的首个复用参考也通过相同检查。已排除本次复用的教师特征不一致问题。未根据测试集调阈值，未强行选百分位区域。

若直接训练，新增负项在全部样本上都是零；本轮会退化为M4续训且正向语义预算从8%降为6%，不能作为有效的M6负项实验。因此全量缓存门禁禁止启动这种训练。

## 2. 建议的最小修订：GT 校准候选轴

保留P90与I45确定的差异轴，由GT决定这个轴上哪一侧对应输入相对GT的误差。

$$
U_0=\frac{\Delta_p}{\max(\|\Delta_p\|,\epsilon)}
$$

$$
U_{GT}=\operatorname{sign}(a)U_0
$$

$$
C=\operatorname{clip}\left(\frac{|a|-0.2}{0.6},0,1\right)
$$

保留差分范数、两层共识、RGB饱和排除、晚层变化门控G与整图1%质量条件。

$$
M_l=\operatorname{sg}(GBC_lV_l)
$$

$$
L_{axis}=\operatorname{mean}_{l\in\mathcal L_{valid}}
\frac{\sum_xM_l(x)[\max(0,\langle Z_l(\hat T)-Z_l(GT),U_{GT,l}\rangle)]^2}
{\sum_xM_l(x)+\epsilon}
$$

这仍惩罚“相对GT、沿候选轴朝输入误差一侧的残留”，但**不再宣称P90-I45的原始正方向必然表示反射增加**。GT同时参与位置可靠性和轴正负方向校准；监督不是独立反射真值，也可能与既有正向GT特征loss冗余。需要训练结果和之后同预算对照验证收益，不能用非零覆盖率证明语义有效。

不修改图像、标签、数据划分或P90重建。Lrec、Lpolar、空间、纹理、正向语义均保持；辅助梯度预算也保持6%正向+2%候选轴、语义合并8%、总辅助25%。

## 3. 两种模式明确隔离

默认模式为`strict`，继续使用原单向规则，缓存为`data_cache/m6_polar_negative_v1/`。用户确认改用校准方向后，设置：

```bash
M6_DIRECTION_MODE=gt-calibrated \
CUDA_VISIBLE_DEVICES=1,3 EPOCHS=5 OMP_NUM_THREADS=1 \
bash scripts/run_m6_pipeline.sh
```

校准模式有独立公式版本、规则哈希与缓存目录`data_cache/m6_gt_calibrated_v2/`。不能让校准规则复用严格模式的方向或权重作为训练输入；需要重新计算I45/P90的方向与权重。严格预检中完整且哈希一致的M4参考部分可通过`M6_REUSE_ROOT`复用，首个参考须再次通过新鲜GT特征核验。未经选择不自动切换到校准规则。

```bash
M6_REUSE_ROOT=/share/linmingheng-local/xuke/RMagNet-M6/data_cache/m6_polar_negative_v1
```

strict与gt-calibrated均有不加载Qwen的数值测试：GT零点、正残留可微、下降方向、空支持零梯度、方向相反时的模式区别，以及25%/8%/2%硬上限。它们只是数值检查，不替代真实GPU训练验证。

## 4. 资源与其他项目

GPU2上的`ws_c1_m5_d3_s1`继续训练，未停止。原3卡缓存启动时GPU0被外部进程临时占用，引发加载OOM；改用GPU1/3并绑定UUID，未触碰该外部进程。现有DiT文件只读，不共享可变LoRA参数或优化器状态，不需要复制基础大权重。

若校准方案获选择并通过缓存门禁，正式5epoch使用204张训练图、有效batch2，102更新/epoch，共510更新，无早停；只保留best/latest适配器。当前不存在已开始训练或已取得训练指标的结论。

完整数值记录在`results_archive/M6/strict_direction_preflight.json`；第一次显存错误与教师核验日志在`results_archive/M6/first_cache_attempt/`。本次诊断可保留记录，但不能把缓存计算称作已经完成5epoch训练。
