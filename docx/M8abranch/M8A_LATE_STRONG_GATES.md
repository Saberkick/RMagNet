# M8a：冻结 M4 的末段强门控，2 epoch

## 目标与依据

M8 的第14、37层门控在封存测试中的平均绝对变化仅约0.074%、0.114%，第54层约1.12%。latest 的封存PSNR/SSIM为24.8975/0.801451，M4为24.8994/0.801283：没有清晰的整体改善。灰度门控图不是反射分割标签，接近1说明模型基本沿用原编辑更新。

本轮从冻结 M4 final best 重新初始化，只在第54、58个block（索引53、57）的图像注意力残差和MLP残差上训练空间门控。集中调整末段更新，扩大可调幅度，检查前一轮是否受限于控制能力。调整位置和参数化同时发生，因此结果只能评价这套组合，不能单独证明是哪项带来收益。不会续训此前已裁剪成仅推理用途的M8 latest。

## 结构

每层门控读取该层输入图像token，LayerNorm→Linear(3072,64)→SiLU→3×3 depthwise convolution→SiLU→1×1 convolution(64,2)，输出attention/MLP两张空间图。两个门控头合计407,172个可训练参数。输出卷积权重、偏置置零；其他权重随机初始化，seed2026。

$$
G(x)=2\sigma(\kappa u(x)),\qquad G_{\mathrm{init}}(x)=1,\qquad 0<G(x)<2.
$$

$$
H_{\mathrm{attn}}=H+G_{\mathrm{attn}}\odot\Delta H_{\mathrm{attn}},\qquad
H_{\mathrm{out}}=H_{\mathrm{attn}}+G_{\mathrm{MLP}}\odot\Delta H_{\mathrm{MLP}}.
$$

门控乘在原注意力/MLP编辑残差上，不能把原始图像或整个隐藏状态直接清零。小于1削弱该层编辑，大于1加强；黑色不等于判定为反射区域。text分支保留原运算。网格为H/16×W/16，不增加图像裁剪或重采样。

第一个epoch的34次优化更新中，κ从1线性升至4；第二个epoch固定4。每次更新内保持κ一致，防止梯度检查点重算时改变前向函数。

$$
\kappa(s)=1+3\min\left(1,\frac{s}{33}\right),\qquad s=0,\ldots,67.
$$

κ=4时，初始化处logit导数为2，旧参数化为0.125。但Adam会按梯度历史归一化，不能把16倍logit灵敏度理解成16倍实际参数更新。能否出现更强且有用的空间选择，要看实测门控、验证误差和最终封存结果。

## 初始化、冻结与数据

- M4：`/share/linmingheng-local/xuke/RMagNet/runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors`，SHA-256 `897282b1bb9cfe61f96530df72edcf8a44a066bb819a3663e9100862aefdb2a3`。
- 冻结Qwen DiT、M4 LoRA、VAE与固定文本嵌入，只训练新增门控。复用已有NF4主干及BF16激活，门控和优化器FP32。完整官方transformer.forward，保持原timestep的BF16数值顺序。
- 数据：`/share/linmingheng-local/xuke/datasets/rmagnet_sma_dataset3`。204训练、25验证、24封存测试，保持现有动态尺寸和划分，不重生成任何教师缓存。
- manifest SHA-256：`bb211d1de9d399c685d70e80ef292d9a86b9e6733fb78cf1c1724d0d591bb77c`。
- 模型只输入普通I；GT仅参与损失和评价。P90/DoLP不输入门控。

## 损失与预算

$$
L=L_{\mathrm{Charbonnier}}(\hat T,GT)+0.1L_{\mathrm{gradient}}(\hat T,GT)+0.01L_{\mathrm{gate}}.
$$

$$
L_{\mathrm{gate}}=\frac{1}{2}\sum_{\ell\in\{54,58\}}\operatorname{mean}\left((G_{\ell}-1)^2\right).
$$

每层mean同时包含两张门控图。系数与M8相同；此实验不恢复M4训练期的Qwen辅助损失。底座能力来自M4权重；本轮通过GT重建和边缘目标训练末段门控。

| 项目 | 配置 |
|---|---|
| GPU | 物理1、2、3，单模型跨卡切分，不是DDP |
| 每次前向 | 1张；连续6张累积一次更新 |
| epoch | 完整遍历204张训练图 |
| 更新数 | 每epoch34，2epoch共68 |
| AdamW | lr1e-4，weight decay1e-4 |
| 学习率 | 前10更新warmup，cosine衰减到初始lr的10% |
| 梯度裁剪 | 1.0 |
| 验证 | 每epoch，对25张验证图 |
| best选择 | 保存后RGB8 PNG的平均L1，最小改善1e-8；未超过M4时保留初始化best |
| 保存 | best、latest；latest带优化器/调度器状态，可恢复 |
| 停止 | 2epoch，无早停 |

保存预测PNG后重新读取计算PSNR/SSIM，不混用浮点输出指标。SSIM保持项目uniform11×11、zero-pad5的现有口径。训练日志记录两层attention/MLP各自的范围、空间标准差、平均绝对变化、近边界比例、κ、梯度范数及各卡显存峰值。

## 自动检查和运行

启动先核对文件哈希、数据配对、GPU可用显存和磁盘。最大训练图上验证κ=1及κ=4的零初始化门控都与原M4 latent完全一致；检查新增门控梯度有限且非零、第一步参数变化，冻结参数没有梯度。异常直接退出并保留日志。

```bash
./bin/xuke
cd /share/linmingheng-local/xuke/RMagNet-M8a
CUDA_VISIBLE_DEVICES=1,2,3 EPOCHS=2 bash scripts/run_m8a_late_gates.sh
```

后台使用tmux会话`m8a-late-strong-gates`。uv复用个人Python3.12环境；不下载模型、不安装系统软件、不终止其他用户任务。

```text
runs/m8a_m4_late_strong_gates_e2/
├── config.json
├── identity_check.json
├── logs/train.log
├── train.jsonl
├── metrics.jsonl
├── best/gates.safetensors
├── best/state.json
├── latest/gates.safetensors
├── latest/state.json
├── latest/trainer.pt
├── m4_baseline_validation/
├── best_validation/
├── latest_validation/
└── training_summary.json
```

推理必须同时加载M4底座和对应gate权重，并读取checkpoint元数据中的κ设置，不能默认κ=1。门控预览固定0黑/1中灰/2白，不逐图拉伸对比度。当前启动只完成训练及验证，不自动运行封存测试或real20；训练完成后再对固定best/latest进行同口径比较。

## 判定

先看门控变化是否确实增加且仍有空间差异，再看验证集L1、PSNR、SSIM是否改善。门控强度上升但误差变差，说明更强控制本身不能解决涂抹；门控仍接近1，则需审视输入特征、损失或门控只能调节已有残差的能力边界。仅两epoch有收益也不能保证更长训练或外域泛化。
