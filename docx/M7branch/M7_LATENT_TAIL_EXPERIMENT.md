# M7：冻结 M4，在解码前恢复细节

## 目的与边界

验证 M4 已去除反射的内部表示是否还能支持恢复文字、边缘和细纹理。冻结 M4 final best 的 Qwen 主干、原有 LoRA、VAE Encoder 和 Decoder；新增恢复模块以 GT 为监督。第一轮 5 epoch，无早停，不运行封存测试集或 real20。

这不是第二次对保存图片执行 WindowSeat，也不复制整个 20B DiT。先增加两个兼容隐藏特征维度的尾部 block；尚不能保证它优于像素恢复方案。

## 模型来源

- 基础模型：现有固定快照 Qwen-Image-Edit-2509，沿用项目模型锁与 uv Python 3.12 环境。
- 去反射模型：`RMagNet/runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors`。
- M4 SHA-256：`897282b1bb9cfe61f96530df72edcf8a44a066bb819a3663e9100862aefdb2a3`。
- 复制 M4 第 59、60 个 block（索引 58、59），以实际 NF4 解量化主干加 M4 LoRA 构成 BF16 初始化。复制基座冻结，新增 rank 8 LoRA 可训练；这不是复制原生 Qwen 的未适应参数。
- 独立分支：`experiment/m7-m4-latent-tail`；独立工作目录：`/share/linmingheng-local/xuke/RMagNet-M7`。其他工作目录的改动不迁移、不覆盖。

## 结构

```text
I -> frozen VAE Encoder -> zI -> frozen M4 -> H60 and zM4
                                      |
H60 + final text tokens/time/RoPE -> copied blocks 59,60 + new LoRA
                                      |
                             LayerNorm + 3072->128 projection
                                      |
I -> trainable shallow texture encoder + zI + zM4 -> convolutional fusion
                                      |
                               bounded latent residual Δz
                                      |
                         zM4 + Δz -> frozen VAE Decoder -> Tfinal
```

$$
(H_{60},z_{M4})=F_{M4}(E(I))
$$

$$
\Delta z=0.25\tanh\left(R_{\phi}(H_{60},z_I,z_{M4},A_{\phi}(I))\right)
$$

$$
\hat T=D(z_{M4}+\Delta z)
$$

输出卷积零初始化，因此初始 latent 与 M4 完全相同；用真实前向验证。新 LoRA 的梯度第一步可能为零：输出头先学到非零权重后才向前传递梯度，第 3 次更新检查 LoRA 梯度必须非零。

原图纹理分支三层 stride=2 卷积，得到 1/8 分辨率特征。Qwen token 特征从 1/16 网格插值到 1/8 latent 网格；结合输入 latent 与 M4 latent，经 128 通道融合和三个局部残差块输出 16 通道修正。不是直接把原图像素加回来，避免无条件恢复反射。

冻结 Decoder 的参数，但保留 Decoder 对 latent 的梯度。冻结 M4 缓存与原始 GT 分离，恢复模块的输入只依赖 I，不依赖 GT、P90、DoLP。

## 数据与缓存

复用 `datasets/rmagnet_sma_dataset3` 已纠正标签的数据及原有拍摄分组划分：204 train、25 validation、24 sealed test；实际数量以配置与清单检查为准。保留现有动态尺寸及比例，不重新缩图，不更改标签。

缓存仅对 train/validation 输入图生成，VAE posterior mode，时间步 499，固定 M4 提示词。缓存 H60、最终文本 token、时间嵌入、mask、RoPE、zI、zM4。GT 不参与缓存前向。封存测试不生成缓存、不评估。

缓存每样本原子保存并记录输入/缓存 SHA-256、M4/数据集身份和尺寸，断点执行可复用已校验样本。完整缓存通过后才训练。复制尾部基座另存一次，训练 checkpoint 不重复保存基座。

由于缓存含位置编码，不直接翻转、裁剪或打乱 token 进行增广。

## 损失

$$
\rho(e)=\sqrt{e^2+10^{-6}}
$$

$$
L_{rec}=\operatorname{mean}\rho(\hat T-GT)
$$

$$
L_{gradient}=\tfrac12\left[\operatorname{mean}\rho(\nabla_x\hat T-\nabla_xGT)+\operatorname{mean}\rho(\nabla_y\hat T-\nabla_yGT)\right]
$$

$$
L_{multiscale}=\tfrac12\sum_{s\in\{2,4\}}\operatorname{mean}\rho(\operatorname{AvgPool}_s(\hat T)-\operatorname{AvgPool}_s(GT))
$$

$$
L_{correction}=\operatorname{mean}|\Delta z|
$$

$$
L=L_{rec}+0.10L_{gradient}+0.20L_{multiscale}+0.005L_{correction}
$$

重建对齐整体；梯度项监督真实笔画和边缘；多尺度项约束颜色和局部结构；小幅修正项降低对已恢复内容的破坏。固定系数是首轮起点，未根据测试集选择。该轮不加入额外 Qwen 教师损失或 DoLP，便于判断结构是否有效。

## 预算与评价

| 参数 | 首轮 |
|---|---|
| GPU | 1、2、3 三卡 DDP；最多支持四卡 |
| 每卡 batch | 1 |
| 每卡梯度累积 | 2 |
| 有效 batch | 6 |
| epoch | 5，完整遍历训练集 |
| 每 epoch 更新 | 34（204 / 6） |
| 总更新 | 170 |
| LoRA rank / scale | 8 / 1（等效 alpha=rank） |
| 学习率 | 1e-4，新 LoRA 与恢复分支相同 |
| 优化器 | AdamW，weight decay 1e-4 |
| 调度 | 10 步 warmup，cosine 至 0.1 倍学习率 |
| 精度 | BF16 autocast，训练参数/优化器 FP32 |
| 梯度裁剪 | 1.0 |
| 保存 | best；latest 含优化器，可续训 |
| 模型选择 | 最低验证集 RGB8 平均 L1，允许 epoch0 基线胜出 |

每 epoch 在 25 张验证图上保存 PNG，计算 PSNR、SSIM、L1、边缘误差及低/高变化区误差，并以同协议 M4 基线对照。SSIM 沿用项目 uniform 11×11、zero padding 5 的定义，不和其他协议混排。记录各损失、新 LoRA 梯度、修正幅度、逐 rank 显存及实际更新数。

若最后全局 batch 不满，固定补齐并记录重复样本；该三卡配置无需补齐。最后文本输出分支部分参数可能未参与图像损失，DDP 启用 unused-parameter 检测。

## 运行与输出

```bash
cd /share/linmingheng-local/xuke/RMagNet-M7
CUDA_VISIBLE_DEVICES=1,2,3 EPOCHS=5 OMP_NUM_THREADS=1 \
  bash scripts/run_m7_latent_tail.sh
```

后台启动使用本实验专用 tmux 会话，不占用或关闭其他会话。SSH 断开不终止任务。

```text
data_cache/m7_m4_tail_v1/
  frozen_tail.safetensors
  frozen_tail.json
  samples/ records/ logs/ manifest.json
runs/m7_m4_latent_tail_e5/
  config.json
  logs/train.log
  train.jsonl metrics.jsonl
  best/model.safetensors
  latest/model.safetensors
  latest/trainer.pt
  best_metrics.json latest_metrics.json
  best_validation/ latest_validation/
  training_summary.json
```

完整推理仍需先执行冻结 M4，再接本恢复模块；缓存只是冻结部分的等价预计算，不是部署输入要求。checkpoint 仅包含新增可训练权重，必须与固定复制尾部基座和 M4 身份配套。

续训到累计 10 epoch：

```bash
RESUME=1 EPOCHS=10 CUDA_VISIBLE_DEVICES=1,2,3 \
  bash scripts/run_m7_latent_tail.sh
```

实际启动与梯度/保存检查结果另记 `M7_STARTUP_REPORT.md`。首轮结论仅来自验证集，未证明泛化提升。
