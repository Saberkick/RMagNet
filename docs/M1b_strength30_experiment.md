# M1b：语义损失强度 10% → 30% 对照

本分支只改变**语义项系数**。模型、损失定义、DoLP 阈值、DINO 层、训练/验证索引、Stage 2 初始 LoRA、随机种子、学习率、每组 2 epoch 和 100 次更新均保持 M1b 原实验设置。

原校准记录 `runs/m1b/sem_calibration.json` 使用训练图 13，在起点让语义项对预测 RGB 的梯度范数约为基础损失的 10%，得到 `λ=0.002936870578313556`。由于输入、权重和校准点不变，30% 试验使用**恰好 3 倍**系数：`λ=0.008810611734940669`。这是起点单张图上的目标梯度比例，不保证训练全过程处处为 30%。脚本验证原校准的样本、种子、DINO 层、阈值和 10% 记录，并另存 `runs/m1b/sem_calibration_f30.json`，不覆盖原文件。

## 手动运行

进入服务器后，等待 `./bin/xuke` 打开新 bash，再单独切换目录：

```bash
ssh srtp_xuke
./bin/xuke
cd /share/linmingheng-local/xuke/RMagNet
git branch --show-current  # experiment/m1b-semantic-30pct
bash scripts/m1b_strength30.sh prepare
```

确认 GPU 2、3 空闲后，依次运行三组。每条命令只运行一组；中途失败时已完成的组不用重跑。`base` 与原实验一样没有 DINO 项，`dolp` 使用真实蒙版，`shuffle` 使用同面积置乱蒙版。

```bash
CUDA_VISIBLE_DEVICES=2,3 bash scripts/m1b_strength30.sh base
CUDA_VISIBLE_DEVICES=2,3 bash scripts/m1b_strength30.sh dolp
CUDA_VISIBLE_DEVICES=2,3 bash scripts/m1b_strength30.sh shuffle
CUDA_VISIBLE_DEVICES=2 bash scripts/m1b_strength30.sh eval
```

三个 LoRA 分别保存在 `runs/m1b/strength30_e2/{base,dolp,shuffle}/final_transmission_lora.safetensors`，每个约 3.2 GiB，总计约 9.6 GiB。评估保存 11、12、17 三张图的 8 位 PNG、逐图 CSV 和汇总，并与旧的 `runs/m1b/probe_e2` 10% 实验生成 `runs/m1b/strength30_e2/STRENGTH_COMPARISON.md`。脚本会检查两组均为 100 次更新、同一初始权重、数据索引与训练参数；若不匹配会停止比较。

## 判读顺序

1. 先核对 30% 的 A 是否重现旧 A。如果 A 明显变化，说明存在未控制的随机性或环境变化，不能直接将 B 的变化归因于语义系数。
2. 比较 10% 与 30% 各自的 B−A、B−C，重点看**蒙版内 L1/PSNR**的逐图方向，再看全图 PSNR/SSIM 与非蒙版 L1。
3. 并排看文字、细纹理和新增物体。即使 DINO 特征距离下降，若真实纹理或文字恶化，也不能判为改善。

旧实验中 B 相对 A 的全图 PSNR 仅高约 0.027 dB，蒙版内 L1 略差，C 的蒙版指标也与 B 接近。30% 试验需要出现比这些微小差异更一致的 B 优势，才支持“原语义项太弱”的解释。三张验证图仍只能作为方向性信号，不能证明泛化。
