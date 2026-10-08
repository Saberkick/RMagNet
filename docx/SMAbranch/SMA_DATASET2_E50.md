# SMA data_set2 / 50 epoch

## 当前状态
错配实验 runs 已清理，记录保存在 `results_archive/SMA_gtnoise5/`。新数据已上传且 SHA-256 一致。**等待用户确认新数据 I/GT 映射及合并/独立方案，未生成缓存或启动训练。**

## 数据审计
- 本地源：`D:/Develop/PhotoManager/data_set2`，308 文件 / 77 组 / 46 拍摄编号，约 38 MB。
- 原始压缩包：`/share/linmingheng-local/xuke/tmp/data_set2.zip`。
- SHA-256：`198237d2e26f019cfe38d0b05e8f0c27b55a1a0218b1efddb0678f0db88dca75`。
- 每组普通 JPEG、GT JPEG、P90 JPEG、DoLP PNG，四图逻辑尺寸一致。
- 新编号与旧数据拍摄编号无重叠。抽查 `_GT.jpg` 更干净，但仍要求明确确认命名，避免旧数据标签颠倒问题。
- 保留本地原始数据；上传为副本。

## 处理与划分
RGB Lanczos、DoLP BOX；不裁剪、不补边、不放大；约 512×384 像素预算，两边均为16倍数。新图最大215040像素，长宽比误差最大2.669%。同一编号全部进入同一集合，长宽比分桶只安排全局step。训练数量按完整拍摄组微调为4倍数，确保各样本每epoch恰好出现一次。

| 方案 | 训练 | 验证 | 测试 | 每epoch更新 | 50epoch更新 |
|---|---:|---:|---:|---:|---:|
| 旧数据+新数据 |204|26|26|51|2550|
| 仅新数据 |60|8|9|15|750|

合并时旧144/18/17划分完全保留，新增60/8/9；旧封存17张可以单独报告。旧错位排除样本不会重新加入。对源文件哈希与处理后GT哈希执行跨集合重复检查。

## 缓存
目标数据 `/share/linmingheng-local/xuke/datasets/rmagnet_sma_dataset2`，缓存 `data_cache/sma_dataset2_v1`。使用固定最终 M4 LoRA 教师，保存 Q16/Q20 输入与GT、Q37/Q39/Q41 GT、Q37输入和P90，以及Q52/Q54/Q56差异产生的late gate。只缓存训练集。合并时旧144个训练缓存可按四图SHA与教师SHA一致性硬链接复用，新增60个重新生成；仅新数据时60个全部生成。

## 训练
固定最终M4，重新拟合训练集PCA与语义memory（5个feature epoch），然后从零初始化reader训练，避免沿用错配实验状态。普通输入和P90共同训练、同一GT；不取消L_polar。推理只输入普通图。

- 四GPU（默认0,1,2,3），每卡batch1、累积1、BF16/NF4固定主干。
- reader-only 4,610,050可训练参数，memory/M4/VAE冻结。
- 沿用SMA损失与控制：重建L1+.2(1-SSIM)+.1edge；普通/P90均值；polar系数.10；空间/纹理/语义目标各8%基础输出梯度，总辅助上限25%，36步渐进。
- AdamW，LR1e-4，WD.01，20步warmup，cosine，梯度裁剪1。
- 50完整epoch，关闭早停；每epoch验证，按保存PNG的macro L1最小选best。
- 仅保留best_sma.safetensors和latest_sma.safetensors，无优化器状态和中间epoch权重；保存best/latest验证图、指标与日志。
- 结果目录 `runs/sma_dataset2_e50`；memory初始化 `runs/sma_dataset2_memory_pretrain`。

## 脚本与启动条件
- `src/rmagnet/sma_dataset2.py prepare`：必须显式提供 `--labels gt-suffix` 或 `--labels bare-gt`；合并时加 `--base /share/linmingheng-local/xuke/datasets/rmagnet_m2_aspect`。
- `sma_dataset2.py reuse-cache`：只复制来源一致的缓存硬链接。
- `scripts/background_sma_dataset2.sh`：tmux中依次缓存、memory预训练、正式50epoch；任何阶段失败则停止，不进入后续。
- `runs/sma_launch/dataset2_e50.console.log` / `.exit_code` 保存阶段及退出状态。
- 启动前GPU检查及至少16GiB磁盘余量检查。

## 已完成的检查
Python语法编译、bash -n、确定性分组划分、四卡整epoch覆盖条件、新旧编号不重叠、上传哈希一致。尚未执行GPU训练验证；用户确认数据定义后再完成准备与启动检查。
