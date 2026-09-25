# M2 数据处理报告

## 状态

M2 的 180 组数据已经完成确定性缩放、七类长宽比分桶和 train/validation/test 分组划分。本节点没有加载 VAE/Qwen，没有生成 Q20 缓存，也没有训练模型。

## 数据位置

- 原始 ZIP：`/share/linmingheng-local/xuke/datasets/rmagnet_m2_source/data_set.zip`
- ZIP SHA-256：`051645ee5c88f96fa967c790213b2d0501857f73378a92ed5c55d9d9751ce774`
- ZIP 大小：692,112,677 bytes，磁盘显示约 661 MiB。
- 处理后数据：`/share/linmingheng-local/xuke/datasets/rmagnet_m2_aspect`
- 处理后大小：约 163 MiB。
- 完整 manifest：`/share/linmingheng-local/xuke/datasets/rmagnet_m2_aspect/manifest.json`
- 总览预览：`/share/linmingheng-local/xuke/datasets/rmagnet_m2_aspect/previews/overview.png`

处理后的目录：

```text
rmagnet_m2_aspect/
├── blended/             # 180 张待处理输入 PNG
├── reflection_90/       # 180 张 90°反射增强 PNG
├── dolp/                # 180 张单通道 DoLP PNG
├── transmission_layer/  # 180 张 GT PNG
├── splits/
│   ├── train.txt
│   ├── validation.txt
│   └── test.txt
├── previews/
├── manifest.json
└── README.md
```

## 尺寸处理

基准像素预算为：

\[
512\times384=196608
\]

每张图根据自身原始长宽比单独计算目标宽高，然后取 16 的倍数。处理过程不裁剪、不 padding、不主动放大，并对同一组 `I / P90 / DoLP / GT` 使用完全相同的目标尺寸。

- 处理后宽度：240～688，中位数 464。
- 处理后高度：272～864，中位数 416。
- 像素数：176,640～217,600，中位数 194,560。
- 最大长宽比相对误差：1.6598%。
- 最窄样本 `47_3404_1435`：`628×2254 → 240×864`，比例误差 0.3008%。
- 最宽样本 `105_2252_1177`：`3459×1399 → 672×272`，比例误差 0.0765%。

RGB 输入、P90 和 GT 缩小时使用 Lanczos。DoLP 作为标量场缩小时使用 BOX area approximation。输出全部保存为 PNG，避免再次有损压缩。

## 七个长宽比桶

桶仅用于统计和后续多卡采样调度；桶不会把图像拉伸成统一宽高。

| 桶 | 全部 | Train | Validation | Test |
|---|---:|---:|---:|---:|
| 极竖 `<0.50` | 7 | 5 | 1 | 1 |
| 竖 `[0.50,0.75)` | 36 | 29 | 3 | 4 |
| 轻竖 `[0.75,0.90)` | 21 | 17 | 2 | 2 |
| 近方 `[0.90,1.10)` | 24 | 18 | 3 | 3 |
| 轻横 `[1.10,1.50)` | 50 | 41 | 5 | 4 |
| 横 `[1.50,2.00)` | 23 | 19 | 2 | 2 |
| 极横 `≥2.00` | 19 | 15 | 2 | 2 |

Validation 和 test 都覆盖七个桶。

## 数据划分

| Split | 样本数 | 拍摄组数 | 用途 |
|---|---:|---:|---|
| Train | 144 | 108 | 梯度更新 |
| Validation | 18 | 12 | 每个 epoch 评价、选 checkpoint 和 early stopping |
| Test | 18 | 12 | 超参数固定后的最终报告 |

划分种子为 2026。分组键是 sample ID 第一个下划线之前的拍摄编号。同一拍摄编号下的多个区域样本只能进入同一个 split，避免同源内容同时出现在训练集和测试集。

自动检查确认：

- 三个 split 的 180 个 ID 并集完整；
- 三个 split 两两无 ID 交集；
- 三个 split 两两无拍摄组交集；
- 每个模态均为 180 张；
- 每组四张处理图尺寸相同；
- RGB 模态为 RGB，DoLP 为单通道 L；
- manifest 标记为 complete；
- 所有源文件和处理文件都记录 SHA-256。

## 代码

- `src/rmagnet/m2_prepare_data.py`：扫描 ZIP、验证配对、规划尺寸、分桶、group split、缩放、哈希、manifest 和预览。
- `scripts/prepare_m2_data.sh`：使用固定 uv 环境启动预处理。

预处理采用 staging 目录。只有全部 180 组成功并完成检查后，才原子发布到正式数据目录，避免把中断结果误认为完整数据。

## 下一步闸门

从七个桶各选至少一张，加上最小和最大原图，执行冻结 VAE 编解码和 Qwen block 20 前向检查。确认动态 token 网格、输出尺寸和显存峰值后，再修改 C1-L20 cache builder 生成 M2 的完整 Q20 缓存。
