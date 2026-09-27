# M3 阶段一：离线语义缓存生成报告

> 状态：完成并通过独立审计  
> 日期：2026-09-27  
> 分支：`experiment/m3-semantic-separation`  
> 缓存版本：`m3-semantic-separation-v1`

## 1. 产物

缓存目录：

```text
/share/linmingheng-local/xuke/RMagNet/data_cache/m3_semantic_v1/
```

最终规模：

- 训练样本：144。
- Q20 image token：110,324。
- 文件：295。
- 占用空间：约 65 MB。
- 临时 `E_I/E_90` scratch 已在审计通过后删除。
- validation/test 没有参与 PCA、原型、归一化统计或关系图拟合。

主要文件：

```text
data_cache/m3_semantic_v1/
├── manifest.json
├── pca/
│   ├── content_pca.safetensors
│   └── residual_pca.safetensors
├── prototypes/
│   ├── prototypes.safetensors
│   └── fit_report.json
├── samples/                  # 144 个紧凑训练缓存
├── previews/                 # 144 个可视化面板
└── audit/
    ├── cache_check.json
    └── cluster_occupancy.json
```

## 2. 实际生成流程

1. 严格校验纠正后的 M2 数据 manifest 和 144 个 train ID。
2. 复用现有 `Q20(GT)` 作为干净内容表示 `C`。
3. 使用四张 GPU 并行提取：

   \[
   E_I=Q20(I)-C,\qquad E_{90}=Q20(P90)-C
   \]

4. 教师为关闭全部 LoRA 的固定基础 Qwen，使用 block 20、索引 19、timestep 499 和确定性 VAE posterior mode。
5. 分别拟合 32 维内容 PCA 与共享残差 PCA。
6. 构造不含绝对坐标的污染状态特征。
7. 使用三种固定种子拟合 `K=4` balanced Sinkhorn 软原型。
8. 生成 soft posterior、置信度、反射证据、软边界及稀疏内容关系边。
9. 独立重新读取所有紧凑缓存，检查源图哈希、张量范围、posterior 和、关系索引和簇容量。
10. 审计通过后删除完整残差 scratch。

## 3. PCA 与软状态结果

| 项目 | 结果 |
|---|---:|
| 内容 PCA 32维解释比例 | 0.659707 |
| 残差 PCA 32维解释比例 | 0.528806 |
| 三种初始化最小原型稳定性 | 1.000000 |
| 状态0占比 | 0.249997 |
| 状态1占比 | 0.249981 |
| 状态2占比 | 0.249991 |
| 状态3占比 | 0.249994 |

四个状态保持为无人工命名的 soft responsibility，不作为硬 T/R 标签。

## 4. 审计结果

`audit/cache_check.json` 的最终状态为 `passed`：

- 样本数：144。
- token 数：110,324。
- 源文件哈希错误：0。
- FP16 存储后 posterior 求和最大误差：`0.000244140625`。
- 关系索引全部在各自 token grid 范围内。
- 反射证据覆盖 `[0,1]`。
- 七种长宽比桶全部保留。
- scratch 已删除。

哈希：

```text
manifest.json
2fd98c2811d383f4325cb22e2e6002f91f504ccd5830bc824af186f833d7b920c

audit/cache_check.json
00cd2daf4c76871f1c9b1dd36b626becf851757a8f4b211597a555935a1daf90
```

## 5. 显存与环境

提取使用 GPU 0–3、每卡单样本顺序前向：

- 最大 allocated：15.049 GiB。
- 最大 reserved：15.158 GiB。
- 没有发现随样本增长的显存累积。

服务器当时的系统用户态 NVIDIA 库指向 580，而运行中的内核驱动为 535.179。没有修改系统文件或使用 sudo；只在个人目录创建：

```text
/share/linmingheng-local/xuke/lib/nvidia-535.179/
```

M3 脚本通过 `LD_LIBRARY_PATH` 显式加载与内核匹配的 535.179 库。PyTorch 2.8.0+cu126 随后能识别全部八张 RTX 3090。

## 6. 代码入口

```bash
# 可恢复地生成完整缓存
GPUS=0,1,2,3 bash scripts/prepare_m3_cache.sh

# 只读复查现有缓存
bash scripts/check_m3_cache.sh
```

实现文件：

- `src/rmagnet/m3_cache.py`
- `scripts/prepare_m3_cache.sh`
- `scripts/check_m3_cache.sh`

生成 manifest 记录的实现提交为：

```text
27e2a31ae5c52eab1e53beefc8c94856235edeb1
```

## 7. 当前边界

阶段一已完成。尚未实现或启动阶段二训练、Smoke、M3-P、M3-S、M3-F 或置乱对照。
