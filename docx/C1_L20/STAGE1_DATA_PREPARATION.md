# C1-L20 阶段一：Qwen 第 20 层缓存准备

## 运行方式

登录服务器并切换到个人环境后执行：

```bash
./bin/xuke
cd /share/linmingheng-local/xuke/RMagNet
CUDA_VISIBLE_DEVICES=0 bash scripts/prepare_c1_l20.sh
```

脚本使用 uv 管理的固定环境 `/share/linmingheng-local/xuke/envs/windowseat-py312`，并通过 `uv run --no-project --python` 明确使用该环境；同时设置 `UV_OFFLINE=1`，不会解析、升级或下载依赖，也不会在项目内生成新的环境或 lock 文件。Hugging Face 被设为离线模式。

默认输入：

```text
/share/linmingheng-local/xuke/datasets/rmagnet_stage1_512x384
```

默认输出：

```text
data_cache/c1_l20/
├── gt_features/   # 每张训练 GT 的 Q20 FP16 safetensors
├── weights/       # FP16 weight_token[24,32] 与 weight_pixel[384,512]
├── previews/      # D_Q / DoLP / S / W 及四联图
└── manifest.json
```

控制台日志写入 `runs/c1_l20/prepare.console.log`。

## 数据范围

现有数据共 53 对。阶段一严格排除验证编号 `11、12、17`，只为其余 50 对训练图生成缓存。输入角色为：

- `blended/{id}.png`：待处理图 I
- `transmission_layer/{id}.png`：GT
- `dolp/{id}.png`：8 位灰度 DoLP

三者必须均为 512×384，ID 集合、尺寸和 DoLP 模式必须完全一致，否则在加载大模型前中止。

## 特征和权重定义

- 冻结 Qwen Image Edit 2509 主干。
- 禁用全部 LoRA。
- 使用确定性 VAE posterior mode。
- 使用第 20 个 block（代码索引 19）、timestep 499 和 WindowSeat 发布的固定文本 embeddings。
- 提取 `Q20(I)` 与 `Q20(GT)`，计算 token 级余弦差异。

`D_Q` 按每张图原始差异的 2%/98% 分位做稳健 min-max 并截断到 `[0,1]`。DoLP 使用原始 8 位数值除以 255，不做每图对比度拉伸。

```text
S = D_Q * (0.7 + 0.3 * D_DoLP)
W_raw = clip(1 + 2*S, 1, 3)
W = W_raw / mean(W_raw)
```

token 权重双线性上采样到 512×384 后再次做均值为 1 的归一化。

`W_raw` 的范围是 `[1,3]`。最终 `W` 要求均值严格为 1，因此只要图中存在高权重区域，其他区域就会低于 1；最终 `W` 不可能同时保持最小值为 1。清单分别记录两者的统计量。

DoLP 只能把 `D_Q` 乘以 `[0.7,1.0]` 内的系数。代码逐图检查 `0.7*D_Q <= S <= D_Q`，所以 DoLP 自身无法在 `D_Q=0` 的位置产生高权重。

## 缓存与可追溯信息

`manifest.json` 记录：

- 全部 ID、训练 ID和排除的验证 ID
- I、GT、DoLP 的路径与 SHA-256
- 图像尺寸与 Q20 特征形状
- Qwen 和 WindowSeat 仓库 revision
- block 20/index 19、timestep 499、token 网格
- 固定 embeddings 的 SHA-256、tensor 形状和 dtype
- D_Q 的归一化参数、公式版本及逐图权重统计
- 生成代码所在 Git commit 与 UTC 时间
- 各缓存文件路径和 SHA-256

WindowSeat 公开的是固定文本 embeddings，没有发布可逆的原始 prompt 字符串。因此清单用 embeddings 文件哈希作为固定提示条件的精确身份。

完整 `Q20(I)` 不落盘；每张图计算完差异后立即释放。完整 `Q20(GT)` 以 FP16 保存，供后续训练期 Qwen 特征损失直接使用。

## 重跑规则

- 完整缓存再次执行时会报告 `already_complete`，不会覆盖。
- 非空但不完整的输出目录会拒绝混写。
- 确认要从头重建时才使用：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/prepare_c1_l20.sh --overwrite
```
