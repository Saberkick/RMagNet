# SMA：干净内容记忆实验

日期：2026-10-08。分支：`experiment/sma-semantic-memory`。

## 目标与模型身份

在现有单个 M4 DiT 内增加显式干净内容表示，在 Q37 后预测、在 Q39/Q41 后读取。第一轮仅训练两个读取模块，整个 M4 主干、原 Transmission LoRA、VAE 和预训练内容预测器保持冻结。

固定基础权重：

```text
runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors
SHA-256 = 897282b1bb9cfe61f96530df72edcf8a44a066bb819a3663e9100862aefdb2a3
```

以上是本次实际 sha256sum 的 64 位结果。最初设计草稿末尾多写了一个 a，启动身份校验拒绝后已纠正；没有更换 M4 权重。

## 数据口径

当前实际数据是 `m2-variable-aspect-v3-excluded-misaligned`：144 训练、18 验证、17 测试。既有记录已排除配准失败的 `108_421_1388`，本实验遵守该记录，不恢复、不修改原始数据。

manifest SHA-256：`c1e1a6e9db7bbe186d59dace597b3ce8241e22805d945aba1ddd0821e36724d2`。

输入来自纠正后的 blended，GT 来自 transmission_layer，P90 来自 reflection_90。文件名含 `_GT` 不用于推断物理身份；按 manifest 的角色及处理后哈希检查。保留原长宽比，每卡 batch 1，使用已有长宽比分组调度，禁用会破坏缓存绝对位置对应的增强。

测试集包含早期权重接触过的场景；17 曾用于早期验证。既有测试集只能做历史比较，不作为完全未见场景泛化的唯一证据。本次训练不运行测试集。

## 模块

```text
I -> frozen VAE -> frozen M4 blocks 1–37 -> remaining frozen M4 -> decoder -> T
                                   |
                                   +-> fixed PCA256 -> CleanMemory
                                                          |
                                        reads after blocks 39, 41
```

Q37 特征为动态 N×3072。对通道做 LayerNorm，使用训练集等量 token 拟合的固定 PCA，将通道降到 256；不白化。内容预测器含两个局部卷积/区域注意力块，每块 16 个区域 token。输出保留 N×256 二维内容网格，区域 token 不声称对应真实物体。

两个读取模块同时采用同位置内容读取与区域交叉注意力，目标可训练参数共 **4,610,050**。输出投影零初始化；gate 初始 sigmoid(-2)，不同时置零。初始化输出严格回退 M4，随后由输出重建梯度打开读取路径。

真实短跑发现该 Qwen 的残差状态 RMS 约 3–4 百万，单位幅值注入会被 BF16 吞掉。已采用有界的相对残差，架构版本 `sma-rms-v1`：

$$
h_l'=h_l+\sigma(g_l)\,\operatorname{stopgrad}(\operatorname{RMS}(h_l))\,\tanh(\Delta h_l).
$$

RMS 按每个 token 的通道计算；零初始化仍严格恒等。初始 gate 约 0.119，tanh 限制相对注入，避免不受控放大。首轮未缩放短跑虽正常退出，梯度仅约 1e-9、18 张保存 PNG 全部不变，不能作为有效训练验收。修订后必须重新验证梯度与实际输出改变。

Q37 记忆来自同一次前向，不提前读取末层，不增加第二次 DiT 推理。推理只输入 I；GT、P90、DoLP 和教师缓存均不进入推理接口。

## 阶段一：缓存和内容预训练

使用最终 M4-best 为统一冻结教师，保持 LoRA 启用、posterior mode、timestep 499 和固定文本条件。缓存原 M4 损失需要的全部目标与晚层 gate，另存 Q37(I)、Q37(P90)。新缓存目录：

```text
data_cache/sma_m4final_v1/
```

训练预测器将 I 和 P90 的内容表示映射到同一个 GT 内容表示：

$$
M_I=Z_I+f_\phi(Z_I),\qquad M_{90}=Z_{90}+f_\phi(Z_{90}).
$$

内容项采用中心化 token 的余弦距离；关系项匹配水平/垂直相邻 token 的余弦相似度：

$$
L_{mem}=0.7L_{content}+0.3L_{relation}.
$$

$$
L_{pre}=\tfrac12L_{mem}(M_I,Z_Y)+\tfrac12L_{mem}(M_{90},Z_Y)+0.1L_{amplitude}.
$$

幅值项为按训练 PCA 标准差缩放后的 Smooth L1，弥补余弦距离不约束幅值的问题。预训练 5 个 feature epoch、每 epoch 144 更新、每次同时处理 I/P90，一共 720 次小网络更新。学习率 1e-4，AdamW，weight decay 0.01，裁剪 1.0。

预训练报告明确只报告训练拟合，不以此宣称语义泛化。正式生成器由独立 18 张验证集选权重。P90 只是另一混合观测，不是纯反射 GT。

输出：`runs/sma_memory_pretrain/memory.safetensors` 和 `report.json`。该文件含固定投影、预训练 memory 和零初始化读取模块。

## 阶段二：正式训练 20 epoch

固定预训练 memory，训练读取模块。保持 M4 输出损失：

$$
L_{rec}=L_1+0.2(1-SSIM)+0.1L_{edge}.
$$

$$
L_{base}=\tfrac12L_{rec}(T_I,Y)+\tfrac12L_{rec}(T_{90},Y)+0.10L_{polar}.
$$

L_polar 为双向 stop-gradient 的一致性约束，使用 1+G 加权 Charbonnier。普通输出另有原 M4 的空间恢复/保持、Q16/Q20 纹理、Q37/Q39/Q41 内容与邻接关系三组监督。各组目标输出梯度比例为基础项的 8%，36 updates 渐入，EMA 0.9，合并上限 25%。这些是梯度比例，不是直接乘到 loss 数值上的固定系数。

本实验统一缓存教师和在线教师为同一份最终 M4，并在教师前向旁路所有新模块，修复旧轮“缓存开 LoRA、在线关 LoRA”的表示不对称。新读取模块不能读取依赖 GT 的 gate，gate 只用于训练监督。

| 参数 | 值 |
|---|---|
| GPU | 0,1,2,3，最多四张 |
| 每卡 batch / 累积 | 1 / 1 |
| 有效 batch | 4 |
| 每 epoch 更新 | 144/4 = 36 |
| 正式 epoch / 更新上限 | 20 / 720 |
| 读取模块学习率 | 1e-4 |
| warmup / scheduler | 20 updates / cosine |
| optimizer | AdamW，weight decay 0.01 |
| backbone / adapter 精度 | NF4 + BF16 / 新模块 FP32 |
| 梯度裁剪 | 1.0 |
| 验证 | 每 epoch，18 张保存后 8 位 PNG |
| best 选择 | 最小验证宏平均 L1 |
| 提前结束 | 默认关闭早停，按要求训练 20 epoch；遇异常直接失败 |
| 保留 | best/latest SMA；不保存优化器状态 |

每卡冻结权重依旧占用显存；四卡 DDP 不将显存合并。读取模块后的冻结 DiT/VAE 仍传播输入梯度，不能使用 no_grad。教师输出梯度先算完释放，再重算生成器反传，沿用 M4 的内存策略。

checkpoint 重算显式捕获各次前向的启用状态、二维网格与 detached memory，防止教师旁路开关影响生成器旧图重算。新增模块不修改原始 LoRA 状态，不会在教师阶段把 LoRA 意外解冻。

## 命令

环境由 uv 管理，复用个人目录的 Python 3.12 环境与现有权重，不安装新软件或驱动，不使用 sudo/Docker。

```bash
./bin/xuke
cd /share/linmingheng-local/xuke/RMagNet

# 首次准备；四卡分片缓存，随后 GPU0 做小网络预训练
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/prepare_sma.sh

# 前台训练（epoch 可选，最大20）
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 bash scripts/train_sma.sh 20

# 后台训练；不重复覆盖已有运行目录
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 bash scripts/background_sma.sh 20

# 只读查看，无需 attach
tail -n 3 runs/sma_e20/metrics.jsonl
tail -n 20 runs/sma_launch/sma_e20.console.log
tmux ls

# 日后手动评估；本次启动任务不运行封存测试
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_sma.sh
```

后台 session：`sma_e20`。控制台日志：`runs/sma_launch/sma_e20.console.log`。训练结果：`runs/sma_e20/`。退出码：`runs/sma_launch/sma_e20.exitcode`，仅进程结束后出现。tmux 能承受 SSH 断开，不能承受机器重启；本次不提供优化器级恢复。

检查 GPU 空闲和基础权重哈希通过才启动，不覆盖非空 run。默认只运行一个 R2 候选，不同时训练全部架构消融；原图记忆、空间错位、关闭语义项等对照另做同预算实验，不能由本次单组训练推断因果贡献。

## 验证与验收记录

小型 GPU 检查 `python -m src.rmagnet.sma_check` 已验证：恒等初始化；非零读取梯度；参数更新；冻结 memory；权重保存重载；15×52、44×18 动态网格；教师前向与 checkpoint 重算隔离。

正式运行前另做短四卡训练，验证真实 Qwen/VAE/教师反传与 PNG 评价。短跑参数不作为正式初始化；正式运行重新从零初始化读取模块与同一预训练 memory 开始。短跑结果只证明流程可执行，不能证明模型效果。

启动实测与最终提交号补记在 `SMA_LAUNCH_RECORD.md`。正式训练稳定若干步后结束人工检查并断开 SSH，不等待 20 epoch 完成。
