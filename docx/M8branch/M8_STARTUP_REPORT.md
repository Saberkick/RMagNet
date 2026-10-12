# M8 启动记录

日期：2026-10-12（客户端上海日期）。

- 已在独立目录 `/share/linmingheng-local/xuke/RMagNet-M8` 启动。
- 分支：`experiment/m8-m4-spatial-gates`；tmux会话：`m8-spatial-gates`。
- 运行目录：`runs/m8_m4_gates_14_37_54_e5/`。
- GPU1、2、3模型并行，一份冻结M4；有效batch6；5 epoch/170次优化更新；无早停。
- 可训练参数 610,758，只属于第14、37、54层新增空间门控。
- 官方完整前向与启用初始门控的latent完全一致；核对大尺寸训练图 `130_5411_2729`，两个max error均为0。
- 启动调试曾发现时间步BF16构造顺序与VAE autocast边界差异，已改为直接沿用官方完整transformer forward，并清理旧3步探针的权重/图片/优化器产物。最终实验从新的零输出门控初始化，不续用探针。
- 已观察 4 次实际更新；三层梯度有限且非零，三张门控输出卷积的checkpoint权重均已改变，原DiT/LoRA/VAE均无梯度。
- latest门控与optimizer/scheduler状态已保存，cursor为 `{'epoch': 0, 'batch': 4, 'step': 4}`。
- 最近显存峰值（逻辑0/1/2对应物理1/2/3）：`{'0': 9.343053817749023, '1': 5.2247161865234375, '2': 5.236752986907959}` GiB；无OOM。
- 门控开始产生空间差异；当前仍接近1是恒等初始化和初期小更新的预期表现，不代表已证明有效。
- 25张验证基线：PSNR 24.0578 dB、SSIM 0.875925；不是封存测试。
- 不停止其他项目，不继续监控完整5 epoch；SSH退出后由tmux继续运行。

## 查看

```bash
./bin/xuke
cd /share/linmingheng-local/xuke/RMagNet-M8
tail -n 20 runs/m8_m4_gates_14_37_54_e5/logs/train.log
tmux attach -t m8-spatial-gates
```

Ctrl+B再按D可脱离tmux。完整结构、loss、数据来源、预算与续训命令见 `M8_SPATIAL_GATES_DESIGN.md`。
