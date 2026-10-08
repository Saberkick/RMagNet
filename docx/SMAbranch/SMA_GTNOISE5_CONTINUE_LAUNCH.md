# GTnoise5: stable continuation to cumulative epoch 20

Code: `65d936b9ecbb6c34df90e64ee51e861d624ae54e`, branch `experiment/sma-gtnoise5-e4`.

Parent: `runs/sma_gtnoise5_e4/latest_sma.safetensors`; SHA-256 `e49d21356f014ab15a75cbf3af461ea356a27aaeadc35b0cb42ae118336cf223`. Start after 4 epochs / 144 steps. Add 16 epochs / 576 updates, end at cumulative 20 epochs / 720 steps. Same fixed seven incorrect GTs, four GPUs 0-3, frozen M4/VAE/memory, reader-only training, no early stopping. Optimizer, 576-step cosine scheduler and controller EMA restarted; 20 new updates warmup to LR 1e-4. No optimizer moments restored.

Parent latest validation exactly reproduced: PSNR 24.02377457, SSIM 0.85349508. First new step is 145, epoch index 4. Verified 25 new updates; latest checked global step 169; gradient norm 0.02172818; allocated GPU peak 17.840 GiB. Finite losses, nonzero updates, frozen parameter checks and 25% auxiliary gradient cap passed. Four GPU process contexts present. Parent latest unchanged; historical best inherited.

## Outputs

- tmux: `sma_gtnoise5_e20_continue`.
- Run: `runs/sma_gtnoise5_e20_continue/`.
- Console: `runs/sma_launch/sma_gtnoise5_e20_continue.console.log`.
- Exit code on completion: `runs/sma_launch/sma_gtnoise5_e20_continue.exitcode`.
- Keep only best/latest; preserve epoch-4 run, metrics and test results. Best is still step 0 at handoff.

SSH disconnected after stable startup. No wait for remaining epochs; no automatic sealed-test evaluation. See `SMA_GTNOISE5_CONTINUE_E20.md` and `materials/SMA_GTNOISE5_CONTINUE_LAUNCH.json`.
