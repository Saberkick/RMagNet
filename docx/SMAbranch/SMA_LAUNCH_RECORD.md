# SMA formal training launch record

## Status at handoff

- Server: `srtp_xuke6`; project: `/share/linmingheng-local/xuke/RMagNet`.
- Branch: `experiment/sma-semantic-memory`; training code: `ff829b5624f76e23dce6581f86bc98ef42095ecb`.
- Architecture: `sma-rms-v1`; frozen final M4 + pretrained clean-content memory + trainable Q39/Q41 readers.
- 20 epochs, 144 train images, four GPUs (0,1,2,3), per-GPU batch 1, accumulation 1: 36 updates/epoch, 720 planned updates. No early stopping.
- Verified 17 formal updates before handoff; finite losses, nonzero reader gradients, frozen M4/VAE/memory, actual auxiliary/base gradient ratio within the configured 25% cap. This is a startup check, not a completed experiment or quality claim.
- Latest checked reader gradient norm: 0.00631289; peak allocated GPU memory: 17.886 GiB.
- Four training processes occupy GPUs 0-3. GPUs 4-7 were not used.
- Formal initialization uses `runs/sma_memory_pretrain/memory.safetensors` with fresh zero-output readers, never a smoke checkpoint.

## Background session and outputs

- tmux session: `sma_e20`; pane PID at launch: `245793`.
- Console: `runs/sma_launch/sma_e20.console.log`.
- Per-update logs/config: `runs/sma_e20/metrics.jsonl`, `runs/sma_e20/run_config.json`.
- Best/latest: `runs/sma_e20/best_sma.safetensors`, `runs/sma_e20/latest_sma.safetensors`.
- Validation images/metrics: `runs/sma_e20/best_validation/`, `runs/sma_e20/latest_validation/`.
- Completion: `runs/sma_e20/training_summary.json`; launcher exit code: `runs/sma_launch/sma_e20.exitcode` (not present while running).
- Only new SMA state is saved; fixed M4 is referenced by SHA-256. No optimizer restore is provided.

```bash
./bin/xuke
cd /share/linmingheng-local/xuke/RMagNet
tmux ls
tail -n 3 runs/sma_e20/metrics.jsonl
tail -n 20 runs/sma_launch/sma_e20.console.log
```

SSH is disconnected after this stable startup check. tmux continues independently; no automatic polling or supervision is installed. Machine restarts are not covered.

## Short verification and limits

The corrected four-GPU three-update smoke completed successfully and changed all 18 saved validation PNGs. A previous unscaled prototype was ineffective because its residual was too small relative to Qwen activations; RMS-scaled residual reads corrected the gradient path. Full evidence: `materials/SMA_PREFLIGHT_AND_SMOKE.json`.

The five-epoch feature-memory pretraining reduced training feature loss from 0.2569354 to 0.2257314. This does not establish semantic quality or downstream generalization. Formal validation selects the minimum macro-average saved-PNG L1; sealed test and real20 are not run during this launch task. The current dataset preserves the earlier exclusion of misregistered sample `108_421_1388`: 144 train, 18 validation, 17 sealed test.

See `SMA_IMPLEMENTATION_AND_TRAINING.md` for losses, gradient control, cache provenance, and module details. Machine-readable startup evidence: `materials/SMA_FORMAL_LAUNCH.json`.
