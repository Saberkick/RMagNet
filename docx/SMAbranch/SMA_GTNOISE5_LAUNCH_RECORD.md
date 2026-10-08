# SMA GT-noise5: launch record

- Branch: `experiment/sma-gtnoise5-e4`; code commit: `5ff8e48a9ca76ce0537defc3799965ee7db08b2f`.
- Four GPUs 0-3; 4 image epochs, 144 optimizer updates planned; no early stopping.
- Fixed 7/144 wrong GTs; new PCA/memory initialization and five feature epochs. Original SMA best/latest are not used to initialize this run.
- Verified 14 formal updates, with 2 noisy training observations already consumed. Losses finite, nonzero reader gradients, M4/VAE/memory frozen; auxiliary/base ratio within 25% cap.
- Latest checked reader gradient norm: 0.00256140; allocated memory peak: 17.886 GiB.
- Zero-output initialization verified; saved-PNG initial validation PSNR is identical to the original M4 baseline.
- Cache audit: all seven GT feature sets bitwise match the intended clean donor cache; all four samplers cover each of the seven noisy samples exactly once per epoch; validation/test labels unchanged.
- Memory pretraining loss: 0.2823596 -> 0.2523825 (training fit only, not quality evidence).
- Original SMA best/latest SHA-256 remain unchanged after cleanup. Deleted smoke/temporary/initialization artifacts: 176.18 MiB in total. Reports, evaluation images, necessary caches, best/latest retained.

## Background run

- tmux: `sma_gtnoise5_e4`.
- Results: `runs/sma_gtnoise5_e4/`.
- Console: `runs/sma_launch/sma_gtnoise5_e4.console.log`.
- Exit status after completion: `runs/sma_launch/sma_gtnoise5_e4.exitcode`.
- Only best/latest SMA weights are retained; initialization-only memory files were deleted after all training ranks finished loading. Fresh reruns require regenerating memory pretraining; no optimizer recovery is provided.
- SSH is disconnected after stable startup; no polling, no automatic sealed-test evaluation, and no wait for four epochs to finish.

See `SMA_GTNOISE5_E4.md`, `materials/SMA_GTNOISE5_SETUP.json`, and `materials/SMA_GTNOISE5_LAUNCH.json` for mapping, loss/schedule details, and provenance.
