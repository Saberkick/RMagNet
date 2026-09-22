# M1b semantic strength: 10% versus 30%

Both groups used the same Stage 2 initial LoRA, 50 training images, seed 2026,
2 epochs and 100 optimizer steps. The only intended change is the semantic
coefficient, tripled from the original single-batch 10% gradient calibration.
Values below come from saved 8-bit PNGs at 512×384 on IDs 11/12/17.

| Group | Arm | PSNR ↑ | SSIM ↑ | Mask PSNR ↑ | Mask L1 ↓ | Outside L1 ↓ |
|---|---|---:|---:|---:|---:|---:|
| 10% | base | 23.115082 | 0.821463 | 20.192964 | 0.074587 | 0.039660 |
| 10% | dolp | 23.142321 | 0.821748 | 20.210213 | 0.074683 | 0.039501 |
| 10% | shuffle | 23.123438 | 0.821429 | 20.206872 | 0.074715 | 0.039709 |
| 30% | base | 23.125546 | 0.821531 | 20.201764 | 0.074705 | 0.039684 |
| 30% | dolp | 23.133847 | 0.821689 | 20.172124 | 0.075058 | 0.039389 |
| 30% | shuffle | 23.114592 | 0.821360 | 20.218828 | 0.074397 | 0.039938 |

## Matched differences (B − A, B − C)

Positive PSNR/SSIM and negative L1 mean B improved over its comparator.

| Group | Contrast | ΔPSNR | ΔSSIM | ΔMask PSNR | ΔMask L1 | ΔOutside L1 |
|---|---|---:|---:|---:|---:|---:|
| 10% | dolp − base | +0.027238 | +0.000285 | +0.017249 | +0.000096 | -0.000159 |
| 10% | dolp − shuffle | +0.018883 | +0.000319 | +0.003340 | -0.000032 | -0.000208 |
| 30% | dolp − base | +0.008301 | +0.000158 | -0.029640 | +0.000353 | -0.000295 |
| 30% | dolp − shuffle | +0.019255 | +0.000329 | -0.046703 | +0.000661 | -0.000550 |

Inspect the three saved image triplets before interpreting small metric changes.
A stronger DINO gradient does not itself establish semantic or text fidelity.
