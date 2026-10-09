#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "$(dirname "$0")/sma_env.sh"
RUN_NAME="${RUN_NAME:-m5_c1_pixel_e30}"
CHOICE="${CHOICE:-best}"; SPLIT="${SPLIT:-test}"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9_-]+$ && "$CHOICE" =~ ^(best|latest)$ && "$SPLIT" =~ ^(test|validation)$ ]] || exit 2
uvpython -m src.rmagnet.m5_train --checkpoint "$PROJECT/runs/$RUN_NAME/$CHOICE.safetensors" --split "$SPLIT" --output "$PROJECT/runs/$RUN_NAME/eval_${SPLIT}_${CHOICE}"
