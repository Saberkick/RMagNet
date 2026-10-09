#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "$(dirname "$0")/sma_env.sh"
IFS=',' read -r -a cards <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#cards[@]} == 1 ]] || { echo 'M5-R trains on one GPU; select a single idle card' >&2; exit 2; }
MIN_FREE_MIB=4096 check_gpus
EPOCHS="${1:-${EPOCHS:-30}}"
RUN_NAME="${RUN_NAME:-m5_c1_pixel_e30}"
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ && "$RUN_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
uvpython -m src.rmagnet.m5_train --epochs "$EPOCHS" --patience "${PATIENCE:-4}" --min-epochs 5 --accumulation "${ACCUMULATION:-4}" --output "$PROJECT/runs/$RUN_NAME"
