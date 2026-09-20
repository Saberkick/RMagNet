#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RUN_DIR="${RUN_DIR:-/share/linmingheng-local/xuke/RMagNet/runs/stage1_smoke}"
export EPOCHS=2
export SAVE_EVERY=2
export VALIDATE_EVERY=2
export WARMUP_STEPS=1
export RESUME=none

exec "$(dirname "$0")/run_stage1.sh" --max-steps "${MAX_STEPS:-2}"
