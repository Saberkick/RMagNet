#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RUN_DIR="${RUN_DIR:-/share/linmingheng-local/xuke/RMagNet/runs/stage2_smoke}"
export EPOCHS=1
export WARMUP_STEPS=1
export SAVE_EVERY=1
export VALIDATE_EVERY=1
export KEEP_CHECKPOINTS=1
export RESUME=none

exec "$(dirname "$0")/run_stage2.sh" --max-steps "${MAX_STEPS:-1}"
