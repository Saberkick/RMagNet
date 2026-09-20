#!/usr/bin/env bash
set -euo pipefail
ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
SMOKE_CACHE="$PROJECT/cache/stage3_smoke"
SMOKE_RUN="$PROJECT/runs/stage3_smoke_2gpu"
CUDA_VISIBLE_DEVICES="${CACHE_GPU:-0}" bash "$PROJECT/scripts/prepare_stage3_cache.sh" \
  --cache-root "$SMOKE_CACHE" --ids 13,11
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" RUN_DIR="$SMOKE_RUN" CACHE_ROOT="$SMOKE_CACHE" \
  SAVE_EVERY=1 VALIDATE_EVERY=1 bash "$PROJECT/scripts/run_stage3.sh" \
  --train-ids 13 --val-ids 11 --max-steps 1 --resume none
