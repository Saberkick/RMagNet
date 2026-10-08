#!/usr/bin/env bash
set -euo pipefail
ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
export DATA_ROOT="$ROOT/datasets/rmagnet_sma_gtnoise5"
export CACHE_ROOT="$PROJECT/data_cache/sma_gtnoise5_v1"
export MEMORY_DIR="$PROJECT/runs/sma_gtnoise5_memory_pretrain"
export RUN_DIR="$PROJECT/runs/sma_gtnoise5_e4"
export EPOCHS=4 MAX_STEPS=0 EARLY_STOPPING_PATIENCE=0 LEARNING_RATE=1e-4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" NUM_WORKERS="${NUM_WORKERS:-1}"
source "$PROJECT/scripts/sma_env.sh"
mode="${1:-all}"
case "$mode" in all|prepare|train) ;; *) echo 'Use all, prepare, or train' >&2; exit 2;; esac
if [[ "$mode" != train ]]; then
  uvpython -m src.rmagnet.sma_gtnoise \
    --source "$ROOT/datasets/rmagnet_m2_aspect" --output "$DATA_ROOT" \
    --cache-source "$PROJECT/data_cache/sma_m4final_v1" --cache-output "$CACHE_ROOT"
  # Rebuild seven changed caches; pretrain a fresh memory with the same noisy GTs.
  MEMORY_EPOCHS=5 bash "$PROJECT/scripts/prepare_sma.sh"
fi
if [[ "$mode" != prepare ]]; then
  exec bash "$PROJECT/scripts/train_sma.sh" 4
fi
