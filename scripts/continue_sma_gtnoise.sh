#!/usr/bin/env bash
set -euo pipefail
ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
export DATA_ROOT="$ROOT/datasets/rmagnet_sma_gtnoise5"
export CACHE_ROOT="$PROJECT/data_cache/sma_gtnoise5_v1"
export CONTINUE_RUN="$PROJECT/runs/sma_gtnoise5_e4"
export RUN_DIR="$PROJECT/runs/sma_gtnoise5_e20_continue"
export EPOCHS=20 MAX_STEPS=0 EARLY_STOPPING_PATIENCE=0 LEARNING_RATE=1e-4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" NUM_WORKERS="${NUM_WORKERS:-1}"
exec bash "$PROJECT/scripts/train_sma.sh" 20
