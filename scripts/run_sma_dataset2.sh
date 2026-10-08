#!/usr/bin/env bash
set -euo pipefail
PROJECT=/share/linmingheng-local/xuke/RMagNet
ROOT=/share/linmingheng-local/xuke
export DATA_ROOT="$ROOT/datasets/rmagnet_sma_dataset2"
export CACHE_ROOT="$PROJECT/data_cache/sma_dataset2_v1"
export MEMORY_DIR="$PROJECT/runs/sma_dataset2_memory_pretrain"
export RUN_DIR="$PROJECT/runs/sma_dataset2_e50"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export EPOCHS=50 EARLY_STOPPING_PATIENCE=0 MAX_STEPS=0
export LEARNING_RATE=1e-4 NUM_WORKERS=1 MEMORY_EPOCHS=5
unset CONTINUE_RUN
cd "$PROJECT"
source scripts/sma_env.sh
[[ "$(df --output=avail -B1 "$ROOT" | tail -1 | tr -d ' ')" -ge 16106127360 ]] || { echo 'Need 16 GiB headroom before starting'; exit 6; }
printf 'PHASE=cache %s\n' "$(date -u +%FT%TZ)"
bash scripts/prepare_sma.sh
printf 'PHASE=train %s\n' "$(date -u +%FT%TZ)"
bash scripts/train_sma.sh 50
printf 'PHASE=complete %s\n' "$(date -u +%FT%TZ)"
