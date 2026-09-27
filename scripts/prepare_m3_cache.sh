#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/linmingheng-local/xuke"
PROJECT="$ROOT/RMagNet"
PYTHON="$ROOT/envs/windowseat-py312/bin/python"
NVIDIA_USER_LIB="$ROOT/lib/nvidia-535.179"
OUTPUT="${OUTPUT:-$PROJECT/data_cache/m3_semantic_v1}"
GPU_LIST="${GPUS:-0,1,2,3}"
LOG_DIR="$PROJECT/runs/m3_semantic_cache"

export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="$NVIDIA_USER_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

cd "$PROJECT"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
SHARDS="${#GPU_ARRAY[@]}"
if [[ "$SHARDS" -lt 1 ]]; then
  echo "No GPUs selected" >&2
  exit 2
fi

PIDS=()
for INDEX in "${!GPU_ARRAY[@]}"; do
  GPU="${GPU_ARRAY[$INDEX]}"
  LOG="$LOG_DIR/extract_shard_${INDEX}.log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m src.rmagnet.m3_cache \
    --output "$OUTPUT" extract \
    --device cuda:0 \
    --shard-index "$INDEX" \
    --num-shards "$SHARDS" \
    >"$LOG" 2>&1 &
  PIDS+=("$!")
  echo "started shard=$INDEX gpu=$GPU pid=${PIDS[-1]} log=$LOG"
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    FAILED=1
  fi
done
if [[ "$FAILED" -ne 0 ]]; then
  echo "At least one M3 extraction shard failed; inspect $LOG_DIR" >&2
  exit 1
fi

FINAL_GPU="${GPU_ARRAY[0]}"
CUDA_VISIBLE_DEVICES="$FINAL_GPU" "$PYTHON" -m src.rmagnet.m3_cache \
  --output "$OUTPUT" finalize --device cuda:0 \
  2>&1 | tee "$LOG_DIR/finalize.log"

"$PYTHON" -m src.rmagnet.m3_cache \
  --output "$OUTPUT" check --cleanup-scratch \
  2>&1 | tee "$LOG_DIR/check.log"

echo "M3 semantic cache complete: $OUTPUT"
