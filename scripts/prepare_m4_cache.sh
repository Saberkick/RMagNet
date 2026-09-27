#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
ENVIRONMENT="$ROOT/envs/windowseat-py312"
UV_BIN="${UV_BIN:-/home/xuke/.local/bin/uv}"
DATA_ROOT="${M4_DATA_ROOT:-$ROOT/datasets/rmagnet_m2_aspect}"
CACHE_ROOT="${M4_CACHE_ROOT:-$PROJECT/data_cache/m4_multilayer_v1}"
GPUS="${GPUS:-0,1,2,3}"
MIN_FREE_MIB="${MIN_FREE_MIB:-22000}"

export LD_LIBRARY_PATH="$ROOT/lib/nvidia-535.179${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_OFFLINE=1
export TMPDIR="$ROOT/tmp"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.80"

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "M4 cache preparation requires four comma-separated GPU indices; got $GPUS" >&2
  exit 1
fi
declare -A SEEN
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ || -n "${SEEN[$gpu]:-}" ]]; then
    echo "Invalid or duplicate GPU index: $gpu" >&2
    exit 2
  fi
  SEEN[$gpu]=1
  free_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if (( free_mib < MIN_FREE_MIB )); then
    echo "GPU $gpu has only ${free_mib} MiB free; need ${MIN_FREE_MIB} MiB" >&2
    exit 3
  fi
done
if [[ ! -x "$UV_BIN" || ! -x "$ENVIRONMENT/bin/python" ]]; then
  echo "Pinned uv/Python runtime is unavailable" >&2
  exit 4
fi
if [[ ! -f "$DATA_ROOT/manifest.json" ]]; then
  echo "Corrected M2 dataset is unavailable: $DATA_ROOT" >&2
  exit 5
fi

mkdir -p "$CACHE_ROOT/logs" "$ROOT/tmp" "$UV_CACHE_DIR"
cd "$PROJECT"
pids=()
for shard in 0 1 2 3; do
  gpu="${GPU_ARRAY[$shard]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" \
    python -m src.rmagnet.m4_cache extract \
    --data-root "$DATA_ROOT" \
    --output "$CACHE_ROOT" \
    --device cuda:0 \
    --shard-index "$shard" \
    --num-shards 4 \
    > "$CACHE_ROOT/logs/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "At least one M4 cache shard failed. Inspect $CACHE_ROOT/logs/" >&2
  exit 6
fi

"$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" \
  python -m src.rmagnet.m4_cache finalize \
  --data-root "$DATA_ROOT" \
  --output "$CACHE_ROOT"

echo "M4 cache complete: $CACHE_ROOT"
