#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
ENVIRONMENT="$ROOT/envs/windowseat-py312"
UV_BIN="${UV_BIN:-/home/xuke/.local/bin/uv}"
DATA_ROOT="${M2_DATA_ROOT:-$ROOT/datasets/rmagnet_m2_aspect}"
OUTPUT="${M2A_CACHE_ROOT:-$PROJECT/data_cache/m2a_q20}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
MIN_FREE_MIB="${M2A_MIN_FREE_MIB:-22000}"

if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "M2a cache preparation requires exactly one physical GPU index; got CUDA_VISIBLE_DEVICES=$GPU_ID" >&2
  exit 1
fi
if [[ ! -x "$UV_BIN" || ! -x "$ENVIRONMENT/bin/python" ]]; then
  echo "Pinned uv runtime is unavailable" >&2
  exit 1
fi
if [[ ! -f "$DATA_ROOT/manifest.json" ]]; then
  echo "M2 dataset is missing: $DATA_ROOT" >&2
  exit 1
fi

FREE_MIB="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if (( FREE_MIB < MIN_FREE_MIB )); then
  echo "GPU $GPU_ID has only ${FREE_MIB} MiB free; M2a requires at least ${MIN_FREE_MIB} MiB before loading Qwen" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_OFFLINE=1
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$ROOT/tmp"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.80"

mkdir -p "$ROOT/tmp" "$UV_CACHE_DIR" "$(dirname "$OUTPUT")"
cd "$PROJECT"

exec "$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" \
  python -m rmagnet.m2a_prepare \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT" \
  --device cuda:0 \
  "$@"
