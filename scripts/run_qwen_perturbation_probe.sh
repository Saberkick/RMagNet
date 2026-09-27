#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
ENVIRONMENT="$ROOT/envs/windowseat-py312"
UV_BIN="${UV_BIN:-/home/xuke/.local/bin/uv}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-4}"
OUTPUT="${OUTPUT:-$PROJECT/runs/qwen_perturbation_probe}"
export LD_LIBRARY_PATH="$ROOT/lib/nvidia-535.179${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "Exactly one physical GPU index is required" >&2
  exit 1
fi
FREE_MIB="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
if (( FREE_MIB < 22000 )); then
  echo "GPU $GPU_ID has only ${FREE_MIB} MiB free" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_OFFLINE=1
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$ROOT/tmp"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.80"

cd "$PROJECT"
exec nice -n 10 "$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" \
  python -m rmagnet.qwen_perturbation_probe \
  --output "$OUTPUT" \
  --device cuda:0 \
  "$@"
