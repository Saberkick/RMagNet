#!/usr/bin/env bash
set -euo pipefail
ROOT=/share/linmingheng-local/xuke
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT/lib/nvidia-535.179${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$ROOT/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$ROOT/.cache/uv" UV_OFFLINE=1 TMPDIR="$ROOT/tmp"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJECT"
LOG="$PROJECT/runs/windowseat_official_sealed24_mode"
mkdir -p "$LOG"
trap 'rc=$?; printf "%s\n" "$rc" > "$LOG/exit_code"' EXIT
for gpu in ${CUDA_VISIBLE_DEVICES//,/ }; do
 free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
 (( free>=22000 )) || { echo "Insufficient GPU memory: $gpu $free MiB"; exit 2; }
done
/home/xuke/.local/bin/uv run --no-project --python "$ROOT/envs/windowseat-py312/bin/python" \
 python -m src.rmagnet.windowseat_sealed24_eval
