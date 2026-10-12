#!/usr/bin/env bash
set -euo pipefail
ROOT=/share/linmingheng-local/xuke
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT/lib/nvidia-535.179${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$ROOT/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$ROOT/.cache/uv" UV_OFFLINE=1 TMPDIR="$ROOT/tmp"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
IFS=, read -r -a cards <<< "$CUDA_VISIBLE_DEVICES"
(( ${#cards[@]}>=2 && ${#cards[@]}<=4 )) || exit 2
declare -A seen
for gpu in "${cards[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ && -z "${seen[$gpu]:-}" ]] || exit 2
  seen[$gpu]=1
  free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( free>=18000 )) || { echo "GPU $gpu has insufficient free memory"; exit 3; }
done
RUN="$PROJECT/runs/m8_m4_gates_14_37_54_e5"
OUTPUT="${OUTPUT:-$RUN/sealed_test_best_latest}"
cd "$PROJECT"
/home/xuke/.local/bin/uv run --no-project --python "$ROOT/envs/windowseat-py312/bin/python" \
  python -m src.rmagnet.m8_sealed_eval \
  --data "$ROOT/datasets/rmagnet_sma_dataset3" \
  --initial "$ROOT/RMagNet/runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors" \
  --best "$RUN/best/gates.safetensors" --latest "$RUN/latest/gates.safetensors" --output "$OUTPUT"
