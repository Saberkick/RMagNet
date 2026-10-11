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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
IFS=, read -r -a cards <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#cards[@]} -eq 2 && "${cards[0]}" != "${cards[1]}" ]] || exit 2
for card in "${cards[@]}"; do
  [[ "$card" =~ ^[0-9]+$ ]] || exit 2
  free="$(nvidia-smi -i "$card" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( free>=22000 )) || { echo "GPU $card is busy"; exit 3; }
done
cd "$PROJECT"
OUTPUT="${OUTPUT:-$PROJECT/runs/m7_m4_latent_tail_e5/sealed_test_best}"
/home/xuke/.local/bin/uv run --no-project --python "$ROOT/envs/windowseat-py312/bin/python" \
  python -m src.rmagnet.m7_sealed_eval \
  --data "$ROOT/datasets/rmagnet_sma_dataset3" \
  --initial "$ROOT/RMagNet/runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors" \
  --tail-cache "$PROJECT/data_cache/m7_m4_tail_v1" \
  --checkpoint "$PROJECT/runs/m7_m4_latent_tail_e5/best/model.safetensors" \
  --output "$OUTPUT"
