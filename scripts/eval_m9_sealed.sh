#!/usr/bin/env bash
set -euo pipefail
ROOT=/share/linmingheng-local/xuke
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT/lib/nvidia-535.179${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$ROOT/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$ROOT/.cache/uv" UV_OFFLINE=1 TMPDIR="$ROOT/tmp"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJECT"
RUN="$PROJECT/runs/m9_m4_latent_only_e4"
for gpu in ${CUDA_VISIBLE_DEVICES//,/ }; do
 free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
 (( free>=22000 )) || { echo "Insufficient free GPU memory: $gpu $free MiB"; exit 2; }
done
test "$(cat "$RUN/pipeline.exit_code")" = 0
/home/xuke/.local/bin/uv run --no-project --python "$ROOT/envs/windowseat-py312/bin/python" \
 python -m src.rmagnet.m9_sealed_eval --data "$ROOT/datasets/rmagnet_sma_dataset3" \
 --initial "$ROOT/RMagNet/runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors" \
 --run "$RUN" --output "$PROJECT/results_archive/M9/m9_latent_only_e4_sealed_test"
