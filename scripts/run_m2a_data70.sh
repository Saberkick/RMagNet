#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
ENVIRONMENT="$ROOT/envs/windowseat-py312"
UV_BIN="${UV_BIN:-/home/xuke/.local/bin/uv}"
DATA_ROOT="${M2_DATA_ROOT:-$ROOT/datasets/rmagnet_m2_aspect}"
RUN_DIR="${RUN_DIR:-$PROJECT/runs/m2_corrected_a_data70_noq20}"
INITIAL="${INITIAL:-$PROJECT/runs/stage2_transmission_r128/best_transmission_lora.safetensors}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MIN_FREE_MIB="${M2_TRAIN_MIN_FREE_MIB:-22000}"

if [[ ! -x "$UV_BIN" || ! -x "$ENVIRONMENT/bin/python" ]]; then
  echo "Pinned uv runtime is unavailable" >&2
  exit 1
fi
if [[ ! -f "$DATA_ROOT/manifest.json" || ! -f "$INITIAL" ]]; then
  echo "M2 data or Stage-2 initialization is missing" >&2
  exit 1
fi
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "M2-A strict comparison requires exactly four GPUs; got $GPU_LIST" >&2
  exit 1
fi
for gpu in "${GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Invalid physical GPU index: $gpu" >&2
    exit 1
  fi
  free_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if (( free_mib < MIN_FREE_MIB )); then
    echo "GPU $gpu has only ${free_mib} MiB free; require ${MIN_FREE_MIB} MiB" >&2
    exit 2
  fi
done
if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to mix with non-empty run directory: $RUN_DIR" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_OFFLINE=1
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$ROOT/.cache/torch"
export TMPDIR="$ROOT/tmp"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.80"

mkdir -p "$ROOT/tmp" "$UV_CACHE_DIR"
cd "$PROJECT"

exec "$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" \
  python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  -m rmagnet.m2a_data_baseline \
  --data-root "$DATA_ROOT" \
  --run-dir "$RUN_DIR" \
  --initial "$INITIAL" \
  --epochs 10 \
  --max-steps 70 \
  --batch-size 1 \
  --gradient-accumulation 1 \
  --learning-rate 5e-6 \
  --weight-decay 0.01 \
  --warmup-steps 20 \
  --ssim-weight 0.2 \
  --edge-weight 0.1 \
  --max-grad-norm 1.0 \
  --validate-every 35 \
  --save-every 35 \
  --num-workers "${NUM_WORKERS:-1}" \
  "$@"
