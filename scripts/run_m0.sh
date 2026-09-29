#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
ENVIRONMENT="$ROOT/envs/windowseat-py312"
UV_BIN="${UV_BIN:-/home/xuke/.local/bin/uv}"
DATA_ROOT="${M0_DATA_ROOT:-$ROOT/datasets/rmagnet_m2_aspect}"
RUN_DIR="${RUN_DIR:-$PROJECT/runs/M0_windowseat_m2_e18}"
INITIAL="${INITIAL:-$ROOT/.cache/huggingface/hub/models--huawei-bayerlab--windowseat-reflection-removal-v1-0/snapshots/c1f59ca02bff68535c976e5e17147b3d9323309e/transformer_lora/pytorch_lora_weights.safetensors}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MIN_FREE_MIB="${M0_TRAIN_MIN_FREE_MIB:-22000}"
EPOCHS="${EPOCHS:-18}"
UPDATES_PER_EPOCH=36
MAX_STEPS=$((EPOCHS * UPDATES_PER_EPOCH))

[[ -x "$UV_BIN" && -x "$ENVIRONMENT/bin/python" ]] || { echo "Pinned uv runtime is unavailable" >&2; exit 1; }
[[ -f "$DATA_ROOT/manifest.json" && -f "$INITIAL" ]] || { echo "M2 data or official WindowSeat LoRA is missing" >&2; exit 1; }
[[ "$EPOCHS" -eq 18 ]] || { echo "M0 is fixed to 18 epochs" >&2; exit 1; }
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
[[ "${#GPUS[@]}" -eq 4 ]] || { echo "M0 requires exactly four GPUs" >&2; exit 1; }
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU index: $gpu" >&2; exit 1; }
  free_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( free_mib >= MIN_FREE_MIB )) || { echo "GPU $gpu has only ${free_mib} MiB free" >&2; exit 2; }
done
if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty run directory: $RUN_DIR" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export UV_CACHE_DIR="$ROOT/.cache/uv" UV_OFFLINE=1
export HF_HOME="$ROOT/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$ROOT/.cache/torch" TMPDIR="$ROOT/tmp"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.80"

mkdir -p "$ROOT/tmp" "$UV_CACHE_DIR"
cd "$PROJECT"
exec "$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" \
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m rmagnet.m0_train \
  --data-root "$DATA_ROOT" --run-dir "$RUN_DIR" --initial "$INITIAL" \
  --epochs "$EPOCHS" --max-steps "$MAX_STEPS" \
  --batch-size 1 --gradient-accumulation 1 \
  --learning-rate 5e-6 --weight-decay 0.01 --warmup-steps 32 \
  --ssim-weight 0.2 --edge-weight 0.1 --max-grad-norm 1.0 \
  --validate-every "$UPDATES_PER_EPOCH" --save-every "$UPDATES_PER_EPOCH" \
  --num-workers "${NUM_WORKERS:-1}" "$@"
