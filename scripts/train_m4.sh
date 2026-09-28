#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
ENVIRONMENT="$ROOT/envs/windowseat-py312"
UV_BIN="${UV_BIN:-/home/xuke/.local/bin/uv}"
CACHE_ROOT="${M4_CACHE_ROOT:-$PROJECT/data_cache/m4_multilayer_v1}"
EPOCHS="${1:-${EPOCHS:-10}}"
MAX_STEPS="${MAX_STEPS:-0}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
RUN_NAME="${RUN_NAME:-m4_multilayer_e${EPOCHS}}"
RUN_DIR="${RUN_DIR:-$PROJECT/runs/$RUN_NAME}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MIN_FREE_MIB="${MIN_FREE_MIB:-22000}"

export LD_LIBRARY_PATH="$ROOT/lib/nvidia-535.179${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$GPUS"
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

if [[ ! "$EPOCHS" =~ ^[1-9][0-9]*$ || ! "$MAX_STEPS" =~ ^[0-9]+$ || ! "$EARLY_STOPPING_PATIENCE" =~ ^[0-9]+$ ]]; then
  echo "EPOCHS must be positive; MAX_STEPS and EARLY_STOPPING_PATIENCE must be non-negative" >&2
  exit 1
fi
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "M4 training requires four visible GPUs; got $GPUS" >&2
  exit 2
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
if [[ ! -f "$CACHE_ROOT/manifest.json" ]]; then
  echo "M4 cache is not finalized: $CACHE_ROOT/manifest.json" >&2
  exit 4
fi
if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Run directory is non-empty: $RUN_DIR" >&2
  exit 5
fi

cd "$PROJECT"
COMMON_ARGS=(
  --cache-root "$CACHE_ROOT"
  --run-dir "$RUN_DIR"
  --epochs "$EPOCHS"
  --max-steps "$MAX_STEPS"
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
  --batch-size 1
  --gradient-accumulation 1
  --learning-rate 5e-6
  --warmup-steps 20
  --gradient-ramp-steps 36
  --spatial-gradient-ratio 0.08
  --texture-gradient-ratio 0.08
  --semantic-gradient-ratio 0.08
  --aux-gradient-cap 0.25
  --consistency-coefficient 0.10
  --ssim-weight 0.2
  --edge-weight 0.1
  --num-workers 1
)

"$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" \
  python -m src.rmagnet.m4_train "${COMMON_ARGS[@]}" --preflight-only

exec "$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" \
  python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  -m src.rmagnet.m4_train "${COMMON_ARGS[@]}"
