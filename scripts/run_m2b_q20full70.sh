#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
ENVIRONMENT="$ROOT/envs/windowseat-py312"
UV_BIN="${UV_BIN:-/home/xuke/.local/bin/uv}"
RUN_DIR="${RUN_DIR:-$PROJECT/runs/m2_corrected_b_q20full70}"

export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_OFFLINE=1
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$ROOT/tmp"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")" -ne 4 ]]; then
  echo "M2-B requires exactly four visible GPUs" >&2
  exit 1
fi
if [[ ! -x "$UV_BIN" || ! -x "$ENVIRONMENT/bin/python" ]]; then
  echo "Pinned uv executable or Python environment is missing" >&2
  exit 1
fi
if [[ -d "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "Run directory is not empty: $RUN_DIR" >&2
  exit 1
fi

IFS=',' read -r -a gpu_ids <<<"$CUDA_VISIBLE_DEVICES"
for gpu in "${gpu_ids[@]}"; do
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
  if (( free_mib < 22000 )); then
    echo "GPU $gpu has only ${free_mib} MiB free; need at least 22000 MiB" >&2
    exit 1
  fi
done

mkdir -p "$ROOT/tmp" "$UV_CACHE_DIR"
cd "$PROJECT"
"$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" \
  python -m rmagnet.m2b_q20 --preflight-only

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi

exec "$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" \
  torchrun --standalone --nproc-per-node=4 -m rmagnet.m2b_q20 \
  --epochs 10 --max-steps 70 --batch-size 1 --gradient-accumulation 1 \
  --learning-rate 5e-6 --warmup-steps 20 \
  --local-coefficient 0.25 --keep-coefficient 0.10 \
  --gradient-measure-every 20 --validate-every 35 --save-every 35 \
  --num-workers "${NUM_WORKERS:-1}"
