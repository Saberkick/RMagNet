#!/usr/bin/env bash
set -euo pipefail
ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
ENVIRONMENT="$ROOT/envs/windowseat-py312"
UV_BIN="${UV_BIN:-/home/xuke/.local/bin/uv}"
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
mkdir -p "$ROOT/tmp" "$PROJECT/runs/c1_l20"
cd "$PROJECT"
"$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" \
  torchrun --standalone --nproc-per-node=4 -m rmagnet.c1_l20_repair_gt \
  --restart-staging 2>&1 | tee "$PROJECT/runs/c1_l20/repair_gt_cache.console.log"
"$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" \
  python -m rmagnet.c1_l20_cache_check
