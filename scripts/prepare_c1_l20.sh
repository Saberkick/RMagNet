#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
ENVIRONMENT="$ROOT/envs/windowseat-py312"
UV_BIN="${UV_BIN:-/home/xuke/.local/bin/uv}"
DATA_ROOT="${C1_DATA_ROOT:-$ROOT/datasets/rmagnet_stage1_512x384}"
OUTPUT="${C1_CACHE_DIR:-$PROJECT/data_cache/c1_l20}"
LOG_DIR="$PROJECT/runs/c1_l20"

if [[ ! -x "$UV_BIN" ]]; then
  echo "uv is not executable: $UV_BIN" >&2
  exit 1
fi
if [[ ! -x "$ENVIRONMENT/bin/python" ]]; then
  echo "Pinned uv environment is missing: $ENVIRONMENT" >&2
  exit 1
fi

export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_OFFLINE=1
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$ROOT/tmp"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$ROOT/tmp" "$UV_CACHE_DIR" "$LOG_DIR"
cd "$PROJECT"

"$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" python -m rmagnet.c1_l20_prepare \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT" \
  --device cuda:0 \
  "$@" 2>&1 | tee "$LOG_DIR/prepare.console.log"
