#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
ENVIRONMENT="$ROOT/envs/windowseat-py312"
UV_BIN="${UV_BIN:-/home/xuke/.local/bin/uv}"
CACHE_ROOT="${M2A_CACHE_ROOT:-$PROJECT/data_cache/m2a_q20}"

export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_OFFLINE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT"
exec "$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" \
  python -m rmagnet.m2a_check --cache-root "$CACHE_ROOT" "$@"
