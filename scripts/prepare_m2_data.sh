#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
ENVIRONMENT="$ROOT/envs/windowseat-py312"
UV_BIN="${UV_BIN:-/home/xuke/.local/bin/uv}"
ARCHIVE="${M2_ARCHIVE:-$ROOT/datasets/rmagnet_m2_source/data_set.zip}"
OUTPUT="${M2_DATA_ROOT:-$ROOT/datasets/rmagnet_m2_aspect}"

if [[ ! -x "$UV_BIN" ]]; then
  echo "uv is not executable: $UV_BIN" >&2
  exit 1
fi
if [[ ! -x "$ENVIRONMENT/bin/python" ]]; then
  echo "Pinned uv environment is missing: $ENVIRONMENT" >&2
  exit 1
fi
if [[ ! -f "$ARCHIVE" ]]; then
  echo "M2 source archive is missing: $ARCHIVE" >&2
  exit 1
fi
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT" >&2
  exit 1
fi

export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_OFFLINE=1
export TMPDIR="$ROOT/tmp"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

mkdir -p "$ROOT/tmp" "$UV_CACHE_DIR" "$(dirname "$OUTPUT")"
cd "$PROJECT"

exec "$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" \
  python -m rmagnet.m2_prepare_data \
  --archive "$ARCHIVE" \
  --output "$OUTPUT" \
  --target-pixels "${M2_TARGET_PIXELS:-196608}" \
  --multiple "${M2_SIZE_MULTIPLE:-16}" \
  --seed "${M2_SPLIT_SEED:-2026}" \
  "$@"
