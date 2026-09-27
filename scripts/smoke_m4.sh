#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
RUN_NAME="${RUN_NAME:-m4_smoke_2steps}"
RUN_DIR="${RUN_DIR:-$PROJECT/runs/$RUN_NAME}"

if [[ -e "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Smoke run directory is non-empty: $RUN_DIR" >&2
  exit 1
fi

EPOCHS=1 \
MAX_STEPS=2 \
RUN_NAME="$RUN_NAME" \
RUN_DIR="$RUN_DIR" \
bash "$PROJECT/scripts/train_m4.sh" 1
