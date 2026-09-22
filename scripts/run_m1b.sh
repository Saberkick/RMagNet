#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/run_m1b.sh {base|dolp|shuffle} [extra m1b_train args]" >&2
  exit 2
fi
ARM="$1"
shift
case "$ARM" in base|dolp|shuffle) ;; *) echo "bad arm: $ARM" >&2; exit 2 ;; esac
ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
ENV_DIR="$ROOT/envs/windowseat-py312"
EPOCHS="${EPOCHS:-8}"
RUN_GROUP="${RUN_GROUP:-primary_e${EPOCHS}}"
RUN_DIR="$PROJECT/runs/m1b/$RUN_GROUP/$ARM"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-2,3}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$ROOT/tmp"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
if [[ "$ARM" != base && -z "${SEM_WEIGHT:-}" ]]; then
  CALIBRATION="$PROJECT/runs/m1b/sem_calibration.json"
  if [[ ! -f "$CALIBRATION" ]]; then
    echo "Missing $CALIBRATION; run bash scripts/calibrate_m1b.sh first" >&2
    exit 1
  fi
  SEM_WEIGHT=$("$ENV_DIR/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["sem_weight"])' "$CALIBRATION")
fi
mkdir -p "$PROJECT/runs/m1b/$RUN_GROUP"
cd "$PROJECT"
"$ENV_DIR/bin/python" -m rmagnet.m1b_train \
  --arm "$ARM" --epochs "$EPOCHS" --run-dir "$RUN_DIR" \
  --sem-weight "${SEM_WEIGHT:-0.0}" \
  --learning-rate "${LEARNING_RATE:-5e-6}" \
  "$@" 2>&1 | tee "$PROJECT/runs/m1b/$RUN_GROUP/${ARM}.console.log"
