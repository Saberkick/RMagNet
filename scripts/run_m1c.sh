#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
PYTHON="$ROOT/envs/windowseat-py312/bin/python"
RUN_ROOT="$PROJECT/runs/m1c/probe_e2"
MAPS="$PROJECT/runs/m1c/softmaps"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$ROOT/tmp"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/tmp" "$PROJECT/runs/m1c" "$RUN_ROOT"
cd "$PROJECT"

prepare() {
  if [[ ! -f "$MAPS/manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES="${M1C_MODEL_GPU:-2}" "$PYTHON" -m rmagnet.m1c_softmap --output "$MAPS"
  fi
  if [[ ! -f "$PROJECT/runs/m1c/calibration.json" ]]; then
    CUDA_VISIBLE_DEVICES="${M1C_GPUS:-2,3}" "$PYTHON" -m rmagnet.m1c_train --mode calibrate
  fi
}

train_arm() {
  local arm="$1"
  if [[ -e "$RUN_ROOT/$arm/DONE.json" ]]; then
    echo "M1c $arm already complete; refusing to rerun" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="${M1C_GPUS:-2,3}" "$PYTHON" -m rmagnet.m1c_train \
    --mode train --arm "$arm" --epochs 2 --max-steps 100 \
    --run-dir "$RUN_ROOT/$arm" 2>&1 | tee "$RUN_ROOT/${arm}.console.log"
}

evaluate() {
  CUDA_VISIBLE_DEVICES="${M1C_MODEL_GPU:-2}" "$PYTHON" -m rmagnet.m1c_eval \
    --run-root "$RUN_ROOT" --output-dir "$RUN_ROOT/evaluation" \
    2>&1 | tee "$RUN_ROOT/evaluation.console.log"
}

case "${1:-}" in
  prepare) prepare ;;
  base|soft|shift) train_arm "$1" ;;
  eval) evaluate ;;
  all)
    prepare
    train_arm base
    train_arm soft
    train_arm shift
    evaluate
    ;;
  *) echo "usage: bash scripts/run_m1c.sh {prepare|base|soft|shift|eval|all}" >&2; exit 2 ;;
esac
