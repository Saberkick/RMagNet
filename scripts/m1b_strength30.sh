#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
PYTHON="$ROOT/envs/windowseat-py312/bin/python"
GROUP=strength30_e2
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$ROOT/tmp"
cd "$PROJECT"

case "${1:-}" in
  prepare)
    "$PYTHON" -m rmagnet.m1b_strength30 prepare
    ;;
  base|dolp|shuffle)
    ARM="$1"
    if [[ "$ARM" == base ]]; then
      WEIGHT=0.0
    else
      WEIGHT=$("$PYTHON" -m rmagnet.m1b_strength30 weight)
    fi
    EPOCHS=2 RUN_GROUP="$GROUP" SEM_WEIGHT="$WEIGHT" \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}" \
      bash scripts/run_m1b.sh "$ARM"
    ;;
  eval)
    RUN_GROUP="$GROUP" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" \
      bash scripts/eval_m1b.sh
    "$PYTHON" -m rmagnet.m1b_strength30 compare
    ;;
  *)
    echo "usage: bash scripts/m1b_strength30.sh {prepare|base|dolp|shuffle|eval}" >&2
    exit 2
    ;;
esac
