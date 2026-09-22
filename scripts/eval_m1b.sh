#!/usr/bin/env bash
set -euo pipefail
ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
RUN_GROUP="${RUN_GROUP:-primary_e8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$ROOT/tmp"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT"
"$ROOT/envs/windowseat-py312/bin/python" -m rmagnet.m1b_eval \
  --run-root "$PROJECT/runs/m1b/$RUN_GROUP" \
  --output-dir "$PROJECT/runs/m1b/$RUN_GROUP/evaluation" "$@"
