#!/usr/bin/env bash
set -euo pipefail
ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
ENV_DIR="$ROOT/envs/windowseat-py312"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$ROOT/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
cd "$PROJECT"
"$ENV_DIR/bin/python" -m rmagnet.stage3_eval "$@"
