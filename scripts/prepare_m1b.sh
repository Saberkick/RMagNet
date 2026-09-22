#!/usr/bin/env bash
set -euo pipefail
ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/.cache/huggingface"
export TMPDIR="$ROOT/tmp"
cd "$PROJECT"
"$ROOT/envs/windowseat-py312/bin/python" -m rmagnet.m1b_prepare \
  --data-root "$ROOT/datasets/rmagnet_stage1_512x384" \
  --threshold "${DOLP_THRESHOLD:-64}"
