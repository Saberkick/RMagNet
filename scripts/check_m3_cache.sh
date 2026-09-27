#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/linmingheng-local/xuke"
PROJECT="$ROOT/RMagNet"
PYTHON="$ROOT/envs/windowseat-py312/bin/python"
NVIDIA_USER_LIB="$ROOT/lib/nvidia-535.179"
OUTPUT="${OUTPUT:-$PROJECT/data_cache/m3_semantic_v1}"

export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$NVIDIA_USER_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$PROJECT"
"$PYTHON" -m src.rmagnet.m3_cache --output "$OUTPUT" check
