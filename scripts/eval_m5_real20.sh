#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "$(dirname "$0")/sma_env.sh"
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || exit 2
MIN_FREE_MIB=20000 check_gpus
uvpython -m src.rmagnet.m5_real20
