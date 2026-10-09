#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "$(dirname "$0")/sma_env.sh"
check_gpus
uvpython -m src.rmagnet.m5_cache "$@"
