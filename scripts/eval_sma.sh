#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/sma_env.sh"
CHECKPOINT="${1:-$PROJECT/runs/sma_e20/best_sma.safetensors}"
OUTPUT="${2:-$PROJECT/runs/sma_e20/test_best}"
uvpython -m src.rmagnet.sma_eval --checkpoint "$CHECKPOINT" --output "$OUTPUT" --split "${SPLIT:-test}"
