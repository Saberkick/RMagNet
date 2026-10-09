#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2
source "$(dirname "${BASH_SOURCE[0]}")/sma_env.sh"
check_gpus
run="$PROJECT/runs/sma_dataset2_e50"
(( $(python3 -c 'import json; print(json.load(open("runs/sma_dataset2_e50/training_summary.json"))["epochs_completed"])') == 50 ))
pids=()
CUDA_VISIBLE_DEVICES=0 uvpython -m src.rmagnet.sma_dataset2_eval --checkpoint "$run/best_sma.safetensors" --output "$run/test_best" > "$run/test_best.console.log" 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=1 uvpython -m src.rmagnet.sma_dataset2_eval --checkpoint "$run/latest_sma.safetensors" --output "$run/test_latest" > "$run/test_latest.console.log" 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=2 uvpython -m src.rmagnet.sma_dataset2_eval --output "$run/test_m4best" > "$run/test_m4best.console.log" 2>&1 &
pids+=("$!")
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
