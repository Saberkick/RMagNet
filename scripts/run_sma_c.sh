#!/usr/bin/env bash
set -euo pipefail
source /share/linmingheng-local/xuke/RMagNet/scripts/sma_env.sh
EPOCHS="${1:-${EPOCHS:-10}}"
export RUN_DIR="${RUN_DIR:-$PROJECT/runs/sma_c1_e${EPOCHS}}"
bash scripts/train_sma_c.sh "$EPOCHS"
# Evaluate only after the full requested budget; smoke never enters this script.
check_gpus
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
CUDA_VISIBLE_DEVICES="${gpus[0]}" uvpython -m src.rmagnet.sma_dataset2_eval --checkpoint "$RUN_DIR/best_sma.safetensors" --output "$RUN_DIR/test_best" > "$RUN_DIR/test_best.console.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES="${gpus[1]}" uvpython -m src.rmagnet.sma_dataset2_eval --checkpoint "$RUN_DIR/latest_sma.safetensors" --output "$RUN_DIR/test_latest" > "$RUN_DIR/test_latest.console.log" 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES="${gpus[2]}" uvpython -m src.rmagnet.sma_real20 --checkpoint "$RUN_DIR/best_sma.safetensors" --output "$RUN_DIR/real20_best" > "$RUN_DIR/real20_best.console.log" 2>&1 & p2=$!
CUDA_VISIBLE_DEVICES="${gpus[3]}" uvpython -m src.rmagnet.sma_real20 --checkpoint "$RUN_DIR/latest_sma.safetensors" --output "$RUN_DIR/real20_latest" > "$RUN_DIR/real20_latest.console.log" 2>&1 & p3=$!
rc=0
for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid" || rc=1; done
(( rc == 0 )) || exit 1
# Same trained model, validation-only condition interventions (not trained C0).
CUDA_VISIBLE_DEVICES="${gpus[0]}" uvpython -m src.rmagnet.sma_dataset2_eval --checkpoint "$RUN_DIR/best_sma.safetensors" --split validation --condition-mode off --output "$RUN_DIR/validation_condition_off" > "$RUN_DIR/validation_condition_off.console.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES="${gpus[1]}" uvpython -m src.rmagnet.sma_dataset2_eval --checkpoint "$RUN_DIR/best_sma.safetensors" --split validation --condition-mode on --output "$RUN_DIR/validation_condition_on" > "$RUN_DIR/validation_condition_on.console.log" 2>&1 & p1=$!
rc=0; for pid in "$p0" "$p1"; do wait "$pid" || rc=1; done
(( rc == 0 )) || exit 1
uvpython -m src.rmagnet.sma_c_report --run-dir "$RUN_DIR"
