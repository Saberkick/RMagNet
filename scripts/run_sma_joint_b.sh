#!/usr/bin/env bash
set -euo pipefail
source /share/linmingheng-local/xuke/RMagNet/scripts/sma_env.sh
EPOCHS="${1:-${EPOCHS:-4}}"
export RUN_DIR="${RUN_DIR:-$PROJECT/runs/sma_joint_b_e${EPOCHS}}"
[[ ! -e "$RUN_DIR" ]] || exit 2
bash scripts/train_sma_joint_b.sh "$EPOCHS"
check_gpus
pids=()
for spec in '0 best test' '1 latest test' '2 best real20' '3 latest real20'; do
 read -r gpu choice dataset <<< "$spec"
 checkpoint="$RUN_DIR/${choice}_sma.safetensors"
 if [[ "$dataset" == test ]]; then
   module=src.rmagnet.sma_dataset2_eval
 else
   module=src.rmagnet.sma_real20
 fi
 CUDA_VISIBLE_DEVICES="$gpu" uvpython -m "$module" --checkpoint "$checkpoint" --output "$RUN_DIR/${dataset}_${choice}" > "$RUN_DIR/${dataset}_${choice}.console.log" 2>&1 &
 pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || exit "$status"
uvpython -m src.rmagnet.sma_joint_report --run-dir "$RUN_DIR"
