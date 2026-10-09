#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,3}"
source /share/linmingheng-local/xuke/RMagNet/scripts/sma_env.sh
RUN="$PROJECT/runs/sma_c1_e10"
IFS=',' read -r -a gpus <<< "$CUDA_VISIBLE_DEVICES"
(( ${#gpus[@]} == 3 )) || exit 2
check_gpus
[[ -f "$RUN/best_sma.safetensors" && -f "$RUN/latest_sma.safetensors" ]] || exit 3
run_job() {
 local gpu="$1" mode="$2" choice="$3" folder="$4"
 if [[ -f "$RUN/$folder/evaluation.json" ]]; then echo "Completed: $folder"; return; fi
 [[ ! -e "$RUN/$folder" ]] || { echo "Incomplete output exists: $folder" >&2; return 4; }
 if [[ "$mode" == real20 ]]; then
  CUDA_VISIBLE_DEVICES="$gpu" uvpython -m src.rmagnet.sma_real20 --checkpoint "$RUN/${choice}_sma.safetensors" --output "$RUN/$folder" > "$RUN/$folder.console.log" 2>&1
 else
  local args=(--checkpoint "$RUN/${choice}_sma.safetensors" --output "$RUN/$folder")
  if [[ "$mode" != test ]]; then args+=(--split validation --condition-mode "$mode"); fi
  CUDA_VISIBLE_DEVICES="$gpu" uvpython -m src.rmagnet.sma_dataset2_eval "${args[@]}" > "$RUN/$folder.console.log" 2>&1
 fi
}
wait_batch() { local rc=0; for pid in "$@"; do wait "$pid" || rc=1; done; (( rc == 0 )); }
run_job "${gpus[0]}" test best test_best & p0=$!
run_job "${gpus[1]}" test latest test_latest & p1=$!
run_job "${gpus[2]}" real20 best real20_best & p2=$!
wait_batch "$p0" "$p1" "$p2"
check_gpus
run_job "${gpus[0]}" real20 latest real20_latest & p0=$!
run_job "${gpus[1]}" off best validation_condition_off & p1=$!
run_job "${gpus[2]}" on best validation_condition_on & p2=$!
wait_batch "$p0" "$p1" "$p2"
uvpython -m src.rmagnet.sma_c_report --run-dir "$RUN" --allow-interrupted
