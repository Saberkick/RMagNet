#!/usr/bin/env bash
set -euo pipefail
SINGLE_GPU="${CUDA_VISIBLE_DEVICES:-2}"
source "$(dirname "$0")/sma_env.sh"
export CUDA_VISIBLE_DEVICES="$SINGLE_GPU" OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
[[ "$SINGLE_GPU" =~ ^[0-9]+$ ]] || { echo 'Select exactly one GPU' >&2; exit 2; }
export PHYSICAL_GPU_INDEX="$SINGLE_GPU"
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$SINGLE_GPU" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"
check_single_gpu() { CUDA_VISIBLE_DEVICES="$SINGLE_GPU" check_gpus; }
C1_EPOCHS="${C1_EPOCHS:-${EPOCHS:-20}}"
M5_EPOCHS="${M5_EPOCHS:-${EPOCHS:-20}}"
RUN_GROUP="${RUN_GROUP:-ws_c1_m5_dataset3_s1_e20}"
STAGES="${STAGES:-all}"
[[ "$RUN_GROUP" =~ ^[A-Za-z0-9_-]+$ && "$C1_EPOCHS" =~ ^[1-9][0-9]*$ && "$M5_EPOCHS" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$STAGES" == all || "$STAGES" == c1 || "$STAGES" == m5 ]] || exit 2
GROUP="$PROJECT/runs/$RUN_GROUP"
DATA="$ROOT/datasets/rmagnet_sma_dataset3"
CACHE="$PROJECT/data_cache/${RUN_GROUP}_layers"
MEMORY="$GROUP/memory_prepare"
RGB="$PROJECT/data_cache/${RUN_GROUP}_rgb8"
INITIAL="$(uvpython -c "from src.rmagnet.qwen_backend import check_snapshots;print(check_snapshots()[1]/'transformer_lora/pytorch_lora_weights.safetensors')")"
EXPECTED_INITIAL=2f04225c5be1cac69a5d76faf4c95515d5503210cfee73c0bb1e79309f5a74e9
mkdir -p "$GROUP/logs"
exec 9>"$GROUP/pipeline.lock"
flock -n 9 || { echo 'This pipeline is active' >&2; exit 3; }
printf '%s\n' "$$" > "$GROUP/pipeline.pid"
phase() {
  uvpython - "$GROUP" "$1" "$C1_EPOCHS" "$M5_EPOCHS" "$SINGLE_GPU" <<'PY'
import json,sys,datetime
from pathlib import Path
from src.rmagnet.resume_bundle import atomic_json
j={'phase':sys.argv[2],'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'c1_epochs':int(sys.argv[3]),'m5_epochs':int(sys.argv[4]),'gpu':sys.argv[5],'early_stopping':False,'resumable':True,'retention':'best and latest bundles'}
root=Path(sys.argv[1]);atomic_json(root/'status.json',j)
with (root/'pipeline_events.jsonl').open('a') as f:f.write(json.dumps(j)+'\n')
PY
  echo "PHASE $1"
}
finish() {
  local rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$GROUP/pipeline.exit_code"
  if (( rc == 130 )); then phase paused || true
  elif (( rc != 0 )); then phase failed || true; fi
  exit "$rc"
}
trap finish EXIT
check_single_gpu
available="$(df --output=avail -B1 "$ROOT" | tail -1 | tr -d ' ')"
(( available >= 19327352832 )) || { echo 'Need 18GiB available for resumable atomic checkpoint replacements' >&2; exit 6; }
if [[ "$STAGES" != m5 ]]; then
  phase prepare
  CUDA_VISIBLE_DEVICES="$SINGLE_GPU" RUN_GROUP="$RUN_GROUP" bash scripts/prepare_windowseat_dataset3_single.sh >> "$GROUP/logs/prepare.log" 2>&1
  check_single_gpu
  phase c1_training
  resume=()
  [[ ! -d "$GROUP/c1/latest" && ! -d "$GROUP/c1/.previous_latest" ]] || resume=(--resume)
  uvpython -m src.rmagnet.windowseat_c1_single --data-root "$DATA" --cache-root "$CACHE" \
    --memory-file "$MEMORY/memory.safetensors" --initial "$INITIAL" --run-dir "$GROUP/c1" \
    --epochs "$C1_EPOCHS" --accumulation 4 --save-every "${SAVE_EVERY:-10}" "${resume[@]}" >> "$GROUP/logs/c1.log" 2>&1
fi
if [[ "$STAGES" == c1 ]]; then phase c1_complete; exit 0; fi
[[ -f "$GROUP/c1/best/model.safetensors" ]] || { echo 'C1 best missing' >&2; exit 7; }
check_single_gpu
phase c1_rgb_cache
uvpython -m src.rmagnet.m5_cache --data "$DATA" --checkpoint "$GROUP/c1/best/model.safetensors" \
  --initial "$INITIAL" --cache "$RGB" >> "$GROUP/logs/rgb_cache.log" 2>&1
MIN_FREE_MIB=4096 check_single_gpu
phase m5_training
resume=()
[[ ! -d "$GROUP/m5/latest" && ! -d "$GROUP/m5/.previous_latest" ]] || resume=(--resume)
uvpython -m src.rmagnet.windowseat_m5_single --data "$DATA" --cache "$RGB" \
  --upstream-checkpoint "$GROUP/c1/best/model.safetensors" --output "$GROUP/m5" \
  --epochs "$M5_EPOCHS" --accumulation 4 --save-every "${SAVE_EVERY:-10}" "${resume[@]}" >> "$GROUP/logs/m5.log" 2>&1
phase complete
echo 'C1+M5 complete; no sealed-test or real20 evaluation during training'
