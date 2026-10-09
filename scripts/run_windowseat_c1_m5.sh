#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/sma_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
C1_EPOCHS="${C1_EPOCHS:-50}"
M5_EPOCHS="${M5_EPOCHS:-50}"
MEMORY_EPOCHS="${MEMORY_EPOCHS:-5}"
RUN_GROUP="${RUN_GROUP:-ws_c1_m5_filtered_e50}"
[[ "$RUN_GROUP" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
for count in "$C1_EPOCHS" "$M5_EPOCHS" "$MEMORY_EPOCHS"; do
  [[ "$count" =~ ^[1-9][0-9]*$ ]] || exit 2
done
DATA_ROOT="$ROOT/datasets/rmagnet_sma_dataset2"
GROUP_ROOT="$PROJECT/runs/$RUN_GROUP"
CACHE_ROOT="$PROJECT/data_cache/${RUN_GROUP}_layers"
RGB_CACHE="$PROJECT/data_cache/${RUN_GROUP}_rgb8"
MEMORY_DIR="$GROUP_ROOT/memory_prepare"
C1_RUN="$GROUP_ROOT/c1"
M5_RUN="$GROUP_ROOT/m5"
INITIAL="$ROOT/.cache/huggingface/hub/models--huawei-bayerlab--windowseat-reflection-removal-v1-0/snapshots/c1f59ca02bff68535c976e55e17147b3d9323309e/transformer_lora/pytorch_lora_weights.safetensors"
EXPECTED_INITIAL=2f04225c5be1cac69a5d76faf4c95515d5503210cfee73c0bb1e79309f5a74e9
mkdir -p "$GROUP_ROOT/logs" "$CACHE_ROOT/logs"
exec 9>"$GROUP_ROOT/pipeline.lock"
flock -n 9 || { echo 'Pipeline already active' >&2; exit 3; }
[[ ! -e "$GROUP_ROOT/pipeline.exit_code" ]] || { echo 'Previous pipeline ended; inspect status and use a new RUN_GROUP' >&2; exit 4; }
printf '%s\n' "$$" > "$GROUP_ROOT/pipeline.pid"
phase() {
  local name="$1"
  uvpython - "$GROUP_ROOT" "$name" "$C1_EPOCHS" "$M5_EPOCHS" <<'PY'
import json,sys,datetime
from pathlib import Path
p=Path(sys.argv[1])/'status.json'
j={'phase':sys.argv[2],'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'c1_epochs':int(sys.argv[3]),'m5_epochs':int(sys.argv[4]),'early_stopping':False,'retention':'best/latest only for each trainable model'}
t=p.with_suffix('.tmp');t.write_text(json.dumps(j,indent=2)+'\n');t.replace(p)
PY
  echo "PHASE $name"
}
finish() {
  local rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$GROUP_ROOT/pipeline.exit_code"
  if (( rc != 0 )); then phase failed || true; fi
  exit "$rc"
}
trap finish EXIT
IFS=',' read -r -a cards <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#cards[@]} == 4 ]] || { echo 'Select exactly four GPUs' >&2; exit 5; }
CACHE_GPU_DEVICES="${CACHE_GPU_DEVICES:-$CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a cache_cards <<< "$CACHE_GPU_DEVICES"
[[ ${#cache_cards[@]} -ge 1 && ${#cache_cards[@]} -le 4 ]] || exit 5
for gpu in "${cards[@]}" "${cache_cards[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || exit 5
done
[[ "$(sha256sum "$INITIAL" | cut -d ' ' -f1)" == "$EXPECTED_INITIAL" ]] || exit 5
wait_gpus() {
  local selection="$1" minimum="$2" label="$3"
  local busy free_mib gpu
  local -a selected
  local -A seen=()
  IFS=',' read -r -a selected <<< "$selection"
  for gpu in "${selected[@]}"; do
    [[ -z "${seen[$gpu]:-}" ]] || { echo 'Duplicate GPU' >&2; return 5; }
    seen[$gpu]=1
  done
  while true; do
    busy=0
    for gpu in "${selected[@]}"; do
      free_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
      [[ "$free_mib" =~ ^[0-9]+$ ]] || return 5
      (( free_mib >= minimum )) || busy=1
    done
    (( busy == 1 )) || break
    phase "waiting_${label}_gpus"
    echo "Waiting for GPUs $selection: require $minimum MiB free on each; next check in 300s"
    sleep 300
  done
}
available="$(df --output=avail -B1 "$ROOT" | tail -1 | tr -d ' ')"
(( available >= 21474836480 )) || { echo 'Require 20GiB free for layer cache, two C1 checkpoints and atomic saves' >&2; exit 6; }
uvpython -m src.rmagnet.windowseat_c1_audit --data-root "$DATA_ROOT" --output "$GROUP_ROOT/preflight_data.json"
wait_gpus "$CACHE_GPU_DEVICES" 22000 cache
phase layer_cache
pids=()
for (( shard=0; shard<${#cache_cards[@]}; shard++ )); do
  CUDA_VISIBLE_DEVICES="${cache_cards[$shard]}" uvpython -m src.rmagnet.sma_cache extract \
    --data-root "$DATA_ROOT" --output "$CACHE_ROOT" --adapter "$INITIAL" \
    --adapter-description 'Official WindowSeat rank128 adapter enabled; frozen teacher; no M4/Stage2 continuation' \
    --device cuda:0 --shard-index "$shard" --num-shards "${#cache_cards[@]}" \
    > "$CACHE_ROOT/logs/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
(( rc == 0 )) || { echo 'Cache extraction failed; no training started' >&2; exit 7; }
uvpython -m src.rmagnet.sma_cache finalize --data-root "$DATA_ROOT" --output "$CACHE_ROOT" --adapter "$INITIAL" \
  --adapter-description 'Official WindowSeat rank128 adapter enabled; frozen teacher; no M4/Stage2 continuation'
wait_gpus "${cache_cards[0]}" 8192 memory
phase memory_prepare
CUDA_VISIBLE_DEVICES="${cache_cards[0]}" uvpython -m src.rmagnet.sma_pretrain \
  --cache-root "$CACHE_ROOT" --output "$MEMORY_DIR" --epochs "$MEMORY_EPOCHS" \
  > "$GROUP_ROOT/logs/memory.log" 2>&1
phase c1_preflight
args=(--joint-lora --data-root "$DATA_ROOT" --cache-root "$CACHE_ROOT"
  --initial "$INITIAL" --initial-sha256 "$EXPECTED_INITIAL" --initial-family 'WindowSeat official'
  --memory-file "$MEMORY_DIR/memory.safetensors" --run-dir "$C1_RUN"
  --epochs "$C1_EPOCHS" --max-steps 0 --learning-rate 1e-4 --lora-learning-rate 5e-6
  --warmup-steps 20 --gradient-ramp-steps 36 --spatial-gradient-ratio .08
  --texture-gradient-ratio .08 --semantic-gradient-ratio .08 --aux-gradient-cap .25
  --consistency-coefficient .10 --ssim-weight .2 --edge-weight .1
  --early-stopping-patience 0 --num-workers "${NUM_WORKERS:-1}"
  --calibration-start-epoch 1 --calibration-interval 8 --cpu-offload-activations)
uvpython -m src.rmagnet.sma_c_train "${args[@]}" --preflight-only > "$GROUP_ROOT/logs/c1_preflight.json"
wait_gpus "$CUDA_VISIBLE_DEVICES" 22000 c1
check_gpus
phase c1_training
"$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" python -m torch.distributed.run \
  --standalone --nproc_per_node=4 -m src.rmagnet.sma_c_train "${args[@]}" > "$GROUP_ROOT/logs/c1.log" 2>&1
uvpython - "$C1_RUN" "$C1_EPOCHS" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);j=json.loads((p/'training_summary.json').read_text())
assert j.get('status')=='complete',j
assert not j.get('early_stopped'),j
c=json.loads((p/'run_config.json').read_text());m=json.loads((p/'latest_metrics.json').read_text())
assert m['step']==int(sys.argv[2])*c['updates_per_epoch'],m
assert {x.name for x in p.glob('*.safetensors')}=={'best_sma.safetensors','latest_sma.safetensors'}
PY
wait_gpus "${cards[0]}" 22000 rgb_cache
phase c1_rgb_cache
CUDA_VISIBLE_DEVICES="${cards[0]}" uvpython -m src.rmagnet.m5_cache \
  --data "$DATA_ROOT" --checkpoint "$C1_RUN/best_sma.safetensors" --initial "$INITIAL" --cache "$RGB_CACHE" \
  > "$GROUP_ROOT/logs/rgb_cache.log" 2>&1
wait_gpus "${cards[0]}" 4096 m5
phase m5_training
CUDA_VISIBLE_DEVICES="${cards[0]}" uvpython -m src.rmagnet.m5_train \
  --data "$DATA_ROOT" --cache "$RGB_CACHE" --upstream-checkpoint "$C1_RUN/best_sma.safetensors" \
  --output "$M5_RUN" --epochs "$M5_EPOCHS" --patience 0 --accumulation 4 --lr 2e-4 \
  > "$GROUP_ROOT/logs/m5.log" 2>&1
uvpython - "$M5_RUN" "$M5_EPOCHS" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);j=json.loads((p/'training_summary.json').read_text())
assert j['completed_epochs']==int(sys.argv[2]) and not j['early_stopped'],j
assert {x.name for x in p.glob('*.safetensors')}=={'best.safetensors','latest.safetensors'}
PY
phase complete
echo 'C1 and M5 completed. No sealed-test or real20 evaluation was run by this pipeline.'
