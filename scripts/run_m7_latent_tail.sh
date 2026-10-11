#!/usr/bin/env bash
set -euo pipefail
ROOT=/share/linmingheng-local/xuke
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
ENVIRONMENT="$ROOT/envs/windowseat-py312"
UV_BIN=/home/xuke/.local/bin/uv
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT/lib/nvidia-535.179${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$ROOT/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$ROOT/.cache/uv" UV_OFFLINE=1 TMPDIR="$ROOT/tmp"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
EPOCHS="${1:-${EPOCHS:-5}}"
RUN_NAME="${RUN_NAME:-m7_m4_latent_tail_e5}"
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ && "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
DATA="$ROOT/datasets/rmagnet_sma_dataset3"
INITIAL="$ROOT/RMagNet/runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors"
CACHE="$PROJECT/data_cache/m7_m4_tail_v1"
RUN="$PROJECT/runs/$RUN_NAME"
mkdir -p "$RUN/logs" "$CACHE/logs"
cd "$PROJECT"
exec 9>"$PROJECT/runs/m7_pipeline.lock"
flock -n 9 || { echo 'M7 pipeline already running'; exit 3; }
IFS=, read -r -a cards <<< "$CUDA_VISIBLE_DEVICES"
(( ${#cards[@]}>=1 && ${#cards[@]}<=4 )) || exit 4
declare -A seen
for card in "${cards[@]}"; do
  [[ "$card" =~ ^[0-9]+$ && -z "${seen[$card]:-}" ]] || exit 4
  seen[$card]=1
  free="$(nvidia-smi -i "$card" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( free>=22000 )) || { echo "GPU $card busy ($free MiB free)"; exit 5; }
done
available="$(df --output=avail -B1 "$PROJECT" | tail -1 | tr -d ' ')"
(( available >= 10737418240 )) || { echo 'Need 10 GiB free for cache and checkpoints'; exit 6; }
py() { "$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" python "$@"; }
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN/pipeline.exit_code"' EXIT
printf '%s\n' "$$" > "$RUN/pipeline.pid"
common=(--data "$DATA" --cache "$CACHE" --initial "$INITIAL")
echo 'PHASE frozen_M4_input_only_cache'
pids=()
for ((i=0;i<${#cards[@]};i++)); do
  CUDA_VISIBLE_DEVICES="${cards[$i]}" py -m src.rmagnet.m7_latent_tail extract "${common[@]}" \
    --shard "$i" --shards "${#cards[@]}" > "$CACHE/logs/shard_$i.log" 2>&1 &
  pids+=("$!")
done
cleanup_workers() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap 'cleanup_workers; exit 130' INT TERM
bad=0
for pid in "${pids[@]}"; do wait "$pid" || bad=1; done
(( bad==0 )) || { echo 'Cache failed; training not started'; exit 7; }
py -m src.rmagnet.m7_latent_tail finalize "${common[@]}"
args=("${common[@]}" --output "$RUN" --epochs "$EPOCHS" --accumulation "${ACCUMULATION:-2}" --lr "${LR:-1e-4}" --rank 8)
if [[ "${RESUME:-0}" == 1 ]]; then args+=(--resume); fi
if [[ "${MAX_STEPS:-0}" != 0 ]]; then args+=(--max-steps "$MAX_STEPS"); fi
echo 'PHASE predecoder_tail_training'
py -m torch.distributed.run --standalone --nproc_per_node="${#cards[@]}" \
  -m src.rmagnet.m7_latent_tail train "${args[@]}" > "$RUN/logs/train.log" 2>&1
echo 'PHASE complete'
