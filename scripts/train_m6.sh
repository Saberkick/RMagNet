#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/m6_env.sh"
EPOCHS="${1:-${EPOCHS:-5}}"
GPUS="${CUDA_VISIBLE_DEVICES:-1,3}"
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid EPOCHS' >&2; exit 2; }
IFS=',' read -r -a DEVICES <<< "$GPUS"
declare -A SEEN
UUIDS=()
for gpu in "${DEVICES[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ && -z "${SEEN[$gpu]:-}" ]] || { echo 'Invalid/duplicate GPU' >&2; exit 2; }
  SEEN[$gpu]=1
  available="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( available >= 22000 )) || { echo "GPU $gpu only ${available} MiB free; leave other jobs running" >&2; exit 3; }
  UUIDS+=("$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d ' ')")
done
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${UUIDS[*]}")"
RUN_DIR="${M6_RUN_DIR:-$PROJECT/runs/m6_b_e${EPOCHS}}"
if [[ -d "$RUN_DIR" && -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Nonempty run directory: $RUN_DIR" >&2; exit 4
fi
py -m torch.distributed.run --standalone --nproc_per_node="${#DEVICES[@]}" \
  -m src.rmagnet.m6_train --epochs "$EPOCHS" --run-dir "$RUN_DIR" \
  --cache-root "${M6_CACHE_ROOT:-$M6_CACHE_DEFAULT}" \
  --num-workers "${NUM_WORKERS:-1}"
