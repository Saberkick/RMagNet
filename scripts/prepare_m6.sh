#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/m6_env.sh"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,3}"
IFS=',' read -r -a DEVICES <<< "$GPUS"
OUT="${M6_CACHE_ROOT:-$PROJECT/data_cache/m6_polar_negative_v1}"
LOGDIR="$PROJECT/runs/m6_cache_logs"
mkdir -p "$LOGDIR"
pids=()
stop_workers() { for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done; }
trap stop_workers INT TERM
for i in "${!DEVICES[@]}"; do
  available="$(nvidia-smi -i "${DEVICES[$i]}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( available >= 22000 )) || { stop_workers; echo "GPU ${DEVICES[$i]} is busy" >&2; exit 3; }
  CUDA_VISIBLE_DEVICES="${DEVICES[$i]}" setsid "$UV_BIN" run --no-project --python "$PYTHON" python -m src.rmagnet.m6_cache extract \
    --output "$OUT" --num-shards "${#DEVICES[@]}" --shard-index "$i" \
    > "$LOGDIR/shard_${i}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; stop_workers; fi; done
if (( failed )); then echo 'M6 cache worker failed; inspect shard logs' >&2; exit 1; fi
CUDA_VISIBLE_DEVICES="${DEVICES[0]}" py -m src.rmagnet.m6_cache finalize --output "$OUT"
