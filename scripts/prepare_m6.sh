#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/m6_env.sh"
GPUS="${CUDA_VISIBLE_DEVICES:-1,3}"
IFS=',' read -r -a DEVICES <<< "$GPUS"
OUT="${M6_CACHE_ROOT:-$M6_CACHE_DEFAULT}"
REUSE="${M6_REUSE_ROOT:-$ROOT/RMagNet/data_cache/m4_multilayer_v1}"
LOGDIR="$PROJECT/runs/m6_cache_logs"
mkdir -p "$LOGDIR"
pids=()
stop_workers() { for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done; }
trap stop_workers INT TERM
for i in "${!DEVICES[@]}"; do
  [[ "${DEVICES[$i]}" =~ ^[0-9]+$ ]] || { stop_workers; echo 'Use physical GPU indices' >&2; exit 2; }
  available="$(nvidia-smi -i "${DEVICES[$i]}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( available >= 22000 )) || { stop_workers; echo "GPU ${DEVICES[$i]} is busy" >&2; exit 3; }
  uuid="$(nvidia-smi -i "${DEVICES[$i]}" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"
  CUDA_VISIBLE_DEVICES="$uuid" setsid "$UV_BIN" run --no-project --python "$PYTHON" python -m src.rmagnet.m6_cache extract \
    --output "$OUT" --reuse-root "$REUSE" --num-shards "${#DEVICES[@]}" --shard-index "$i" \
    > "$LOGDIR/shard_${i}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; stop_workers; fi; done
if (( failed )); then echo 'M6 cache worker failed; inspect shard logs' >&2; exit 1; fi
CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "${DEVICES[0]}" --query-gpu=uuid --format=csv,noheader | tr -d ' ')" py -m src.rmagnet.m6_cache finalize --output "$OUT"
