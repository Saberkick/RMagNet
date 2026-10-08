#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/sma_env.sh"
check_gpus
mkdir -p "$CACHE_ROOT/logs"
IFS=',' read -r -a gpu_list <<< "$CUDA_VISIBLE_DEVICES"
pids=()
trap 'for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done' INT TERM
for shard in "${!gpu_list[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpu_list[$shard]}" uvpython -m src.rmagnet.sma_cache extract \
    --output "$CACHE_ROOT" --adapter "$INITIAL" --shard-index "$shard" --num-shards "${#gpu_list[@]}" \
    > "$CACHE_ROOT/logs/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || { echo 'Cache worker failed; training not started' >&2; exit 5; }
uvpython -m src.rmagnet.sma_cache finalize --output "$CACHE_ROOT" --adapter "$INITIAL"
CUDA_VISIBLE_DEVICES="${gpu_list[0]}" uvpython -m src.rmagnet.sma_pretrain --cache-root "$CACHE_ROOT" --output "$MEMORY_DIR" --epochs "${MEMORY_EPOCHS:-5}"
