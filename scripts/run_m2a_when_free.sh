#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
RUN_DIR="$PROJECT/runs/m2a_q20"
GATE_DIR="$PROJECT/runs/m2a_q20_gate"
FULL_CACHE="$PROJECT/data_cache/m2a_q20"
POLL_SECONDS="${M2A_POLL_SECONDS:-60}"
MIN_FREE_MIB="${M2A_MIN_FREE_MIB:-22000}"
GPU_CANDIDATES="${M2A_GPU_CANDIDATES:-0 1 2 3 4 5 6 7}"

mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/dispatcher.lock"
if ! flock -n 9; then
  echo "Another M2a dispatcher already owns $RUN_DIR/dispatcher.lock" >&2
  exit 1
fi

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

cd "$PROJECT"
log "Waiting for one GPU with at least ${MIN_FREE_MIB} MiB free"
while true; do
  selected=""
  for gpu in $GPU_CANDIDATES; do
    free_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if (( free_mib >= MIN_FREE_MIB )); then
      selected="$gpu"
      log "Selected physical GPU $gpu with ${free_mib} MiB free"
      break
    fi
  done
  if [[ -n "$selected" ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

export CUDA_VISIBLE_DEVICES="$selected"
export M2A_MIN_FREE_MIB="$MIN_FREE_MIB"

log "Running seven-bucket dynamic-shape gate"
M2A_CACHE_ROOT="$GATE_DIR" bash scripts/prepare_m2a_q20.sh --gate-only
M2A_CACHE_ROOT="$GATE_DIR" bash scripts/check_m2a_q20.sh
log "Seven-bucket gate passed"

log "Generating full 144-sample M2a cache"
M2A_CACHE_ROOT="$FULL_CACHE" bash scripts/prepare_m2a_q20.sh
M2A_CACHE_ROOT="$FULL_CACHE" bash scripts/check_m2a_q20.sh

printf '%s\n' "$selected" > "$RUN_DIR/gpu.txt"
date --iso-8601=seconds > "$RUN_DIR/completed_at.txt"
log "M2a full cache completed and validated"
