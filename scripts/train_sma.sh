#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/sma_env.sh"
EPOCHS="${1:-${EPOCHS:-20}}"
MAX_STEPS="${MAX_STEPS:-0}"
RUN_DIR="${RUN_DIR:-$PROJECT/runs/sma_e${EPOCHS}}"
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ && "$EPOCHS" -le 20 && "$MAX_STEPS" =~ ^[0-9]+$ ]] || exit 2
check_gpus
IFS=',' read -r -a gpu_list <<< "$CUDA_VISIBLE_DEVICES"
# The inherited aspect sampler and M4 accounting are intentionally fixed to 4.
(( ${#gpu_list[@]} == 4 )) || { echo 'This first SMA training implementation requires exactly four GPUs' >&2; exit 3; }
args=(--data-root "$DATA_ROOT" --cache-root "$CACHE_ROOT" --initial "$INITIAL" --initial-sha256 "$EXPECTED_INITIAL"
  --memory-file "$MEMORY_DIR/memory.safetensors" --run-dir "$RUN_DIR"
  --epochs "$EPOCHS" --max-steps "$MAX_STEPS" --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-0}"
  --learning-rate "${LEARNING_RATE:-1e-4}" --warmup-steps 20 --gradient-ramp-steps 36
  --spatial-gradient-ratio .08 --texture-gradient-ratio .08 --semantic-gradient-ratio .08
  --aux-gradient-cap .25 --consistency-coefficient .10 --ssim-weight .2 --edge-weight .1
  --num-workers "${NUM_WORKERS:-1}")
if [[ -n "${CONTINUE_RUN:-}" ]]; then args+=(--continue-run "$CONTINUE_RUN"); fi
uvpython -m src.rmagnet.sma_train "${args[@]}" --preflight-only
exec "$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" \
  python -m torch.distributed.run --standalone --nproc_per_node=4 -m src.rmagnet.sma_train "${args[@]}"
