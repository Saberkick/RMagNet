#!/usr/bin/env bash
set -euo pipefail
source /share/linmingheng-local/xuke/RMagNet/scripts/sma_env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
check_gpus
EPOCHS="${1:-${EPOCHS:-4}}"
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ && "$EPOCHS" -le 20 ]] || exit 2
DATA="$ROOT/datasets/rmagnet_sma_dataset2"
CACHE="$PROJECT/data_cache/sma_dataset2_v1"
MEMORY="$PROJECT/runs/sma_joint_b_init"
RUN="${RUN_DIR:-$PROJECT/runs/sma_joint_b_e${EPOCHS}}"
[[ ! -e "$RUN" ]] || { echo 'Run already exists' >&2; exit 3; }
[[ "$(df --output=avail -B1 "$ROOT" | tail -1 | tr -d ' ')" -ge 10737418240 ]] || exit 4
uvpython -m src.rmagnet.sma_joint_init --output "$MEMORY"
args=(--joint-lora --data-root "$DATA" --cache-root "$CACHE" --initial "$INITIAL" --initial-sha256 "$EXPECTED_INITIAL"
 --memory-file "$MEMORY/memory.safetensors" --run-dir "$RUN" --epochs "$EPOCHS" --max-steps "${MAX_STEPS:-0}"
 --learning-rate 1e-4 --lora-learning-rate 5e-6 --warmup-steps 20 --gradient-ramp-steps 36
 --spatial-gradient-ratio .08 --texture-gradient-ratio .08 --semantic-gradient-ratio .08 --aux-gradient-cap .25
 --consistency-coefficient .10 --ssim-weight .2 --edge-weight .1 --early-stopping-patience 0 --num-workers "${NUM_WORKERS:-1}")
uvpython -m src.rmagnet.sma_train "${args[@]}" --preflight-only
exec "$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" python -m torch.distributed.run --standalone --nproc_per_node=4 -m src.rmagnet.sma_train "${args[@]}"
