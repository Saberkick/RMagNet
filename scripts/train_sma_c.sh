#!/usr/bin/env bash
set -euo pipefail
source /share/linmingheng-local/xuke/RMagNet/scripts/sma_env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
check_gpus
IFS=',' read -r -a gpu_list <<< "$CUDA_VISIBLE_DEVICES"
(( ${#gpu_list[@]} == 4 )) || { echo 'C requires four GPUs' >&2; exit 2; }
EPOCHS="${1:-${EPOCHS:-10}}"
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ && "$EPOCHS" -le 30 ]] || exit 2
RUN="${RUN_DIR:-$PROJECT/runs/sma_c1_e${EPOCHS}}"
[[ ! -e "$RUN" ]] || { echo 'Run exists; refusing overwrite' >&2; exit 3; }
[[ "$(df --output=avail -B1 "$ROOT" | tail -1 | tr -d ' ')" -ge 10737418240 ]] || exit 4
uvpython -m src.rmagnet.sma_joint_init --output "$PROJECT/runs/sma_joint_b_init"
args=(--joint-lora --data-root "$ROOT/datasets/rmagnet_sma_dataset2" --cache-root "$PROJECT/data_cache/sma_dataset2_v1"
 --initial "$INITIAL" --initial-sha256 "$EXPECTED_INITIAL" --memory-file "$PROJECT/runs/sma_joint_b_init/memory.safetensors"
 --run-dir "$RUN" --epochs "$EPOCHS" --max-steps "${MAX_STEPS:-0}" --learning-rate 1e-4 --lora-learning-rate 5e-6
 --warmup-steps 20 --gradient-ramp-steps 36 --spatial-gradient-ratio .08 --texture-gradient-ratio .08
 --semantic-gradient-ratio .08 --aux-gradient-cap .25 --consistency-coefficient .10 --ssim-weight .2 --edge-weight .1
 --early-stopping-patience 0 --num-workers "${NUM_WORKERS:-1}"
 --calibration-start-epoch "${CALIBRATION_START_EPOCH:-1}" --calibration-interval "${CALIBRATION_INTERVAL:-8}")
uvpython -m src.rmagnet.sma_c_train "${args[@]}" --preflight-only
exec "$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" python -m torch.distributed.run --standalone --nproc_per_node=4 -m src.rmagnet.sma_c_train "${args[@]}"
