#!/usr/bin/env bash
set -euo pipefail
ROOT=/share/linmingheng-local/xuke
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT/lib/nvidia-535.179${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="$ROOT/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$ROOT/.cache/uv" UV_OFFLINE=1 TMPDIR="$ROOT/tmp"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
EPOCHS="${1:-${EPOCHS:-5}}"
RUN_NAME="${RUN_NAME:-m8_m4_gates_14_37_54_e5}"
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ && "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
IFS=, read -r -a cards <<< "$CUDA_VISIBLE_DEVICES"
(( ${#cards[@]}>=2 && ${#cards[@]}<=4 )) || exit 3
declare -A seen
for gpu in "${cards[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ && -z "${seen[$gpu]:-}" ]] || exit 3
  seen[$gpu]=1
  free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( free>=18000 )) || { echo "GPU $gpu has insufficient free VRAM ($free MiB)"; exit 4; }
done
cd "$PROJECT"
RUN="$PROJECT/runs/$RUN_NAME";mkdir -p "$RUN/logs"
exec 9>"$PROJECT/runs/m8_pipeline.lock";flock -n 9 || exit 5
printf '%s\n' "$$" > "$RUN/pipeline.pid"
trap 'rc=$?; printf "%s\n" "$rc" > "$RUN/pipeline.exit_code"' EXIT
available="$(df --output=avail -B1 "$PROJECT" | tail -1 | tr -d ' ')"
(( available>=2147483648 )) || exit 6
args=(--data "$ROOT/datasets/rmagnet_sma_dataset3"
 --initial "$ROOT/RMagNet/runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors"
 --output "$RUN" --epochs "$EPOCHS" --accumulation "${ACCUMULATION:-6}" --lr "${LR:-1e-4}")
[[ "${RESUME:-0}" != 1 ]] || args+=(--resume)
[[ "${MAX_STEPS:-0}" == 0 ]] || args+=(--max-steps "$MAX_STEPS")
/home/xuke/.local/bin/uv run --no-project --python "$ROOT/envs/windowseat-py312/bin/python" \
  python -m src.rmagnet.m8_spatial_gates "${args[@]}" > "$RUN/logs/train.log" 2>&1
