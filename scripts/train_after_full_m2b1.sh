#!/usr/bin/env bash
set -euo pipefail
ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
ENVIRONMENT="$ROOT/envs/windowseat-py312"
UV_BIN="${UV_BIN:-/home/xuke/.local/bin/uv}"
RUN_NAME="${RUN_NAME:-AfterFullM2-B1}"
RUN_DIR="${RUN_DIR:-$PROJECT/runs/$RUN_NAME}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${EARLY_STOPPING_PATIENCE:-4}"
STEPS_PER_EPOCH=36
MAX_STEPS=$((EPOCHS * STEPS_PER_EPOCH))
export UV_CACHE_DIR="$ROOT/.cache/uv" UV_OFFLINE=1
export HF_HOME="$ROOT/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$ROOT/tmp" PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
[[ "$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")" -eq 4 ]] || { echo 'Exactly four GPUs are required' >&2; exit 1; }
[[ -x "$UV_BIN" && -x "$ENVIRONMENT/bin/python" ]] || { echo 'Pinned uv/Python missing' >&2; exit 1; }
[[ -f "$PROJECT/data_cache/m2a_q20/manifest.json" ]] || { echo 'M2 Q20 cache missing' >&2; exit 1; }
if [[ -d "$RUN_DIR" && -n "$(find "$RUN_DIR" -mindepth 1 -print -quit)" ]]; then echo "Run directory is not empty: $RUN_DIR" >&2; exit 1; fi
IFS=',' read -r -a gpu_ids <<<"$CUDA_VISIBLE_DEVICES"
for gpu in "${gpu_ids[@]}"; do
 free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
 (( free_mib >= 22000 )) || { echo "GPU $gpu has only ${free_mib} MiB free" >&2; exit 1; }
done
mkdir -p "$ROOT/tmp" "$UV_CACHE_DIR"
cd "$PROJECT"
"$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" python -m rmagnet.after_full_m2b1 --preflight-only
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then exit 0; fi
exec "$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" torchrun --standalone --nproc-per-node=4 -m rmagnet.after_full_m2b1 \
 --run-dir "$RUN_DIR" --epochs "$EPOCHS" --max-steps "$MAX_STEPS" \
 --batch-size 1 --gradient-accumulation 1 --learning-rate 5e-6 --warmup-steps 20 \
 --local-coefficient 0.25 --keep-coefficient 0.10 \
 --target-q-gradient-ratio 0.30 --lambda-q-min 0.0 --lambda-q-max 0.5 \
 --validate-every "$STEPS_PER_EPOCH" --save-every "$STEPS_PER_EPOCH" \
 --early-stopping-patience "$PATIENCE" --num-workers "${NUM_WORKERS:-1}"
