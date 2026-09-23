#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
ENVIRONMENT="$ROOT/envs/windowseat-py312"
UV_BIN="${UV_BIN:-/home/xuke/.local/bin/uv}"
EPOCHS="${1:-${EPOCHS:-10}}"
RUN_NAME="${RUN_NAME:-c1_l20_e${EPOCHS}}"
RUN_DIR="$PROJECT/runs/c1_l20/$RUN_NAME"
MAX_STEPS="${MAX_STEPS:-0}"
RESUME="${RESUME:-auto}"
CACHE="$PROJECT/data_cache/c1_l20"
INITIAL="$PROJECT/runs/stage2_transmission_r128/best_transmission_lora.safetensors"

if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$MAX_STEPS" =~ ^[0-9]+$ ]]; then
  echo "MAX_STEPS must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_NAME may contain only letters, digits, dot, underscore and dash" >&2
  exit 2
fi
if [[ "$RESUME" != "auto" && "$RESUME" != "none" ]]; then
  echo "RESUME must be auto or none" >&2
  exit 2
fi

export UV_CACHE_DIR="$ROOT/.cache/uv"
export UV_OFFLINE=1
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR="$ROOT/tmp"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")" -ne 4 ]]; then
  echo "train_c1_l20.sh requires exactly four visible GPUs" >&2
  exit 1
fi
if [[ ! -x "$UV_BIN" || ! -x "$ENVIRONMENT/bin/python" ]]; then
  echo "Pinned uv executable or environment is missing" >&2
  exit 1
fi
if [[ ! -f "$CACHE/manifest.json" || ! -f "$INITIAL" ]]; then
  echo "C1-L20 cache manifest or Stage-2 initialization is missing" >&2
  exit 1
fi
available_kib="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
if (( available_kib < 30 * 1024 * 1024 )); then
  echo "At least 30 GiB free space is required for top-3 and resumable checkpoints" >&2
  exit 1
fi

if [[ -d "$RUN_DIR" ]] && [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  if [[ "$RESUME" == "none" ]]; then
    echo "Run directory is non-empty and RESUME=none: $RUN_DIR" >&2
    exit 1
  fi
  if [[ ! -f "$RUN_DIR/checkpoints/last/trainer_state.pt" ]]; then
    echo "Run directory is non-empty but has no resumable last checkpoint: $RUN_DIR" >&2
    exit 1
  fi
fi

mkdir -p "$ROOT/tmp" "$UV_CACHE_DIR" "$RUN_DIR/logs"
cd "$PROJECT"
exec 9>"$RUN_DIR/.train.lock"
if ! flock -n 9; then
  echo "Another process is already using run directory: $RUN_DIR" >&2
  exit 1
fi

python3 - <<'PY'
import json
from pathlib import Path
root = Path('/share/linmingheng-local/xuke/RMagNet')
cache = root / 'data_cache/c1_l20'
manifest = json.loads((cache / 'manifest.json').read_text())
assert manifest.get('complete') is True
assert manifest.get('formula_version') == 'c1-l20-qwen-majority-v1'
assert len(manifest['train_ids']) == 50 and len(manifest['samples']) == 50
assert manifest['qwen_feature']['block_zero_based_index'] == 19
assert manifest['qwen_feature']['flow_timestep'] == 499
assert len(list((cache / 'gt_features').glob('*.safetensors'))) == 50
assert len(list((cache / 'weights').glob('*.npz'))) == 50
print('preflight cache check: OK (50 samples, Qwen block 20, timestep 499)')
PY

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "PREFLIGHT_ONLY=1: stopping before model load"
  exit 0
fi

"$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" \
  torchrun --standalone --nproc-per-node=4 -m rmagnet.c1_l20_train \
  --mode train --run-dir "$RUN_DIR" --epochs "$EPOCHS" --max-steps "$MAX_STEPS" \
  --batch-size 1 --gradient-accumulation 2 --learning-rate 5e-5 \
  --local-coefficient 0.25 --keep-coefficient 0.10 \
  --gradient-measure-every 20 --early-stop-patience 4 --minimum-epochs 5 \
  --num-workers "${NUM_WORKERS:-1}" --resume "$RESUME" \
  2>&1 | tee -a "$RUN_DIR/logs/console.log"
