#!/usr/bin/env bash
set -euo pipefail

ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
ENV_DIR="$ROOT/envs/windowseat-py312"
DATA_ROOT="${DATA_ROOT:-$ROOT/datasets/rmagnet_stage1_512x384}"
RUN_DIR="${RUN_DIR:-$ROOT/RMagNet/runs/stage1_reflection_r8}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$ROOT/.cache/uv"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
NPROC="${NPROC_PER_NODE:-${#GPU_ARRAY[@]}}"
mkdir -p "$RUN_DIR"

cd "$PROJECT"
"$ENV_DIR/bin/python" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC" \
  -m rmagnet.stage1_train \
  --data-root "$DATA_ROOT" \
  --run-dir "$RUN_DIR" \
  --epochs "${EPOCHS:-100}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --gradient-accumulation "${GRADIENT_ACCUMULATION:-1}" \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --reflection-rank "${REFLECTION_RANK:-8}" \
  --warmup-steps "${WARMUP_STEPS:-20}" \
  --save-every "${SAVE_EVERY:-100}" \
  --validate-every "${VALIDATE_EVERY:-50}" \
  --keep-checkpoints "${KEEP_CHECKPOINTS:-2}" \
  --resume "${RESUME:-auto}" \
  "$@" 2>&1 | tee -a "$RUN_DIR/train.log"
