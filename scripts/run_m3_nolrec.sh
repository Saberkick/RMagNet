#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/linmingheng-local/xuke"
PROJECT="$ROOT/RMagNet"
PYTHON="$ROOT/envs/windowseat-py312/bin/python"
NVIDIA_USER_LIB="$ROOT/lib/nvidia-535.179"
RUN_DIR="${RUN_DIR:-$PROJECT/runs/m3_nolrec_70}"

export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH="$NVIDIA_USER_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

cd "$PROJECT"
exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  -m src.rmagnet.m3_nolrec \
  --run-dir "$RUN_DIR" \
  --epochs 10 \
  --max-steps 70 \
  --batch-size 1 \
  --gradient-accumulation 1 \
  --learning-rate 5e-6 \
  --warmup-steps 20 \
  --cluster-coefficient 0.25 \
  --relation-coefficient 0.10 \
  --consistency-coefficient 0.10 \
  --boundary-coefficient 0.05 \
  --validate-every 35 \
  --save-every 35 \
  --num-workers 1
