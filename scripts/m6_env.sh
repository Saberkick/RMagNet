#!/usr/bin/env bash
# Source from an M6 launcher; keep all caches and temporary files in personal storage.
ROOT=/share/linmingheng-local/xuke
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/envs/windowseat-py312/bin/python"
UV_BIN="$ROOT/.local/bin/uv"
if [[ ! -x "$UV_BIN" ]]; then UV_BIN=/home/xuke/.local/bin/uv; fi
export LD_LIBRARY_PATH="$ROOT/lib/nvidia-535.179${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$ROOT/.cache/uv" UV_OFFLINE=1 TMPDIR="$ROOT/tmp"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR"
cd "$PROJECT"
py() { "$UV_BIN" run --no-project --python "$PYTHON" python "$@"; }
