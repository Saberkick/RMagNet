#!/usr/bin/env bash
set -euo pipefail
ROOT=/share/linmingheng-local/xuke
PROJECT="$ROOT/RMagNet"
ENVIRONMENT="$ROOT/envs/windowseat-py312"
UV_BIN="${UV_BIN:-/home/xuke/.local/bin/uv}"
export LD_LIBRARY_PATH="$ROOT/lib/nvidia-535.179${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$ROOT/.cache/uv" UV_OFFLINE=1 TMPDIR="$ROOT/tmp"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.80"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT/data_cache/sma_m4final_v1}"
DATA_ROOT="${DATA_ROOT:-$ROOT/datasets/rmagnet_m2_aspect}"
MEMORY_DIR="${MEMORY_DIR:-$PROJECT/runs/sma_memory_pretrain}"
INITIAL="$PROJECT/runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors"
EXPECTED_INITIAL=897282b1bb9cfe61f96530df72edcf8a44a066bb819a3663e9100862aefdb2a3
cd "$PROJECT"
uvpython() { "$UV_BIN" run --no-project --python "$ENVIRONMENT/bin/python" python "$@"; }
check_gpus() {
  IFS=',' read -r -a gpu_list <<< "$CUDA_VISIBLE_DEVICES"
  if (( ${#gpu_list[@]} < 1 || ${#gpu_list[@]} > 4 )); then echo "Use 1–4 GPUs" >&2; return 1; fi
  declare -A seen
  for gpu in "${gpu_list[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ && -z "${seen[$gpu]:-}" ]] || return 2
    seen[$gpu]=1
    local free_mib
    free_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    (( free_mib >= ${MIN_FREE_MIB:-22000} )) || { echo "GPU $gpu is busy: $free_mib MiB free" >&2; return 3; }
  done
  [[ "$(sha256sum "$INITIAL" | cut -d ' ' -f1)" == "$EXPECTED_INITIAL" ]] || return 4
}
