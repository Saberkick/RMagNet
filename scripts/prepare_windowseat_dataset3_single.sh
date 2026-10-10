#!/usr/bin/env bash
set -euo pipefail
SINGLE_GPU="${CUDA_VISIBLE_DEVICES:-2}"
source "$(dirname "$0")/sma_env.sh"
export CUDA_VISIBLE_DEVICES="$SINGLE_GPU"
[[ "$SINGLE_GPU" =~ ^[0-9]+$ ]] || exit 2
export PHYSICAL_GPU_INDEX="$SINGLE_GPU"
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i "$SINGLE_GPU" --query-gpu=uuid --format=csv,noheader | tr -d ' ')"
GROUP="${RUN_GROUP:-ws_c1_m5_dataset3_s1_e20}"
[[ "$GROUP" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
DATA="$ROOT/datasets/rmagnet_sma_dataset3"
CACHE="$PROJECT/data_cache/${GROUP}_layers"
MEMORY="$PROJECT/runs/$GROUP/memory_prepare"
INITIAL="$(uvpython -c "from src.rmagnet.qwen_backend import check_snapshots;print(check_snapshots()[1]/'transformer_lora/pytorch_lora_weights.safetensors')")"
EXPECTED_INITIAL=2f04225c5be1cac69a5d76faf4c95515d5503210cfee73c0bb1e79309f5a74e9
CUDA_VISIBLE_DEVICES="$SINGLE_GPU" check_gpus
mkdir -p "$PROJECT/runs/$GROUP/logs" "$CACHE"
uvpython -m src.rmagnet.windowseat_c1_audit --data-root "$DATA" --output "$PROJECT/runs/$GROUP/preflight_data.json"
if [[ -f "$CACHE/manifest.json" ]]; then
  # A complete cache is immutable: re-finalizing changes its hash and breaks resume.
  uvpython - "$DATA" "$CACHE" "$EXPECTED_INITIAL" <<'PY'
import sys
from pathlib import Path
from src.rmagnet.sma_data import load_manifest
from src.rmagnet.sma_c_train import load_cache
data,cache=map(Path,sys.argv[1:3]);_,_,splits=load_manifest(data)
manifest,_=load_cache(cache,data,splits['train'])
assert manifest['teacher']['adapter_sha256']==sys.argv[3]
print('Reusing complete, hash-verified cache without changing its manifest')
PY
else
  if [[ ! -f "$CACHE/reuse_record.json" ]]; then
    uvpython -m src.rmagnet.sma_dataset2 reuse-cache --output "$DATA" --cache "$CACHE" --reuse-cache "$PROJECT/data_cache/ws_c1_m5_filtered_e50_layers"
  fi
  uvpython -m src.rmagnet.sma_cache extract --data-root "$DATA" --output "$CACHE" --adapter "$INITIAL" \
    --adapter-description 'Official WindowSeat LoRA enabled; frozen native teacher; filtered old data plus confirmed data_set3' \
    --device cuda:0 --shard-index 0 --num-shards 1
  uvpython -m src.rmagnet.sma_cache finalize --data-root "$DATA" --output "$CACHE" --adapter "$INITIAL" \
    --adapter-description 'Official WindowSeat LoRA enabled; frozen native teacher; filtered old data plus confirmed data_set3'
fi
if [[ ! -f "$MEMORY/memory.safetensors" ]]; then
  uvpython -m src.rmagnet.sma_pretrain --cache-root "$CACHE" --output "$MEMORY" --epochs 5
fi
uvpython - "$CACHE" "$MEMORY" <<'PY'
import json,sys
from pathlib import Path
from src.rmagnet.m4_cache import sha256
c=Path(sys.argv[1]);m=Path(sys.argv[2]);cm=json.loads((c/'manifest.json').read_text());r=json.loads((m/'report.json').read_text())
assert cm['complete'] and r['memory_sha256']==sha256(m/'memory.safetensors')
assert r['cache_manifest_sha256']==sha256(c/'manifest.json') and r['teacher_sha256']==cm['teacher']['adapter_sha256']
assert r['train_images']==cm['source_dataset']['sample_count']
print('Prepared cache and memory verified')
PY
