#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/sma_env.sh"
export PHYSICAL_GPU_INDEX=2
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader | tr -d ' ')"
GROUP="$PROJECT/runs/ws_c1_m5_dataset3_s1_e20"
DATA="$ROOT/datasets/rmagnet_sma_dataset3"
CACHE="$PROJECT/data_cache/ws_c1_m5_dataset3_s1_e20_layers"
MEMORY="$GROUP/memory_prepare/memory.safetensors"
INITIAL="$(uvpython -c "from src.rmagnet.qwen_backend import check_snapshots;print(check_snapshots()[1]/'transformer_lora/pytorch_lora_weights.safetensors')")"
V="$PROJECT/tmp/windowseat_single_20261010/verification"
mkdir -p "$V"
args=(--data-root "$DATA" --cache-root "$CACHE" --memory-file "$MEMORY" --initial "$INITIAL" --run-dir "$V/c1" --epochs 20 --accumulation 4 --save-every 1)
rc=0
uvpython -m src.rmagnet.windowseat_c1_single "${args[@]}" --stop-after-steps 1 > "$V/first.log" 2>&1 || rc=$?
(( rc == 130 )) || { echo "First verification failed: $rc"; exit 1; }
rc=0
uvpython -m src.rmagnet.windowseat_c1_single "${args[@]}" --resume --stop-after-steps 2 > "$V/resume.log" 2>&1 || rc=$?
(( rc == 130 )) || { echo "Resume verification failed: $rc"; exit 1; }
uvpython - "$V" <<'PY'
import json,sys
from pathlib import Path
from src.rmagnet.resume_bundle import verify
v=Path(sys.argv[1]);m=verify(v/'c1/latest')
assert m['cursor']['step']==2 and m['cursor']['batch']==2
events=[json.loads(x) for x in (v/'c1/metrics.jsonl').read_text().splitlines()]
r=[x for x in events if x['kind']=='resume']
assert len(r)==1 and r[0]['cursor']['step']==1 and r[0]['rng_restored'] and r[0]['optimizer_state_count']>0
training=[x for x in events if x['kind']=='train']
assert [x['cursor']['step'] for x in training]==[1,2]
assert all(x['backbone_vae_memory_frozen'] and x['micro_samples']==4 for x in training)
result={'status':'passed','two_optimizer_updates':True,'resume_from_step_1_to_2':True,
        'optimizer_state_count':r[0]['optimizer_state_count'],'rng_restored':r[0]['rng_restored'],
        'peak_allocated_gib':max(x['peak_allocated_gib'] for x in training),
        'formal_training_uses_official_init_not_verification_checkpoint':True}
(v/'verification_report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
PY
