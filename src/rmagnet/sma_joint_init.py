"""Recover unchanged pretrained memory and exact fresh readers, no retraining."""
import argparse,json
from pathlib import Path
import torch,safetensors.torch
from .sma import SMA
from .m4_cache import atomic_safetensors,sha256
from .sma_eval import ROOT,M4_SHA

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    old=ROOT/'RMagNet/runs/sma_dataset2_memory_pretrain/report.json'
    report=json.loads(old.read_text())
    source=ROOT/'RMagNet/runs/sma_dataset2_e50/best_sma.safetensors'
    cache=ROOT/'RMagNet/data_cache/sma_dataset2_v1/manifest.json'
    if report['teacher_sha256']!=M4_SHA or report['cache_manifest_sha256']!=sha256(cache):raise RuntimeError('Memory provenance mismatch')
    torch.manual_seed(2026);model=SMA()
    state=safetensors.torch.load_file(source)
    # Readers were not optimized during feature pretraining; recreate exact seed-2026 initialization.
    for key in model.state_dict():
        if not key.startswith('readers.'):model.state_dict()[key].copy_(state[key])
    a.output.mkdir(parents=True,exist_ok=True)
    target=a.output/'memory.safetensors'
    if not target.exists():
        atomic_safetensors(target,model.state_dict(),{'experiment':'SMA','teacher_sha256':M4_SHA,'cache_manifest_sha256':sha256(cache)})
    recovered=safetensors.torch.load_file(target)
    if not all(torch.equal(v,recovered[k]) for k,v in model.state_dict().items()):raise RuntimeError('Recovered tensors mismatch')
    latest=safetensors.torch.load_file(ROOT/'RMagNet/runs/sma_dataset2_e50/latest_sma.safetensors')
    if not all(torch.equal(v,latest[k]) for k,v in state.items() if not k.startswith('readers.')):raise RuntimeError('Memory changed in parent run')
    original_hash=report['memory_sha256']
    report=dict(report, memory_sha256=sha256(target), original_memory_sha256=original_hash,
                recovery='unchanged best/latest PCA and memory; seed-2026 fresh readers; no trained readers reused')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'recovery.json').write_text(json.dumps({'source':str(source),'source_sha256':sha256(source),'original_report':str(old),'recovered_sha256':sha256(target),'byte_identical_to_deleted_original':sha256(target)==original_hash,'memory_best_latest_tensor_equal':True,'readers_fresh':True},indent=2)+'\n')
    print('Frozen memory and fresh readers recovered',sha256(target))
if __name__=='__main__':main()
