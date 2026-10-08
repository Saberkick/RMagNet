"""Evaluate a saved SMA checkpoint; GT is used only for scoring."""
import argparse,json
from pathlib import Path
import torch
import lpips
import safetensors.torch
from safetensors import safe_open
from torch.utils.data import DataLoader
from .sma import SMA,install,SMA_VERSION
from .sma_data import load_manifest
from .m4_cache import sha256
from .m1b_train import load_initial
from .m2b1_q20 import M2ValidationDataset
from .m2a_data_baseline import validate
from .qwen_backend import QwenSharedBackend


ROOT=Path('/share/linmingheng-local/xuke')
M4_SHA='897282b1bb9cfe61f96530df72edcf8a44a066bb819a3663e9100862aefdb2a3'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['validation','test'],default='test')
    p.add_argument('--data-root',type=Path,default=ROOT/'datasets/rmagnet_m2_aspect')
    p.add_argument('--initial',type=Path,default=ROOT/'RMagNet/runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors')
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError(a.output)
    if sha256(a.initial)!=M4_SHA:raise RuntimeError('Fixed M4 identity mismatch')
    _,records,splits=load_manifest(a.data_root)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    backend=QwenSharedBackend.from_local(device)
    backend.set_trainable_branch('transmission');load_initial(backend,a.initial,device);backend.set_trainable_branch(None)
    with safe_open(a.checkpoint,framework='pt') as handle:
        if handle.metadata().get('architecture')!=SMA_VERSION:raise RuntimeError('Unsupported SMA architecture version')
    sma=SMA().to(device)
    sma.load_state_dict(safetensors.torch.load_file(a.checkpoint,device=str(device)),strict=True)
    sma.requires_grad_(False);sma.eval();install(backend,sma)
    model=lpips.LPIPS(net='squeeze',verbose=False).eval().cpu().requires_grad_(False)
    loader=DataLoader(M2ValidationDataset(a.data_root,records,splits[a.split]),batch_size=1)
    report=validate(backend,loader,device,model,a.output,0,2026)
    report.update({'split':a.split,'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha256(a.checkpoint),'m4_sha256':M4_SHA,'dataset_manifest_sha256':sha256(a.data_root/'manifest.json')})
    (a.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['means'],indent=2),flush=True)


if __name__=='__main__':main()
