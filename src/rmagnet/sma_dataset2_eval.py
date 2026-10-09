"""Matched clean merged-test evaluation for frozen M4 and saved SMA states."""
from .sma_joint import JOINT_VERSION, load_joint
import argparse,json,math
from types import MethodType
from .qwen_layer_probe import deterministic_encode
from pathlib import Path
import torch,lpips,safetensors.torch
from safetensors import safe_open
from torch.utils.data import DataLoader
from .sma import SMA,install,SMA_VERSION
from .sma_data import load_manifest
from .sma_eval import ROOT,M4_SHA
from .m4_cache import sha256
from .m1b_train import load_initial
from .m2b1_q20 import M2ValidationDataset
from .m2a_data_baseline import validate,macro
from .qwen_backend import QwenSharedBackend


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,help='Omit for plain fixed M4-best')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['validation','test'],default='test')
    p.add_argument('--data-root',type=Path,default=ROOT/'datasets/rmagnet_sma_dataset2')
    p.add_argument('--initial',type=Path,default=ROOT/'RMagNet/runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors')
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError(a.output)
    if sha256(a.initial)!=M4_SHA:raise RuntimeError('Fixed M4 identity mismatch')
    manifest,records,splits=load_manifest(a.data_root)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.manual_seed(2026);torch.cuda.manual_seed(2026)
    backend=QwenSharedBackend.from_local(device)
    backend.set_trainable_branch('transmission');load_initial(backend,a.initial,device);backend.set_trainable_branch(None)
    checkpoint=a.initial
    if a.checkpoint:
        checkpoint=a.checkpoint
        with safe_open(checkpoint,framework='pt') as h:
            md=h.metadata()
            if md.get('architecture') not in (SMA_VERSION,JOINT_VERSION) or md.get('base_m4_sha256')!=M4_SHA:
                raise RuntimeError('SMA architecture/base identity mismatch')
        sma=SMA().to(device)
        if md['architecture']==JOINT_VERSION:load_joint(checkpoint,backend,sma,device,M4_SHA)
        else:sma.load_state_dict(safetensors.torch.load_file(checkpoint,device=str(device)),strict=True)
        sma.requires_grad_(False);sma.eval();install(backend,sma)
    else:
        def deterministic_m4(self,image,branch):
            self.activate(branch)
            with torch.no_grad():latent=deterministic_encode(self,image)
            edited=self.upstream.flow_step(latent,self.transformer,self.vae,self.embeddings)
            return self.upstream.decode(edited,self.vae)
        backend.forward_normalized=MethodType(deterministic_m4,backend)
    if any(x.requires_grad for x in backend.transformer.parameters()) or any(x.requires_grad for x in backend.vae.parameters()):
        raise RuntimeError('Evaluation backbone must be frozen')
    model=lpips.LPIPS(net='squeeze',verbose=False).eval().cpu().requires_grad_(False)
    loader=DataLoader(M2ValidationDataset(a.data_root,records,splits[a.split]),batch_size=1)
    report=validate(backend,loader,device,model,a.output,0,2026)
    rows=report['per_image'];keys=tuple(report['means'])
    assert {r['id'] for r in rows}==set(splits[a.split]) and len(rows)==len(splits[a.split])
    assert all(math.isfinite(float(r[k])) for r in rows for k in keys)
    parts={}
    for source in ('original-clean-v3','data_set2'):
        selected=[r for r in rows if records[r['id']]['dataset_source']==source]
        parts[source]={'count':len(selected),'means':macro(selected,keys),'ids':[r['id'] for r in selected]}
    report.update({'model':('SMA-B-joint' if a.checkpoint and md['architecture']==JOINT_VERSION else 'SMA') if a.checkpoint else 'M4-best','split':a.split,'sample_count':len(rows),'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'m4_sha256':M4_SHA,'dataset_manifest_sha256':sha256(a.data_root/'manifest.json'),'dataset_root':str(a.data_root),'by_source':parts,'inference_inputs':'ordinary I only; GT exclusively for metrics; no P90/DoLP/cache at inference'})
    report['vae_posterior']='deterministic mode, same for SMA and M4'
    if a.checkpoint:
        config=json.loads((checkpoint.parent/'run_config.json').read_text())
        choice='best_metrics.json' if checkpoint.name=='best_sma.safetensors' else 'latest_metrics.json'
        chosen=json.loads((checkpoint.parent/choice).read_text())
        report['checkpoint_training_step']=chosen['step']
        report['checkpoint_completed_epoch']=chosen['step']//config['updates_per_epoch']
        report['checkpoint_selection_criterion']=chosen['criterion']
    report['evaluation_step_note']='step 0 is the evaluation output folder counter, not the training step'
    (a.output/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'complete','count':len(rows),'means':report['means'],'by_source':parts},indent=2),flush=True)


if __name__=='__main__':main()
