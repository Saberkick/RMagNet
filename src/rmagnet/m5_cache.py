"""Pinned C1 RGB8 cache. GT is never passed to the upstream model."""
import argparse, json, shutil, subprocess, time
from pathlib import Path
from .m4_cache import sha256
from .sma_data import load_manifest

ROOT=Path('/share/linmingheng-local/xuke')
PROJECT=ROOT/'RMagNet'
DATA=ROOT/'datasets/rmagnet_sma_dataset2'
C1=PROJECT/'runs/sma_c1_e10/best_sma.safetensors'
CACHE=PROJECT/'data_cache/m5_c1_rgb8_v1'

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--data',type=Path,default=DATA);p.add_argument('--checkpoint',type=Path,default=C1)
    a=p.parse_args();_,records,splits=load_manifest(a.data)
    identity={'version':'m5-c1-rgb8-v1','checkpoint':str(a.checkpoint),'checkpoint_sha256':sha256(a.checkpoint),
              'data_root':str(a.data),'dataset_manifest_sha256':sha256(a.data/'manifest.json'),
              'vae_posterior':'mode','condition_mode':'learned','quantization':'clamp(0,1)*255, torch.round, RGB uint8 PNG',
              'upstream_inputs':'ordinary I only','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    a.cache.mkdir(parents=True,exist_ok=True);(a.cache/'predictions').mkdir(exist_ok=True)
    index=a.cache/'manifest.json';previous=json.loads(index.read_text()) if index.exists() else None
    if previous:
        for k in ('version','checkpoint_sha256','dataset_manifest_sha256','vae_posterior','condition_mode','quantization'):
            if previous[k]!=identity[k]:raise RuntimeError('Cache identity changed: '+k)
    done=previous.get('samples',{}) if previous else {};identity.update(complete=False,samples=done)
    def flush():
        tmp=index.with_suffix('.tmp');tmp.write_text(json.dumps(identity,indent=2)+'\n');tmp.replace(index)
    flush();backend=None
    import torch
    from PIL import Image
    from .stage1_train import image_tensor
    from .m2a_data_baseline import save_prediction
    for split,ids in splits.items():
        reuse=PROJECT/f'runs/sma_c1_e10/{"validation_condition_learned" if split=="validation" else "test_best"}'
        report_path=reuse/'evaluation.json'
        reuse_ok=False
        if split!='train' and report_path.exists():
            report=json.loads(report_path.read_text())
            reuse_ok=(report['checkpoint_sha256']==identity['checkpoint_sha256'] and report['dataset_manifest_sha256']==identity['dataset_manifest_sha256'])
        for sid in ids:
            out=a.cache/'predictions'/f'{sid}.png';r=records[sid]
            if sid in done:
                if not out.exists() or sha256(out)!=done[sid]['prediction_sha256']:raise RuntimeError('Corrupt partial cache '+sid)
                continue
            source=reuse/'validation/step_000000/predictions'/f'{sid}.png'
            if reuse_ok and source.exists():shutil.copyfile(source,out);origin=str(source)
            else:
                if backend is None:
                    from .qwen_backend import QwenSharedBackend
                    from .m1b_train import load_initial
                    from .sma_eval import M4_SHA
                    from .sma_conditioned import ConditionedSMA,load_joint,install
                    torch.cuda.set_device(0);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
                    device=torch.device('cuda:0');backend=QwenSharedBackend.from_local(device)
                    initial=PROJECT/'runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors'
                    if sha256(initial)!=M4_SHA:raise RuntimeError('M4 source changed')
                    backend.set_trainable_branch('transmission');load_initial(backend,initial,device);backend.set_trainable_branch(None)
                    sma=ConditionedSMA().to(device);load_joint(a.checkpoint,backend,sma,device,M4_SHA)
                    sma.requires_grad_(False).eval();runtime=install(backend,sma);runtime.default_mode='learned'
                    backend.transformer.eval();backend.vae.eval()
                    if any(v.requires_grad for v in backend.transformer.parameters()) or any(v.requires_grad for v in backend.vae.parameters()):raise RuntimeError('Upstream is not frozen')
                with torch.no_grad():
                    pred=backend.forward_normalized(image_tensor(a.data/'blended'/f'{sid}.png').unsqueeze(0).to(device),'transmission')
                    save_prediction((pred.float()+1)*.5,out)
                del pred;origin='fresh deterministic C1 inference'
            with Image.open(out) as im:
                if im.mode!='RGB' or list(im.size)!=r['target_size']:raise RuntimeError('Cache geometry mismatch '+sid)
            done[sid]={'split':split,'size':r['target_size'],'input_sha256':r['processed']['input']['sha256'],
                       'gt_sha256':r['processed']['gt']['sha256'],'prediction_sha256':sha256(out),'origin':origin}
            flush();print(json.dumps({'cached':len(done),'total':len(records),'id':sid,'split':split}),flush=True)
    if set(done)!=set(records):raise RuntimeError('Incomplete cache')
    identity.update(complete=True,completed_unix=time.time(),counts={k:len(v) for k,v in splits.items()});flush()
    print('CACHE COMPLETE',flush=True)

if __name__=='__main__':main()
