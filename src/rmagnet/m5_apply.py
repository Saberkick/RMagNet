"""Apply fixed C1 + M5 to one processed-size image, with no GT/DoLP/P90."""
import argparse,gc,json
from pathlib import Path
import torch
from .m5_cache import CACHE,C1,PROJECT
from .m5_train import read,save_png,load_model
from .m4_cache import sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--t0',type=Path,help='Optional already-produced C1-best RGB8 PNG')
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    identity=json.loads((CACHE/'manifest.json').read_text())
    if not identity['complete'] or sha256(C1)!=identity['checkpoint_sha256']:raise RuntimeError('Pinned C1 identity changed')
    torch.set_num_threads(1);device=torch.device('cuda:0');torch.cuda.set_device(device)
    image=read(a.input)
    h,w=image.shape[-2:]
    if min(h,w)<32 or h%16 or w%16 or h*w>300000:raise ValueError('First M5 experiment supports processed images <=300k pixels, axes multiple16; native 4K requires a separate tiling pipeline')
    t0path=a.t0 or a.output.with_name(a.output.stem+'_c1.png')
    if a.t0 is None:
        if t0path.exists():raise FileExistsError(t0path)
        from .qwen_backend import QwenSharedBackend
        from .m1b_train import load_initial
        from .sma_eval import M4_SHA
        from .sma_conditioned import ConditionedSMA,load_joint,install
        torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
        backend=QwenSharedBackend.from_local(device)
        initial=PROJECT/'runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors'
        if sha256(initial)!=M4_SHA:raise RuntimeError('M4 source identity changed')
        backend.set_trainable_branch('transmission');load_initial(backend,initial,device);backend.set_trainable_branch(None)
        sma=ConditionedSMA().to(device);load_joint(C1,backend,sma,device,M4_SHA);sma.requires_grad_(False).eval()
        runtime=install(backend,sma);runtime.default_mode='learned';backend.transformer.eval();backend.vae.eval()
        with torch.no_grad():save_png((backend.forward_normalized(image.to(device)*2-1,'transmission').float()+1)*.5,t0path)
        # Hook closures can retain the backbone: collect them before pixel inference.
        del backend,sma,runtime;gc.collect();torch.cuda.empty_cache()
    t0=read(t0path)
    if image.shape!=t0.shape:raise ValueError('I and C1 output must have identical geometry')
    model,md=load_model(a.checkpoint,identity,device);model.eval()
    with torch.no_grad():result=model(image.to(device),t0.to(device));save_png(result['prediction'],a.output)
    report={'architecture':md,'input':str(a.input),'input_sha256':sha256(a.input),'c1_png':str(t0path),'c1_png_sha256':sha256(t0path),
            'c1_output_source':'provided by caller; provenance must be C1-best' if a.t0 else 'fresh pinned C1-best deterministic learned-mode',
            'output':str(a.output),'pixel_checkpoint_sha256':sha256(a.checkpoint),'inputs':'ordinary I only; T0 is the fixed C1 intermediate',
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
