"""Native-size real20 evaluation with matched deterministic official tiling."""
import argparse, json, math, subprocess, time
from pathlib import Path
import torch, safetensors.torch
from safetensors import safe_open
from PIL import Image
from .real20_infer import ROOT, DATA, M4, sha256
from .sma import SMA, install, SMA_VERSION
from .sma_eval import M4_SHA
from .m1b_train import load_initial
from .qwen_backend import QwenSharedBackend
from .qwen_layer_probe import deterministic_encode
from .real20_report import read, metric


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    inputs=sorted((DATA/'blended').glob('*.jpg'),key=lambda p:int(p.stem))
    gts=sorted((DATA/'transmission_layer').glob('*.jpg'))
    if len(inputs)!=20 or {p.name for p in inputs}!={p.name for p in gts}: raise RuntimeError('Pair mismatch')
    for image in inputs:
        with Image.open(image) as i,Image.open(DATA/'transmission_layer'/image.name) as g:
            if i.size!=g.size:raise RuntimeError(f'Geometry mismatch {image.name}')
    if sha256(M4)!=M4_SHA:raise RuntimeError('M4 identity mismatch')
    checkpoint=a.checkpoint or M4
    metadata={}
    if a.checkpoint:
        with safe_open(a.checkpoint,framework='pt') as h:metadata=h.metadata()
        if metadata.get('architecture')!=SMA_VERSION or metadata.get('base_m4_sha256')!=M4_SHA:
            raise RuntimeError('SMA identity mismatch')
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.manual_seed(2026);torch.cuda.manual_seed(2026)
    backend=QwenSharedBackend.from_local(device)
    backend.set_trainable_branch('transmission');load_initial(backend,M4,device);backend.set_trainable_branch(None)
    runtime=None
    if a.checkpoint:
        sma=SMA().to(device)
        sma.load_state_dict(safetensors.torch.load_file(a.checkpoint,device=str(device)),strict=True)
        sma.requires_grad_(False);sma.eval();runtime=install(backend,sma)
    backend.activate('transmission');backend.transformer.eval();backend.vae.eval()
    if any(p.requires_grad for p in backend.transformer.parameters()):raise RuntimeError('Unfrozen backbone')
    upstream=backend.upstream
    original_encode,original_flow=upstream.encode,upstream.flow_step
    counts={'tiles':0,'sma_enabled_tiles':0}
    def encode(image,vae):return deterministic_encode(backend,image)
    def flow(latent,*args,**kwargs):
        counts['tiles']+=1
        if runtime is None:return original_flow(latent,*args,**kwargs)
        with runtime.scope(True,(latent.shape[-2]//2,latent.shape[-1]//2)):
            result=original_flow(latent,*args,**kwargs)
            if runtime.state.memory is None:raise RuntimeError('SMA inactive')
            counts['sma_enabled_tiles']+=1
            return result
    upstream.encode,upstream.flow_step=encode,flow
    a.output.mkdir(parents=True)
    started=time.monotonic()
    try:
        upstream.run_inference(backend.vae,backend.transformer,backend.embeddings,backend.resolution,
            str(DATA/'blended'),str(a.output),use_short_edge_tile=True,
            save_comparison=False,save_alternating=False,batch_size=1,num_workers=0)
    finally:upstream.encode,upstream.flow_step=original_encode,original_flow
    outputs=list(a.output.glob('*_windowseat_output.png'))
    if {p.name for p in outputs}!={f'{p.stem}_windowseat_output.png' for p in inputs}:raise RuntimeError('Output IDs mismatch')
    if runtime is not None and counts['sma_enabled_tiles']!=counts['tiles']:raise RuntimeError('Inactive SMA tiles')
    rows=[]
    for image in inputs:
        gt=DATA/'transmission_layer'/image.name
        prediction=a.output/f'{image.stem}_windowseat_output.png'
        with Image.open(prediction) as im,Image.open(gt) as g:
            if im.size!=g.size or im.mode!='RGB':raise RuntimeError('Prediction format mismatch')
            width,height=im.size
        score=metric(read(prediction),read(gt))
        if not all(math.isfinite(v) for v in score.values()):raise RuntimeError('Nonfinite metrics')
        rows.append({'id':image.stem,'width':width,'height':height,**score,
            'input_sha256':sha256(image),'gt_sha256':sha256(gt),'prediction_sha256':sha256(prediction)})
    import csv
    with (a.output/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    training={}
    if a.checkpoint:
        choice='best_metrics.json' if a.checkpoint.name=='best_sma.safetensors' else 'latest_metrics.json'
        selected=json.loads((a.checkpoint.parent/choice).read_text())
        config=json.loads((a.checkpoint.parent/'run_config.json').read_text())
        training={'step':selected['step'],'epoch':selected['step']//config['updates_per_epoch']}
    result={'status':'complete','sample_count':20,'means':{k:sum(r[k] for r in rows)/20 for k in ('l1','psnr','ssim')},
        'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'m4_sha256':M4_SHA,
        'checkpoint_training':training,'dataset':str(DATA),'dataset_commit':subprocess.check_output(['git','-C',str(DATA.parents[1]),'rev-parse','HEAD'],text=True).strip(),
        'vae_posterior':'mode','seed':2026,'processing_resolution':backend.resolution,
        'tiling':'official short-edge tiles, official Lanczos stitching, native output size',
        'metric_domain':'saved 8-bit RGB PNG vs native GT JPEG; image macro average; same SSIM as earlier real20',
        'inference_inputs':'ordinary I only; no GT, P90, DoLP or training cache supplied to model',
        'tile_checks':counts,'elapsed_seconds':time.monotonic()-started,
        'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30,'per_image':rows}
    (a.output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_image'},indent=2),flush=True)

if __name__=='__main__':main()
