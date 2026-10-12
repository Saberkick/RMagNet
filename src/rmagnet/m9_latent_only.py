"""M9: M4 editing LoRA, normalized final-latent L1 only; no RGB training loss."""
import argparse
import csv
import json
import math
import os
import random
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import safetensors.torch as st
from safetensors import safe_open
from diffusers import BitsAndBytesConfig, QwenImageTransformer2DModel
from peft import LoraConfig

from .qwen_backend import QwenSharedBackend, check_snapshots, load_upstream
from .qwen_layer_probe import deterministic_encode
from .m1b_train import load_initial
from .sma_data import load_manifest
from .m5_train import read, save_png, metrics, KEYS
from .m7_latent_tail import EXPECTED_M4, sha, atomic_json, atomic_tensors, event, emit

VERSION = 'm9-m4-final-latent-l1-v1'
STOP = False


def stop_request(*_):
    global STOP
    STOP = True


def identity(a):
    base, official = check_snapshots()
    return {'version': VERSION, 'initial_sha256': sha(a.initial),
            'dataset_sha256': sha(a.data/'manifest.json'),
            'qwen_snapshot': str(base), 'windowseat_snapshot': str(official),
            'latent': 'BF16 posterior mode; fixed per-channel mean/std normalization',
            'flow': 'native full forward; timestep BF16(499)/1000; final z=zI-velocity',
            'loss': 'mean absolute difference of final normalized latent and GT encoded latent'}


def cached(a, sid, device):
    v = st.load_file(a.cache/'samples'/f'{sid}.safetensors', device=str(device))
    return v['input'], v['gt']


@torch.inference_mode()
def prepare(a):
    if sha(a.initial) != EXPECTED_M4:
        raise RuntimeError('Wrong M4 final best')
    _, records, splits = load_manifest(a.data)
    ident = identity(a)
    ids = splits['train'] + splits['validation']
    path = a.cache/'manifest.json'
    if path.exists():
        existing = json.loads(path.read_text())
        if existing['identity'] != ident:
            raise RuntimeError('Different latent cache identity')
        if existing.get('complete'):
            for sid in ids:
                f = a.cache/'samples'/f'{sid}.safetensors'
                if sha(f) != existing['samples'][sid]['cache_sha256']:
                    raise RuntimeError('Bad cached file '+sid)
            emit({'phase': 'cache_reused', 'count': len(ids)}); return
    (a.cache/'samples').mkdir(parents=True, exist_ok=True)
    base, _ = check_snapshots()
    vae = load_upstream().load_qwen_vae(str(base), torch.device('cuda:0')).eval().requires_grad_(False)
    backend = SimpleNamespace(vae=vae)
    report = {'identity': ident, 'complete': False, 'samples': {},
              'splits': {'train': splits['train'], 'validation': splits['validation']},
              'sealed_test_encoded': False}
    for i, sid in enumerate(ids):
        file = a.cache/'samples'/f'{sid}.safetensors'
        source = {role: records[sid]['processed'][role]['sha256'] for role in ('input','gt')}
        md = {'identity': json.dumps(ident, sort_keys=True), 'sources': json.dumps(source, sort_keys=True), 'id': sid}
        reusable = False
        if file.exists():
            with safe_open(file, framework='pt') as f:
                reusable = f.metadata() == md
        if not reusable:
            tensors = {}
            for role, folder in (('input','blended'), ('gt','transmission_layer')):
                x = read(a.data/folder/f'{sid}.png').to('cuda:0')
                z = deterministic_encode(backend, x*2-1)
                if not torch.isfinite(z).all(): raise RuntimeError('Nonfinite VAE '+sid)
                tensors[role] = z.cpu()
            if tensors['input'].shape != tensors['gt'].shape: raise RuntimeError('Latent shape mismatch')
            atomic_tensors(file, tensors, md)
            del x, z, tensors
        values = st.load_file(file)
        if set(values) != {'input','gt'} or not all(torch.isfinite(v).all() for v in values.values()):
            raise RuntimeError('Invalid cached tensor '+sid)
        report['samples'][sid] = {'sources': source, 'latent_shape': list(values['gt'].shape),
                                  'cache_sha256': sha(file)}
        atomic_json(path, report)
        if i < 3 or (i+1)%20 == 0 or i+1 == len(ids):
            emit({'phase': 'cache', 'done': i+1, 'total': len(ids), 'id': sid})
    report['complete'] = True
    atomic_json(path, report); emit({'phase': 'cache_complete', 'count': len(ids)})


class LatentEditor:
    def __init__(self, initial, cards):
        base, official = check_snapshots(); up = load_upstream()
        self.primary = torch.device('cuda:0')
        self.mapping = {'pos_embed': 0, 'time_text_embed': 0, 'txt_norm': 0, 'img_in': 0, 'txt_in': 0,
                        'norm_out': cards-1, 'proj_out': cards-1}
        for i in range(60): self.mapping[f'transformer_blocks.{i}'] = min(cards-1, i*cards//60)
        nf4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 llm_int8_skip_modules=['transformer_blocks.0.img_mod'])
        t = QwenImageTransformer2DModel.from_pretrained(str(base), subfolder='transformer',
                torch_dtype=torch.bfloat16, quantization_config=nf4,
                device_map=self.mapping, local_files_only=True)
        t.add_adapter(LoraConfig.from_pretrained(str(official), subfolder='transformer_lora'), adapter_name='default')
        state = st.load_file(official/'transformer_lora/pytorch_lora_weights.safetensors')
        _, extra = t.load_state_dict(state, strict=False)
        if extra: raise RuntimeError('Unexpected official adapter keys')
        del state
        vae = up.load_qwen_vae(str(base), self.primary).eval().requires_grad_(False)
        embeds = st.load_file(official/'text_embeddings/state_dict.safetensors')
        resolution = json.loads((official/'model_index.json').read_text())['processing_resolution']
        self.backend = QwenSharedBackend(vae, t, embeds, resolution, up)
        self.backend.set_trainable_branch('transmission')
        load_initial(self.backend, initial, self.primary)
        self.names = {n:p for n,p in t.named_parameters() if p.requires_grad}
        if not self.names or any('.lora_' not in n or '.default.' not in n for n in self.names):
            raise RuntimeError('Trainable parameters must be existing editing LoRA only')
        for p in self.names.values(): p.data = p.data.float()
        t.enable_gradient_checkpointing()
        self.prompt = embeds['prompt_embeds'].to(self.primary, dtype=torch.bfloat16)
        self.mask = embeds['prompt_mask'].to(self.primary)
        self.timestep = torch.full((1,), float(499), device=self.primary, dtype=torch.bfloat16)/1000

    def forward(self, z):
        b = self.backend; h,w = z.shape[-2]*8,z.shape[-1]*8
        packed = b.upstream.QwenImageEditPipeline._pack_latents(z[:,:,0],1,16,h//8,w//8).to(torch.bfloat16)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = b.transformer(hidden_states=packed,timestep=self.timestep,
                encoder_hidden_states=self.prompt,encoder_hidden_states_mask=self.mask,
                img_shapes=[[(1,h//16,w//16)]],txt_seq_lens=self.mask.sum(1).tolist(),
                guidance=None,attention_kwargs={},return_dict=False)[0].to(self.primary)
            velocity = b.upstream.QwenImageEditPipeline._unpack_latents(out,height=h,width=w,vae_scale_factor=8)
        return z-velocity


@torch.no_grad()
def evaluate(a, runner, ids, out):
    out.mkdir(parents=True, exist_ok=True)
    runner.backend.transformer.eval(); rows=[]
    for sid in ids:
        z, target = cached(a,sid,runner.primary)
        edited = runner.forward(z)
        latent_l1 = float((edited.float()-target.float()).abs().mean())
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred = (runner.backend.upstream.decode(edited.to(runner.backend.vae.dtype),runner.backend.vae).float()+1)*.5
        if not torch.isfinite(pred).all(): raise RuntimeError('Nonfinite validation')
        path=out/'predictions'/f'{sid}.png'; save_png(pred,path)
        rows.append({'id':sid,'latent_l1':latent_l1,
                    **metrics(read(path),read(a.data/'transmission_layer'/f'{sid}.png'),read(a.data/'blended'/f'{sid}.png'))})
    means={k:sum(r[k] for r in rows)/len(rows) for k in ('latent_l1',*KEYS)}
    report={'count':len(rows),'means':means,'per_image':rows,'split':'validation','RGB8':True}
    atomic_json(out/'metrics.json',report)
    with open(out/'metrics.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return report


def train(a):
    if sha(a.initial)!=EXPECTED_M4: raise RuntimeError('Wrong M4 final best')
    _,records,splits=load_manifest(a.data); ident=identity(a)
    cache=json.loads((a.cache/'manifest.json').read_text())
    if not cache['complete'] or cache['identity']!=ident: raise RuntimeError('Wrong/incomplete cache')
    for sid in splits['train']+splits['validation']:
        if sha(a.cache/'samples'/f'{sid}.safetensors')!=cache['samples'][sid]['cache_sha256']:
            raise RuntimeError('Cache corrupted '+sid)
    if (a.output/'config.json').exists() and not a.resume: raise FileExistsError('Use --resume or new output')
    cards=torch.cuda.device_count()
    if cards not in (2,3,4): raise ValueError('Use 2-4 model parallel GPUs')
    if min(a.epochs,a.accumulation)<1 or a.lr<=0: raise ValueError('Invalid budget')
    torch.manual_seed(2026);random.seed(2026);torch.cuda.set_device(0)
    a.output.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGINT,stop_request);signal.signal(signal.SIGTERM,stop_request)
    runner=LatentEditor(a.initial,cards); params=list(runner.names.values())
    opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=0,foreach=False)
    updates=math.ceil(len(splits['train'])/a.accumulation)
    total=updates*a.epochs
    def factor(s):
        if s<10:return (s+1)/10
        return .1+.9*.5*(1+math.cos(math.pi*min(1,(s-10)/max(1,total-10))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,factor)
    cursor={'epoch':0,'batch':0,'step':0};best=-1e30;best_epoch=0
    def save(which):
        dest=a.output/which;dest.mkdir(parents=True,exist_ok=True)
        atomic_tensors(dest/'transmission_lora.safetensors',runner.names,
                       {'identity':json.dumps(ident),'cursor':json.dumps(cursor)})
        atomic_json(dest/'state.json',{'identity':ident,'cursor':dict(cursor),'best_epoch':best_epoch,'best_psnr':best})
        if which=='latest':
            tmp=dest/'trainer.pt.tmp'
            torch.save({'identity':ident,'cursor':dict(cursor),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                        'best':best,'best_epoch':best_epoch,'cards':cards,'accumulation':a.accumulation,'epochs':a.epochs},tmp)
            tmp.replace(dest/'trainer.pt')
    if a.resume:
        state=torch.load(a.output/'latest/trainer.pt',map_location='cpu',weights_only=False)
        if state['identity']!=ident or state['cards']!=cards or state['accumulation']!=a.accumulation or state['epochs']!=a.epochs:
            raise RuntimeError('Resume configuration differs')
        with safe_open(a.output/'latest/transmission_lora.safetensors',framework='pt') as f:
            md=f.metadata()
        if json.loads(md['cursor'])!=state['cursor']:raise RuntimeError('Model/trainer cursor mismatch')
        load_initial(runner.backend,a.output/'latest/transmission_lora.safetensors',runner.primary)
        opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        for p,values in opt.state.items():
            for k,v in values.items():
                if isinstance(v,torch.Tensor) and k!='step':values[k]=v.to(p.device)
        cursor=state['cursor'];best=state['best'];best_epoch=state['best_epoch'];del state
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    cfg.update(identity=ident,physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),device_map=runner.mapping,
        trainable_parameters=sum(p.numel() for p in params),precision='NF4 backbone; BF16 activations; FP32 LoRA/Adam',
        model_parallel_cards=cards,effective_batch=a.accumulation,updates_per_epoch=updates,planned_updates=total,
        train_count=len(splits['train']),validation_count=len(splits['validation']),sealed_test_count=len(splits['test']),
        loss={'normalized_final_latent_l1':1},pixel_training_loss=False,qwen_feature_loss=False,
        polar_loss=False,dolp=False,gates=False,exposure_alignment=False,weight_decay=0,gradient_clip=1,
        initialization='M4 final best editing LoRA',selection='maximum validation saved-RGB8 PSNR, baseline eligible',
        early_stopping=False,augmentation=False,retention='best and latest only; latest has optimizer',
        validation_every_epoch=True,train_order='seed 2026+epoch; max-area sample moved first')
    atomic_json(a.output/'config.json',cfg)
    emit({'phase':'loaded','cards':cards,'trainable_parameters':cfg['trainable_parameters']})
    if not a.resume:
        sid=max(splits['train'],key=lambda s:math.prod(records[s]['target_size']))
        z,target=cached(a,sid,runner.primary)
        with torch.no_grad():
            direct=deterministic_encode(runner.backend,read(a.data/'blended'/f'{sid}.png').to(runner.primary)*2-1)
            if not torch.equal(direct,z):raise RuntimeError('Cache differs from fresh encode')
        atomic_json(a.output/'cache_parity.json',{'id':sid,'input_max_error':0,'largest_train_geometry':records[sid]['target_size']})
        del z,target,direct
        baseline=evaluate(a,runner,splits['validation'],a.output/'m4_baseline_validation')
        # M4 native RGB8 baseline has previously been evaluated using the identical dataset and forward.
        ref=Path('/share/linmingheng-local/xuke/RMagNet-M8a/runs/m8a_m4_late_strong_gates_e2/m4_baseline_validation/metrics.json')
        if ref.exists():
            old=json.loads(ref.read_text())
            max_delta=max(abs(baseline['means'][k]-old['means'][k]) for k in KEYS)
            if max_delta>1e-4:raise RuntimeError('M4 baseline changed unexpectedly: '+str(max_delta))
            atomic_json(a.output/'baseline_parity.json',{'source':str(ref),'max_metric_delta':max_delta})
        best=baseline['means']['psnr'];save('best')
        atomic_json(a.output/'best_metrics.json',{'cursor':dict(cursor),**baseline})
        event(a.output/'metrics.jsonl',{'cursor':dict(cursor),'baseline':True,**baseline})
        emit({'phase':'baseline','means':baseline['means']})
    started=time.monotonic()
    for d in range(cards):torch.cuda.reset_peak_memory_stats(d)
    stopped=False
    while cursor['epoch']<a.epochs and not stopped:
        ep=cursor['epoch'];ids=list(splits['train']);random.Random(2026+ep).shuffle(ids)
        biggest=max(ids,key=lambda s:math.prod(records[s]['target_size']));ids.remove(biggest);ids.insert(0,biggest)
        runner.backend.transformer.train();runner.backend.vae.eval()
        for gi in range(cursor['batch'],updates):
            if STOP or (a.max_steps and cursor['step']>=a.max_steps):stopped=True;break
            group=ids[gi*a.accumulation:(gi+1)*a.accumulation]
            opt.zero_grad(set_to_none=True);mean_loss=0.
            for sid in group:
                z,target=cached(a,sid,runner.primary)
                edited=runner.forward(z)
                loss=(edited.float()-target.float()).abs().mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite latent loss '+sid)
                (loss/len(group)).backward();mean_loss+=float(loss.detach())/len(group)
                del z,target,edited,loss
            frozen_clean=all(p.grad is None for p in runner.backend.parameters() if not p.requires_grad)
            if not frozen_clean:raise RuntimeError('Frozen backbone/VAE gradient leak')
            squared={d:0. for d in range(cards)};witness={}
            for name,p in runner.names.items():
                if p.grad is not None:
                    q=float(p.grad.float().square().sum());squared[p.device.index]+=q
                    if q>0 and (p.device.index not in witness or q>witness[p.device.index][2]):witness[p.device.index]=(name,p,q)
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True,foreach=False))
            if norm<=0 or any(v<=0 for v in squared.values()):raise RuntimeError('Missing LoRA gradients')
            before={d:p.detach().clone() for d,(_,p,_) in witness.items()} if cursor['step']==0 else {}
            used_lr=opt.param_groups[0]['lr'];opt.step();scheduler.step()
            changed={str(d):bool(torch.any(witness[d][1].detach()!=v)) for d,v in before.items()}
            if changed and not all(changed.values()):raise RuntimeError('LoRA parameters did not change')
            del before
            cursor={'epoch':ep,'batch':gi+1,'step':cursor['step']+1}
            boundary=cursor['batch']==updates
            if boundary:cursor.update(epoch=ep+1,batch=0)
            record={'cursor':dict(cursor),'ids':group,'latent_l1':mean_loss,'gradient_norm':norm,
                    'gradient_norm_by_device':{str(d):v**.5 for d,v in squared.items()},
                    'frozen_no_grad':frozen_clean,'first_update_changed_by_device':changed,'lr':used_lr,
                    'peak_gpu_gib':{str(d):torch.cuda.max_memory_allocated(d)/2**30 for d in range(cards)},
                    'elapsed_seconds':time.monotonic()-started,'training_decoder_calls':0}
            event(a.output/'train.jsonl',record)
            if cursor['step']<=3 or cursor['step']%5==0:emit({'phase':'train',**record})
            if cursor['step']==1 or cursor['step']%20==0:save('latest')
            if boundary:
                report=evaluate(a,runner,splits['validation'],a.output/'latest_validation')
                improved=report['means']['psnr']>best+1e-8
                if improved:
                    best=report['means']['psnr'];best_epoch=cursor['epoch'];save('best')
                    # Fixed 25 filenames: overwrite current best predictions, no epoch directories.
                    import shutil
                    dest=a.output/'best_validation';dest.mkdir(parents=True,exist_ok=True)
                    shutil.copytree(a.output/'latest_validation',dest,dirs_exist_ok=True)
                    atomic_json(a.output/'best_metrics.json',{'cursor':dict(cursor),**report})
                atomic_json(a.output/'latest_metrics.json',{'cursor':dict(cursor),**report})
                event(a.output/'metrics.jsonl',{'cursor':dict(cursor),'best_updated':improved,**report})
                save('latest');emit({'phase':'validation','cursor':dict(cursor),'means':report['means'],'best_updated':improved})
                runner.backend.transformer.train()
        if STOP or (a.max_steps and cursor['step']>=a.max_steps):stopped=True
    save('latest')
    summary={'status':'complete' if cursor['epoch']==a.epochs else 'paused','cursor':cursor,
             'best_epoch':best_epoch,'best_validation_psnr':best,'sealed_test_evaluated':False}
    atomic_json(a.output/'training_summary.json',summary);emit({'phase':'finished',**summary})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('prepare','train'))
    for name in ('data','initial','cache','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=4);p.add_argument('--accumulation',type=int,default=6)
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','1')))
    (prepare if a.phase=='prepare' else train)(a)


if __name__=='__main__':main()
