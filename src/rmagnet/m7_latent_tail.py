"""M7: frozen M4, copied tail blocks with new LoRA, pre-decoder residual restoration.

Input-only frozen caches; GT is read exclusively by pixel supervision/evaluation.
Cache extraction and trainable execution are separate to fit shared 24 GB GPUs.
"""
from __future__ import annotations
import argparse, contextlib, csv, gc, hashlib, json, math, os, random, signal, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
import safetensors.torch as st
from safetensors import safe_open
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
from .qwen_backend import QwenSharedBackend, check_snapshots, load_upstream
from .qwen_layer_probe import deterministic_encode
from .m1b_train import load_initial
from .sma_data import load_manifest
from .m5_train import read, save_png, metrics, KEYS

VERSION = 'm7-m4-predecoder-tail-v1'
EXPECTED_M4 = '897282b1bb9cfe61f96530df72edcf8a44a066bb819a3663e9100862aefdb2a3'
STOP = False

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def atomic_json(p,j):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.json.tmp');t.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n');t.replace(p)

def atomic_tensors(p,j,metadata=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.safetensors.tmp')
    st.save_file({k:v.detach().cpu().contiguous() for k,v in j.items()},t,metadata=metadata);t.replace(p)

def emit(j):print(json.dumps(j,allow_nan=False),flush=True)

def event(p,j):
    with open(p,'a') as f:f.write(json.dumps(j,allow_nan=False)+'\n')

def new_block():return QwenImageTransformerBlock(dim=3072,num_attention_heads=24,attention_head_dim=128)

@torch.no_grad()
def copy_effective_block(source):
    """Export the actually used NF4 base plus trained M4 LoRA, not unadapted Qwen."""
    import bitsandbytes as bnb
    dest=new_block().to(dtype=torch.bfloat16)
    copied=set()
    for name,m in dest.named_modules():
        if not isinstance(m,nn.Linear):continue
        s=source.get_submodule(name)
        b=s.get_base_layer() if hasattr(s,'get_base_layer') else s
        if isinstance(b,bnb.nn.Linear4bit):
            w=bnb.functional.dequantize_4bit(b.weight.data,b.weight.quant_state).float().cpu()
        else:w=b.weight.detach().float().cpu()
        if hasattr(s,'lora_A') and 'default' in s.lora_A:
            w.add_(s.lora_B['default'].weight.detach().float().cpu() @
                   s.lora_A['default'].weight.detach().float().cpu(),alpha=float(s.scaling['default']))
        m.weight.copy_(w);copied.add(name+'.weight')
        if m.bias is not None:
            if b.bias is None:raise RuntimeError('Missing copied bias '+name)
            m.bias.copy_(b.bias.detach().cpu());copied.add(name+'.bias')
        del w
    for name,p in dest.named_parameters():
        if name not in copied:p.copy_(source.get_parameter(name).detach().cpu())
    dest.requires_grad_(False).eval()
    return dest

def identity(a):
    if sha(a.initial)!=EXPECTED_M4:raise RuntimeError('Wrong M4 final best checkpoint')
    base,official=check_snapshots()
    return {'version':VERSION,'m4_sha256':EXPECTED_M4,'dataset_sha256':sha(a.data/'manifest.json'),
            'base_snapshot':str(base),'official_snapshot':str(official),'tail_indices':[58,59],
            'timestep':499,'posterior':'mode','condition':'M4 fixed text, input only'}

def valid_sample(a,sid,ident):
    p=a.cache/'samples'/f'{sid}.safetensors';j=a.cache/'records'/f'{sid}.json'
    if not p.is_file() or not j.is_file():return False
    try:
        r=json.loads(j.read_text())
        return r['identity']==ident and r['sha256']==sha(p)
    except (OSError,KeyError,ValueError):return False

def cache_extract(a):
    torch.set_num_threads(1);torch.cuda.set_device(0)
    _,records,splits=load_manifest(a.data);ident=identity(a)
    ids=splits['train']+splits['validation']
    selected=ids[a.shard::a.shards]
    todo=[s for s in selected if not valid_sample(a,s,ident)]
    emit({'phase':'cache_start','shard':a.shard,'selected':len(selected),'remaining':len(todo)})
    if not todo and (a.shard!=0 or (a.cache/'frozen_tail.safetensors').is_file()):return
    device=torch.device('cuda:0');torch.manual_seed(2026)
    backend=QwenSharedBackend.from_local(device)
    backend.set_trainable_branch('transmission');load_initial(backend,a.initial,device)
    backend.set_trainable_branch(None);backend.eval()
    if any(p.requires_grad for p in backend.parameters()):raise RuntimeError('M4 is not frozen')
    if a.shard==0:
        tensor_state={};parity=[]
        for k,index in enumerate((58,59)):
            block=copy_effective_block(backend.transformer.transformer_blocks[index])
            tensor_state.update({f'blocks.{k}.{n}':v for n,v in block.state_dict().items()})
            del block;gc.collect()
        atomic_tensors(a.cache/'frozen_tail.safetensors',tensor_state,{'identity':json.dumps(ident)})
        del tensor_state;gc.collect();torch.cuda.empty_cache()
        atomic_json(a.cache/'frozen_tail.json',{'identity':ident,'sha256':sha(a.cache/'frozen_tail.safetensors'),
            'construction':'BF16 copy of NF4 dequantized M4 base plus existing M4 default LoRA'})
    captured={}
    def hook(_m,_args,kwargs,out):
        captured['hidden']=out[1].detach().cpu()
        captured['text']=out[0].detach().cpu()
        captured['temb']=kwargs['temb'].detach().cpu()
        mask=kwargs['encoder_hidden_states_mask']
        captured['mask']=mask.detach().cpu() if mask is not None else torch.ones(out[0].shape[:2],dtype=torch.bool)
        for k,r in enumerate(kwargs['image_rotary_emb']):
            captured[f'rope_{k}']=torch.view_as_real(r).detach().cpu() if r.is_complex() else r.detach().cpu()
            captured[f'rope_complex_{k}']=torch.tensor(r.is_complex())
    handle=backend.transformer.transformer_blocks[-1].register_forward_hook(hook,with_kwargs=True)
    try:
        with torch.no_grad():
            for pos,sid in enumerate(todo,1):
                captured.clear();torch.cuda.reset_peak_memory_stats()
                image=read(a.data/'blended'/f'{sid}.png')*2-1
                z=deterministic_encode(backend,image)
                edited=backend.upstream.flow_step(z,backend.transformer,backend.vae,backend.embeddings)
                tensors={**captured,'z_input':z.detach().cpu(),'z_m4':edited.detach().cpu()}
                w,h=records[sid]['target_size'];n=(h//16)*(w//16)
                if tensors['hidden'].shape!=(1,n,3072):raise RuntimeError('Unexpected image token layout')
                if any(not torch.isfinite(v.float()).all() for v in tensors.values()):raise RuntimeError('Nonfinite cache')
                p=a.cache/'samples'/f'{sid}.safetensors'
                atomic_tensors(p,tensors,{'id':sid,'identity':json.dumps(ident)})
                atomic_json(a.cache/'records'/f'{sid}.json',{'id':sid,'identity':ident,'sha256':sha(p),
                    'bytes':p.stat().st_size,'input_sha256':sha(a.data/'blended'/f'{sid}.png'),
                    'size_wh':[w,h],'peak_gib':torch.cuda.max_memory_allocated()/2**30})
                emit({'phase':'cache_sample','shard':a.shard,'done':pos,'total':len(todo),'id':sid,
                    'peak_gib':torch.cuda.max_memory_allocated()/2**30})
                del tensors,z,edited,image;gc.collect();torch.cuda.empty_cache()
    finally:handle.remove()

def cache_finalize(a):
    _,records,splits=load_manifest(a.data);ident=identity(a)
    ids=splits['train']+splits['validation']
    for sid in ids:
        if not valid_sample(a,sid,ident):raise RuntimeError('Incomplete cache '+sid)
        r=json.loads((a.cache/'records'/f'{sid}.json').read_text())
        if r['input_sha256']!=records[sid]['processed']['input']['sha256']:raise RuntimeError('Input changed')
    fr=json.loads((a.cache/'frozen_tail.json').read_text())
    if fr['identity']!=ident or fr['sha256']!=sha(a.cache/'frozen_tail.safetensors'):raise RuntimeError('Wrong copied tail')
    total=sum((a.cache/'samples'/f'{sid}.safetensors').stat().st_size for sid in ids)
    atomic_json(a.cache/'manifest.json',{'complete':True,'identity':ident,'splits':splits,
        'cached_ids':ids,'test_cached':False,'tail_sha256':fr['sha256'],'sample_bytes':total})
    emit({'phase':'cache_complete','train':len(splits['train']),'validation':len(splits['validation']),
          'sample_gib':total/2**30,'tail_gib':(a.cache/'frozen_tail.safetensors').stat().st_size/2**30})

class NewLoRALinear(nn.Module):
    def __init__(self,base,rank):
        super().__init__();self.base=base.requires_grad_(False)
        self.a=nn.Parameter(torch.empty(rank,base.in_features,dtype=torch.float32))
        self.b=nn.Parameter(torch.zeros(base.out_features,rank,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
    def forward(self,x):return self.base(x)+F.linear(F.linear(x,self.a),self.b).to(x.dtype)

def inject_lora(module,rank):
    for name,child in list(module.named_children()):
        if isinstance(child,nn.Linear):setattr(module,name,NewLoRALinear(child,rank))
        else:inject_lora(child,rank)

class ResidualConv(nn.Module):
    def __init__(self,c):
        super().__init__();self.net=nn.Sequential(nn.GroupNorm(8,c),nn.Conv2d(c,c,3,padding=1),nn.SiLU(),nn.Conv2d(c,c,3,padding=1))
    def forward(self,x):return x+self.net(x)*.1

class TailRestorer(nn.Module):
    def __init__(self,tailfile,rank=8):
        super().__init__();self.blocks=nn.ModuleList([new_block().to(dtype=torch.bfloat16) for _ in range(2)])
        self.load_state_dict(st.load_file(tailfile),strict=True)
        self.requires_grad_(False)
        for b in self.blocks:inject_lora(b,rank)
        self.hidden_norm=nn.LayerNorm(3072)
        self.hidden_project=nn.Linear(3072,128)
        self.texture=nn.Sequential(nn.Conv2d(3,32,3,stride=2,padding=1),nn.SiLU(),
            nn.Conv2d(32,48,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(48,64,3,stride=2,padding=1),nn.SiLU())
        self.fuse=nn.Sequential(nn.Conv2d(128+64+32,128,3,padding=1),nn.SiLU(),ResidualConv(128),ResidualConv(128),ResidualConv(128))
        self.output=nn.Conv2d(128,16,3,padding=1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.bound=.25
    def forward(self,c,image):
        x,t=c['hidden'],c['text'];rope=tuple(c[f'rope_{k}'] for k in range(2))
        for block in self.blocks:
            def apply(x,t,block=block):
                return block(hidden_states=x,encoder_hidden_states=t,encoder_hidden_states_mask=c['mask'],
                             temb=c['temb'],image_rotary_emb=rope)
            if self.training:t,x=checkpoint(apply,x,t,use_reentrant=False)
            else:t,x=apply(x,t)
        h,w=image.shape[-2:];gh,gw=h//16,w//16
        features=self.hidden_project(self.hidden_norm(x.float())).transpose(1,2).reshape(1,128,gh,gw)
        features=F.interpolate(features,size=(h//8,w//8),mode='bilinear',align_corners=False)
        texture=self.texture(image)
        if texture.shape[-2:]!=(h//8,w//8):raise RuntimeError('Texture geometry mismatch')
        raw=self.output(self.fuse(torch.cat([features,texture,c['z_input'][:,:,0],c['z_m4'][:,:,0]],1)))
        delta=self.bound*torch.tanh(raw.float())
        return c['z_m4'].float()+delta.unsqueeze(2),delta

def cache_sample(a,sid,device):
    c=st.load_file(a.cache/'samples'/f'{sid}.safetensors',device=str(device))
    for k in range(2):
        if bool(c.pop(f'rope_complex_{k}')):c[f'rope_{k}']=torch.view_as_complex(c[f'rope_{k}'].contiguous())
    return c,read(a.data/'blended'/f'{sid}.png').to(device),read(a.data/'transmission_layer'/f'{sid}.png').to(device)

def decode(vae,z):
    value=z.to(vae.dtype)
    def run(v):return load_upstream().decode(v,vae)
    # This pinned VAE does not expose enable_gradient_checkpointing().
    # Checkpoint the frozen decoder call while preserving gradients to latent.
    decoded=checkpoint(run,value,use_reentrant=False) if torch.is_grad_enabled() and value.requires_grad else run(value)
    return (decoded.float()+1)*.5

def charbonnier(x,y):return ((x-y).square()+1e-6).sqrt().mean()

def gradient_loss(x,y):
    return .5*(charbonnier(x[...,1:,:]-x[...,:-1,:],y[...,1:,:]-y[...,:-1,:])+
               charbonnier(x[...,:,1:]-x[...,:,:-1],y[...,:,1:]-y[...,:,:-1]))

def loss_fn(pred,gt,delta):
    terms={'rec':charbonnier(pred,gt),'gradient':gradient_loss(pred,gt),
        'multiscale':sum(charbonnier(F.avg_pool2d(pred,s),F.avg_pool2d(gt,s)) for s in (2,4))*.5,
        'correction':delta.abs().mean()}
    return terms['rec']+.1*terms['gradient']+.2*terms['multiscale']+.005*terms['correction'],terms

def trainable_state(m):
    return {k:p.detach().cpu().contiguous() for k,p in m.named_parameters() if p.requires_grad}

def restore_trainable(m,state):
    params={k:p for k,p in m.named_parameters() if p.requires_grad}
    if set(state)!=set(params):raise RuntimeError('Restorer state key mismatch')
    with torch.no_grad():
        for k,p in params.items():p.copy_(state[k].to(p.device,p.dtype))

@torch.no_grad()
def evaluate(a,model,vae,ids,device,out,baseline=False):
    model.eval();rows=[];base=[];out.mkdir(parents=True,exist_ok=True)
    for sid in ids:
        c,image,gt=cache_sample(a,sid,device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z,delta=model(c,image);pred=decode(vae,z)
            t0=decode(vae,c['z_m4'])
        if baseline and not torch.equal(z,c['z_m4'].float()):raise RuntimeError('Initial result differs from frozen M4 latent')
        save_png(pred,out/'predictions'/f'{sid}.png')
        rgb=read(out/'predictions'/f'{sid}.png')
        # Same RGB8 domain for both restorer and the frozen M4 baseline.
        t0=(t0.float().clamp(0,1)*255).round()/255
        rows.append({'id':sid,**metrics(rgb,gt.cpu(),image.cpu())})
        base.append({'id':sid,**metrics(t0.cpu(),gt.cpu(),image.cpu())})
        del c,image,gt,pred,t0,z,delta
    means={k:sum(r[k] for r in rows)/len(rows) for k in KEYS}
    bm={k:sum(r[k] for r in base)/len(base) for k in KEYS}
    report={'count':len(rows),'means':means,'m4_baseline':bm,'minus_m4':{k:means[k]-bm[k] for k in KEYS},
        'domain':'saved RGB8, macro average, existing processed sizes, uniform11x11 SSIM',
        'per_image':rows,'initial_latent_parity':baseline}
    atomic_json(out/'metrics.json',report)
    with open(out/'metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return report

def stop_request(*_):
    global STOP;STOP=True

def train(a):
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    device=torch.device('cuda',local);torch.cuda.set_device(device);torch.set_num_threads(1)
    if world>1:dist.init_process_group('nccl',device_id=device)
    signal.signal(signal.SIGTERM,stop_request);signal.signal(signal.SIGINT,stop_request)
    torch.manual_seed(2026);random.seed(2026)
    _,records,splits=load_manifest(a.data);manifest=json.loads((a.cache/'manifest.json').read_text());ident=identity(a)
    if not manifest['complete'] or manifest['identity']!=ident:raise RuntimeError('Cache identity mismatch')
    if manifest['tail_sha256']!=sha(a.cache/'frozen_tail.safetensors'):raise RuntimeError('Tail file changed')
    if any(not valid_sample(a,s,ident) for s in splits['train']+splits['validation']):raise RuntimeError('Broken cache')
    base,_=check_snapshots();vae=load_upstream().load_qwen_vae(str(base),device).eval().requires_grad_(False)
    model=TailRestorer(a.cache/'frozen_tail.safetensors',a.rank).to(device)
    params=[p for p in model.parameters() if p.requires_grad]
    if any(p.requires_grad for p in vae.parameters()):raise RuntimeError('VAE not frozen')
    if any(p.requires_grad for n,p in model.named_parameters() if '.base.' in n):raise RuntimeError('Copied bases not frozen')
    optimizer=torch.optim.AdamW(params,lr=a.lr,weight_decay=1e-4)
    effective=world*a.accumulation;updates=math.ceil(len(splits['train'])/effective)
    def factor(s):
        if s<10:return (s+1)/10
        return .1+.9*.5*(1+math.cos(math.pi*min(1,(s-10)/max(1,a.epochs*updates-10))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    cursor={'epoch':0,'batch':0,'step':0};best=float('inf');best_epoch=0
    if a.resume:
        state=torch.load(a.output/'latest'/'trainer.pt',map_location='cpu',weights_only=False)
        if state['identity']!=ident or state['world_size']!=world or state['accumulation']!=a.accumulation:raise RuntimeError('Resume identity/config mismatch')
        restore_trainable(model,st.load_file(a.output/'latest'/'model.safetensors'))
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        cursor=state['cursor'];best=state['best'];best_epoch=state['best_epoch']
        for values in optimizer.state.values():
            for k,v in values.items():
                if isinstance(v,torch.Tensor):values[k]=v.to(device)
        del state
    elif (a.output/'config.json').exists():raise FileExistsError('Use a new run directory or --resume')
    # Final text-stream outputs do not feed the image head, so some text LoRA
    # parameters are unused in the last copied block. DDP must detect these.
    if world>1:wrapped=DDP(model,device_ids=[local],find_unused_parameters=True)
    else:wrapped=model
    a.output.mkdir(parents=True,exist_ok=True)
    if rank==0:
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
        config.update(identity=ident,world_size=world,effective_batch=effective,updates_per_epoch=updates,
            planned_steps=updates*a.epochs,train_count=len(splits['train']),validation_count=len(splits['validation']),
            test_count=len(splits['test']),trainable_parameters=sum(p.numel() for p in params),
            precision='BF16 autocast; FP32 trainable weights/optimizer',early_stopping=False,
            loss_weights={'rec':1.,'gradient':.1,'multiscale':.2,'correction':.005},retention='best model; latest model+optimizer',
            train_input='I only; no GT/P90/DoLP in frozen features or restoration forward',
            augmentation='none: cached positional features are not geometrically altered',selection='minimum validation RGB8 L1; epoch0 eligible')
        atomic_json(a.output/'config.json',config)
        emit({'phase':'train_loaded',**{k:config[k] for k in ('world_size','effective_batch','updates_per_epoch','trainable_parameters')}})
    def barrier():
        if world>1:dist.barrier()
    def save(which):
        folder=a.output/which;folder.mkdir(parents=True,exist_ok=True)
        atomic_tensors(folder/'model.safetensors',trainable_state(model),{'identity':json.dumps(ident),'cursor':json.dumps(cursor),'rank':str(a.rank)})
        atomic_json(folder/'state.json',{'cursor':cursor,'identity':ident,'best_epoch':best_epoch,'best_l1':best})
        if which=='latest':
            tmp=folder/'trainer.pt.tmp'
            torch.save({'identity':ident,'world_size':world,'accumulation':a.accumulation,'cursor':dict(cursor),
                'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'best':best,'best_epoch':best_epoch},tmp)
            tmp.replace(folder/'trainer.pt')
    def validate(baseline=False):
        nonlocal best,best_epoch
        barrier()
        if rank==0:
            report=evaluate(a,model,vae,splits['validation'],device,a.output/'latest_validation',baseline)
            improved=report['means']['l1']<best-1e-6
            if improved:
                best=report['means']['l1'];best_epoch=cursor['epoch'];save('best')
                import shutil
                target=a.output/'best_validation'
                if target.exists():shutil.rmtree(target)
                shutil.copytree(a.output/'latest_validation',target)
                atomic_json(a.output/'best_metrics.json',{'cursor':dict(cursor),**report})
            atomic_json(a.output/'latest_metrics.json',{'cursor':dict(cursor),**report})
            event(a.output/'metrics.jsonl',{'cursor':dict(cursor),'baseline':baseline,'best_updated':improved,**report})
            save('latest');emit({'phase':'validation','cursor':cursor,'means':report['means'],'minus_m4':report['minus_m4'],'best_updated':improved})
        if world>1:
            values=[best,best_epoch];dist.broadcast_object_list(values,src=0);best,best_epoch=values
        barrier()
    if not a.resume:validate(True)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    initial_probe=model.output.weight.detach().clone()
    ended=False
    while cursor['epoch']<a.epochs and not ended:
        ep=cursor['epoch'];ids=list(splits['train']);random.Random(2026+ep).shuffle(ids)
        # Pad only the final global update; log duplicates and do not drop samples.
        original_count=len(ids);padding=updates*effective-original_count
        ids+=ids[:padding]
        for gi in range(cursor['batch'],updates):
            if STOP or (a.max_steps and cursor['step']>=a.max_steps):ended=True;break
            group=ids[gi*effective:(gi+1)*effective]
            local_ids=group[rank::world]
            optimizer.zero_grad(set_to_none=True);wrapped.train();loss_sums={};last_delta=0.
            for mi,sid in enumerate(local_ids):
                c,image,gt=cache_sample(a,sid,device)
                sync=wrapped.no_sync() if world>1 and mi<len(local_ids)-1 else contextlib.nullcontext()
                with sync:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        z,delta=wrapped(c,image);pred=decode(vae,z);loss,terms=loss_fn(pred,gt,delta)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    (loss/len(local_ids)).backward()
                for k,v in terms.items():loss_sums[k]=loss_sums.get(k,0)+float(v.detach())/len(local_ids)
                last_delta=float(delta.detach().abs().mean())
                del c,image,gt,z,delta,pred,loss,terms
            lo_grad=sum(float(p.grad.detach().float().square().sum()) for n,p in model.named_parameters()
                        if p.grad is not None and (n.endswith('.a') or n.endswith('.b')))**.5
            grad=float(torch.nn.utils.clip_grad_norm_(params,1.))
            if not math.isfinite(grad) or grad<=0:raise RuntimeError('Invalid restoration gradient')
            if any(p.grad is not None for p in vae.parameters()):raise RuntimeError('Frozen VAE gradient leak')
            if any(p.grad is not None for n,p in model.named_parameters() if '.base.' in n):raise RuntimeError('Frozen tail gradient leak')
            optimizer.step();scheduler.step()
            cursor={'epoch':ep,'batch':gi+1,'step':cursor['step']+1}
            if cursor['step']==1 and torch.equal(initial_probe,model.output.weight):raise RuntimeError('Output parameters did not change')
            if cursor['step']==3 and lo_grad<=0:raise RuntimeError('No gradient to new tail LoRA after zero-head warmup')
            stats=torch.tensor([*loss_sums.values(),grad,lo_grad,last_delta,torch.cuda.max_memory_allocated()/2**30],device=device)
            if world>1:dist.all_reduce(stats);stats/=world
            boundary=cursor['batch']==updates
            if boundary:cursor.update(epoch=ep+1,batch=0)
            if rank==0:
                row={'cursor':dict(cursor),'global_ids':group,'rank0_ids':local_ids,'loss':dict(zip(loss_sums,stats[:len(loss_sums)].tolist())),
                    'gradient_norm':float(stats[-4]),'new_lora_gradient_norm':float(stats[-3]),'delta_mean':float(stats[-2]),
                    'mean_rank_peak_gib':float(stats[-1]),'rank0_peak_gib':torch.cuda.max_memory_allocated()/2**30,
                    'lr':optimizer.param_groups[0]['lr'],'elapsed_seconds':time.monotonic()-started,
                    'frozen_vae_and_bases_no_grad':True,'world_size':world,'epoch_pad_samples':padding}
                event(a.output/'train.jsonl',row)
                if cursor['step']<=5 or cursor['step']%5==0:emit({'phase':'train',**row})
                if cursor['step']==5 or cursor['step']%10==0:save('latest')
            if boundary:validate()
            if STOP or (a.max_steps and cursor['step']>=a.max_steps):ended=True;break
    barrier()
    if rank==0:
        save('latest');atomic_json(a.output/'training_summary.json',{'status':'complete' if cursor['epoch']==a.epochs else 'paused',
            'cursor':cursor,'best_epoch':best_epoch,'best_validation_l1':best,'test_evaluated':False,'early_stopping':False})
        emit({'phase':'finished','cursor':cursor,'best_epoch':best_epoch})
    if world>1:dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['extract','finalize','train'])
    for k in ('data','cache','initial'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--max-steps',type=int,default=0)
    p.add_argument('--rank',type=int,default=8);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--accumulation',type=int,default=2);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if min(a.epochs,a.rank,a.accumulation,a.shards)<1 or not 0<=a.shard<a.shards:raise ValueError('Invalid budget/shard')
    if a.phase=='extract':cache_extract(a)
    elif a.phase=='finalize':cache_finalize(a)
    else:
        if a.output is None:raise ValueError('--output required')
        train(a)

if __name__=='__main__':main()
