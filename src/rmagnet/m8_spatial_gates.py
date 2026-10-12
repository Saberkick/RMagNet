"""M8: freeze M4 and train spatial residual gates at blocks 14, 37, 54.

Three-GPU model parallelism, no frozen post-gate cache, no GT conditioning.
Pinned Qwen block implementation is copied only to expose its two residuals.
"""
import argparse,contextlib,csv,inspect,json,math,os,random,signal,time,types
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import safetensors.torch as st
from safetensors import safe_open
from diffusers import BitsAndBytesConfig,QwenImageTransformer2DModel
from peft import LoraConfig
from .qwen_backend import QwenSharedBackend,check_snapshots,load_upstream
from .m1b_train import load_initial
from .qwen_layer_probe import deterministic_encode
from .sma_data import load_manifest
from .m5_train import read,save_png,metrics,KEYS
from .m7_latent_tail import EXPECTED_M4,sha,atomic_json,atomic_tensors,emit,event,decode,charbonnier,gradient_loss

VERSION='m8-frozen-m4-spatial-gates-14-37-54-v1'
LAYERS=(14,37,54)
STOP=False

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(3072);self.project=nn.Linear(3072,64)
        self.spatial=nn.Sequential(nn.Conv2d(64,64,3,padding=1,groups=64),nn.SiLU())
        self.output=nn.Conv2d(64,2,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.last=None
    def forward(self,x,hw):
        h,w=hw
        if x.shape!=(1,h*w,3072):raise RuntimeError('Unexpected image token grid')
        with torch.autocast('cuda',enabled=False):
            f=F.silu(self.project(self.norm(x.float()))).transpose(1,2).reshape(1,64,h,w)
            g=.75+.5*torch.sigmoid(self.output(self.spatial(f)))
        self.last=g
        return g[:,0].flatten(1).unsqueeze(-1),g[:,1].flatten(1).unsqueeze(-1)

def gated_block(block,hidden_states,encoder_hidden_states,encoder_hidden_states_mask,
                temb,image_rotary_emb=None,joint_attention_kwargs=None):
    owner=block._m8_owner
    if not owner.enabled:
        return block._m8_native(hidden_states,encoder_hidden_states,encoder_hidden_states_mask,
                                temb,image_rotary_emb,joint_attention_kwargs)
    ga,gm=owner.gates[str(block._m8_layer)](hidden_states,owner.grid)
    img_mod1,img_mod2=block.img_mod(temb).chunk(2,-1)
    txt_mod1,txt_mod2=block.txt_mod(temb).chunk(2,-1)
    im,ig=block._modulate(block.img_norm1(hidden_states),img_mod1)
    tx,tg=block._modulate(block.txt_norm1(encoder_hidden_states),txt_mod1)
    ai,at=block.attn(hidden_states=im,encoder_hidden_states=tx,
        encoder_hidden_states_mask=encoder_hidden_states_mask,image_rotary_emb=image_rotary_emb,
        **(joint_attention_kwargs or {}))
    # Preserve native BF16 residual rounding; gates themselves remain FP32.
    update=ig*ai
    hidden_states=(hidden_states.float()+ga*update.float()).to(hidden_states.dtype)
    encoder_hidden_states=encoder_hidden_states+tg*at
    im,ig=block._modulate(block.img_norm2(hidden_states),img_mod2)
    update=ig*block.img_mlp(im)
    hidden_states=(hidden_states.float()+gm*update.float()).to(hidden_states.dtype)
    tx,tg=block._modulate(block.txt_norm2(encoder_hidden_states),txt_mod2)
    encoder_hidden_states=encoder_hidden_states+tg*block.txt_mlp(tx)
    if encoder_hidden_states.dtype==torch.float16:encoder_hidden_states=encoder_hidden_states.clip(-65504,65504)
    if hidden_states.dtype==torch.float16:hidden_states=hidden_states.clip(-65504,65504)
    return encoder_hidden_states,hidden_states

class FrozenM4Gates:
    def __init__(self,initial,cards):
        base,lora=check_snapshots();up=load_upstream();self.primary=torch.device('cuda:0')
        self.cards=cards;self.enabled=True;self.grid=None
        mapping={'pos_embed':0,'time_text_embed':0,'txt_norm':0,'img_in':0,'txt_in':0,
                 'norm_out':cards-1,'proj_out':cards-1}
        for i in range(60):mapping[f'transformer_blocks.{i}']=min(cards-1,i*cards//60)
        nf4=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,
                              llm_int8_skip_modules=['transformer_blocks.0.img_mod'])
        transformer=QwenImageTransformer2DModel.from_pretrained(str(base),subfolder='transformer',
            torch_dtype=torch.bfloat16,quantization_config=nf4,device_map=mapping,local_files_only=True)
        transformer.add_adapter(LoraConfig.from_pretrained(str(lora),subfolder='transformer_lora'),adapter_name='default')
        state=st.load_file(lora/'transformer_lora/pytorch_lora_weights.safetensors')
        _,extra=transformer.load_state_dict(state,strict=False)
        if extra:raise RuntimeError('Unexpected official adapter keys')
        del state
        vae=up.load_qwen_vae(str(base),self.primary)
        embeds=st.load_file(lora/'text_embeddings/state_dict.safetensors')
        resolution=json.loads((lora/'model_index.json').read_text())['processing_resolution']
        self.backend=QwenSharedBackend(vae,transformer,embeds,resolution,up)
        self.backend.set_trainable_branch('transmission');load_initial(self.backend,initial,self.primary)
        self.backend.set_trainable_branch(None);self.backend.eval()
        if any(p.requires_grad for p in self.backend.parameters()):raise RuntimeError('M4/VAE must be frozen')
        transformer.enable_gradient_checkpointing()
        self.mapping=mapping;self.gates=nn.ModuleDict()
        for layer in LAYERS:
            block=transformer.transformer_blocks[layer-1]
            if 'zero_cond_t' in inspect.signature(type(block).forward).parameters:
                raise RuntimeError('Unexpected Qwen version: inspect pinned block before patching')
            device=next(block.parameters()).device
            self.gates[str(layer)]=SpatialGate().to(device)
            # Avoid registering owner recursively as a module; it is a plain controller.
            block._m8_owner=self;block._m8_layer=layer
            if hasattr(block,'_old_forward'):
                block._m8_native=block._old_forward
                block._old_forward=types.MethodType(gated_block,block)
            else:
                block._m8_native=block.forward;block.forward=types.MethodType(gated_block,block)

    def clear(self):
        for gate in self.gates.values():gate.last=None

    def forward(self,image):
        self.clear();image=image.to(self.primary)
        self.grid=(image.shape[-2]//16,image.shape[-1]//16)
        edited=self.native_latent(image)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=decode(self.backend.vae,edited)
        return pred,edited

    def native_latent(self,image):
        """Whole native transformer forward; frozen encoder outside autocast."""
        b=self.backend;t=b.transformer;image=image.to(self.primary);h,w=image.shape[-2:]
        with torch.no_grad():z=deterministic_encode(b,image*2-1)
        packed=b.upstream.QwenImageEditPipeline._pack_latents(z[:,:,0],1,16,h//8,w//8).to(torch.bfloat16)
        prompt=b.embeddings['prompt_embeds'].to(self.primary,dtype=torch.bfloat16)
        mask=b.embeddings['prompt_mask'].to(self.primary)
        ts=torch.full((1,),float(499),device=self.primary,dtype=torch.bfloat16)/1000
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=t(hidden_states=packed,timestep=ts,encoder_hidden_states=prompt,
                encoder_hidden_states_mask=mask,img_shapes=[[(1,h//16,w//16)]],
                txt_seq_lens=mask.sum(1).tolist(),guidance=None,attention_kwargs={},return_dict=False)[0].to(self.primary)
            velocity=b.upstream.QwenImageEditPipeline._unpack_latents(out,height=h,width=w,vae_scale_factor=8)
        return z-velocity

    def regularizer(self):
        values=[(g.last-1).square().mean().to(self.primary) for g in self.gates.values()]
        if len(values)!=3:raise RuntimeError('Missing gate values')
        return sum(values)/3

    def stats(self):
        result={}
        for name,g in self.gates.items():
            if g.last is None:raise RuntimeError('Missing gate '+name)
            v=g.last.detach()
            result[name]={'min':float(v.min()),'max':float(v.max()),'mean':float(v.mean()),
                'mean_abs_change':float((v-1).abs().mean()),'spatial_std':float(v.std()),
                'near_bound_fraction':float(((v<.76)|(v>1.24)).float().mean())}
        return result

def pixels(a,sid,device):
    return read(a.data/'blended'/f'{sid}.png').to(device),read(a.data/'transmission_layer'/f'{sid}.png').to(device)

def gate_previews(runner,out,sid,size):
    import numpy as np
    from PIL import Image
    for name,g in runner.gates.items():
        for k,label in enumerate(('attention','mlp')):
            # Fixed grayscale: gate .75 black, 1 middle gray, 1.25 white.
            v=((g.last[0,k].detach().float().cpu()-.75)/.5).clamp(0,1)
            p=out/'gates'/f'{sid}_L{name}_{label}.png';p.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray((v*255).round().byte().numpy()).resize(size,Image.Resampling.NEAREST).save(p)

@torch.no_grad()
def evaluate(a,runner,ids,out,baseline=None):
    out.mkdir(parents=True,exist_ok=True);rows=[];runner.gates.eval()
    for sid in ids:
        image,gt=pixels(a,sid,runner.primary);pred,_=runner.forward(image)
        if not torch.isfinite(pred).all():raise RuntimeError('Nonfinite validation')
        path=out/'predictions'/f'{sid}.png';save_png(pred,path)
        rows.append({'id':sid,**metrics(read(path),gt.cpu(),image.cpu())})
        if runner.enabled and sid in ids[:3]:gate_previews(runner,out,sid,(image.shape[-1],image.shape[-2]))
        runner.clear()
    means={k:sum(v[k] for v in rows)/len(rows) for k in KEYS}
    report={'count':len(rows),'means':means,'per_image':rows,'split':'validation','RGB8':True}
    if baseline:report['minus_m4']={k:means[k]-baseline['means'][k] for k in KEYS}
    atomic_json(out/'metrics.json',report)
    with open(out/'metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return report

def stop_request(*_):
    global STOP;STOP=True

def main():
    p=argparse.ArgumentParser()
    for name in ('data','initial','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--accumulation',type=int,default=6)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--gate-weight',type=float,default=.01)
    p.add_argument('--max-steps',type=int,default=0);p.add_argument('--resume',action='store_true')
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(2026);random.seed(2026)
    cards=torch.cuda.device_count()
    if cards not in (2,3,4) or min(a.epochs,a.accumulation)<1:raise ValueError('Use 2-4 GPUs and positive budgets')
    if sha(a.initial)!=EXPECTED_M4:raise RuntimeError('Wrong M4 final best')
    _,records,splits=load_manifest(a.data)
    ident={'version':VERSION,'m4_sha256':EXPECTED_M4,'dataset_sha256':sha(a.data/'manifest.json'),'layers':list(LAYERS)}
    if (a.output/'config.json').exists() and not a.resume:raise FileExistsError('Use --resume or a new output')
    a.output.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGINT,stop_request);signal.signal(signal.SIGTERM,stop_request)
    torch.cuda.set_device(0);runner=FrozenM4Gates(a.initial,cards)
    params=list(runner.gates.parameters());opt=torch.optim.AdamW(params,lr=a.lr,weight_decay=1e-4)
    updates=math.ceil(len(splits['train'])/a.accumulation)
    def factor(s):
        if s<10:return (s+1)/10
        return .1+.9*.5*(1+math.cos(math.pi*min(1,(s-10)/max(1,updates*a.epochs-10))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,factor)
    cursor={'epoch':0,'batch':0,'step':0};best=float('inf');best_epoch=0
    def save(which):
        dest=a.output/which;dest.mkdir(parents=True,exist_ok=True)
        atomic_tensors(dest/'gates.safetensors',runner.gates.state_dict(),{'identity':json.dumps(ident),'cursor':json.dumps(cursor)})
        atomic_json(dest/'state.json',{'cursor':dict(cursor),'best_epoch':best_epoch,'best_l1':best,'identity':ident})
        if which=='latest':
            tmp=dest/'trainer.pt.tmp'
            torch.save({'identity':ident,'cursor':dict(cursor),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                'best':best,'best_epoch':best_epoch,'accumulation':a.accumulation,'cards':cards},tmp);tmp.replace(dest/'trainer.pt')
    if a.resume:
        state=torch.load(a.output/'latest/trainer.pt',map_location='cpu',weights_only=False)
        with safe_open(a.output/'latest/gates.safetensors',framework='pt') as f:md=f.metadata()
        if json.loads(md['cursor'])!=state['cursor'] or json.loads(md['identity'])!=ident:raise RuntimeError('Inconsistent latest checkpoint')
        if state['identity']!=ident or state['cards']!=cards or state['accumulation']!=a.accumulation:raise RuntimeError('Resume identity mismatch')
        runner.gates.load_state_dict(st.load_file(a.output/'latest/gates.safetensors'),strict=True)
        opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        for param,values in opt.state.items():
            for k,v in values.items():
                if isinstance(v,torch.Tensor) and k!='step':values[k]=v.to(param.device)
        cursor=state['cursor'];best=state['best'];best_epoch=state['best_epoch'];del state
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(identity=ident,model_parallel_cards=cards,physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),
        device_map=runner.mapping,trainable_parameters=sum(x.numel() for x in params),train_count=len(splits['train']),
        validation_count=len(splits['validation']),test_count=len(splits['test']),updates_per_epoch=updates,planned_steps=updates*a.epochs,
        precision='NF4 frozen DiT; BF16 activations; FP32 gates and optimizer',effective_batch=a.accumulation,
        gate_range=[.75,1.25],gate_initial=1,early_stopping=False,loss={'charbonnier':1.,'edge':.1,'gate':a.gate_weight},
        original_lora_frozen=True,vae_frozen=True,input='ordinary I only',gt_usage='loss and evaluation only',
        downstream_cache=False,augmentation=False,retention='best; latest model+optimizer',selection='minimum validation RGB8 L1, min_delta=1e-8')
    atomic_json(a.output/'config.json',config);emit({'phase':'loaded','trainable_parameters':config['trainable_parameters'],'cards':cards})
    if not a.resume:
        # Use a large, actual train sample for numerical identity verification.
        sid=max(splits['train'],key=lambda s:math.prod(records[s]['target_size']))
        image,_=pixels(a,sid,runner.primary)
        with torch.no_grad():
            runner.enabled=False;z_native=runner.native_latent(image);_,z0=runner.forward(image)
            native_error=float((z_native-z0).abs().max())
            if native_error!=0:raise RuntimeError('Manual model-parallel forward differs from native M4: '+str(native_error))
            runner.enabled=True;_,z1=runner.forward(image)
            error=float((z0-z1).abs().max())
            if error!=0:raise RuntimeError('Zero gates do not preserve native M4: '+str(error))
            stats=runner.stats()
            if any(v['min']!=1 or v['max']!=1 for v in stats.values()):raise RuntimeError('Bad gate initialization')
        runner.clear();del image,z_native,z0,z1
        atomic_json(a.output/'identity_check.json',{'id':sid,'max_latent_error':error,'native_forward_max_latent_error':native_error,'gates':stats})
        emit({'phase':'identity_pass','id':sid,'max_latent_error':error,'native_forward_max_latent_error':native_error})
        runner.enabled=False;baseline=evaluate(a,runner,splits['validation'],a.output/'m4_baseline_validation')
        runner.enabled=True;best=baseline['means']['l1'];save('best');save('latest')
        atomic_json(a.output/'best_metrics.json',{'cursor':dict(cursor),**baseline})
        event(a.output/'metrics.jsonl',{'cursor':dict(cursor),'baseline':True,**baseline})
        emit({'phase':'baseline','means':baseline['means']})
    else:baseline=json.loads((a.output/'m4_baseline_validation/metrics.json').read_text())
    started=time.monotonic()
    for d in range(cards):torch.cuda.reset_peak_memory_stats(d)
    initial_outputs={name:g.output.weight.detach().clone() for name,g in runner.gates.items()}
    stopped=False
    while cursor['epoch']<a.epochs and not stopped:
        ep=cursor['epoch'];ids=list(splits['train']);random.Random(2026+ep).shuffle(ids)
        for gi in range(cursor['batch'],updates):
            if STOP or (a.max_steps and cursor['step']>=a.max_steps):stopped=True;break
            group=ids[gi*a.accumulation:(gi+1)*a.accumulation]
            opt.zero_grad(set_to_none=True);runner.gates.train();sums={};gate_stats={}
            for sid in group:
                image,gt=pixels(a,sid,runner.primary);pred,_=runner.forward(image)
                rec=charbonnier(pred,gt);edge=gradient_loss(pred,gt);reg=runner.regularizer()
                loss=rec+.1*edge+a.gate_weight*reg
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite gate loss')
                gate_stats=runner.stats();(loss/len(group)).backward()
                for k,v in {'rec':rec,'edge':edge,'gate_regularizer':reg,'total':loss}.items():sums[k]=sums.get(k,0)+float(v.detach())/len(group)
                runner.clear();del image,gt,pred,rec,edge,reg,loss
            gradients={name:sum(float(p.grad.detach().float().square().sum()) for p in g.parameters() if p.grad is not None)**.5
                       for name,g in runner.gates.items()}
            # torch clip_grad_norm supports parameters on multiple devices.
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
            if not math.isfinite(norm) or norm<=0 or any(v<=0 for v in gradients.values()):raise RuntimeError('Missing/nonfinite gate gradient')
            if any(p.grad is not None for p in runner.backend.parameters()):raise RuntimeError('Frozen M4/VAE gradient leak')
            opt.step();scheduler.step();cursor={'epoch':ep,'batch':gi+1,'step':cursor['step']+1}
            if cursor['step']==1 and any(torch.equal(initial_outputs[n],g.output.weight) for n,g in runner.gates.items()):raise RuntimeError('Gate parameters did not change')
            boundary=cursor['batch']==updates
            if boundary:cursor.update(epoch=ep+1,batch=0)
            row={'cursor':dict(cursor),'ids':group,'loss':sums,'gradient_norm':norm,'layer_gradient_norms':gradients,
                'gate_stats_last_sample':gate_stats,'frozen_no_grad':True,'peak_gib_by_device':{str(d):torch.cuda.max_memory_allocated(d)/2**30 for d in range(cards)},
                'lr':opt.param_groups[0]['lr'],'elapsed_seconds':time.monotonic()-started}
            event(a.output/'train.jsonl',row)
            if cursor['step']<=5 or cursor['step']%5==0:emit({'phase':'train',**row})
            if cursor['step']<=5 or cursor['step']%10==0:save('latest')
            if boundary:
                report=evaluate(a,runner,splits['validation'],a.output/'latest_validation',baseline)
                improved=report['means']['l1']<best-1e-8
                if improved:
                    best=report['means']['l1'];best_epoch=cursor['epoch'];save('best')
                    import shutil
                    dest=a.output/'best_validation'
                    if dest.exists():shutil.rmtree(dest)
                    shutil.copytree(a.output/'latest_validation',dest)
                    atomic_json(a.output/'best_metrics.json',{'cursor':dict(cursor),**report})
                atomic_json(a.output/'latest_metrics.json',{'cursor':dict(cursor),**report})
                event(a.output/'metrics.jsonl',{'cursor':dict(cursor),'best_updated':improved,**report})
                save('latest');emit({'phase':'validation','cursor':cursor,'means':report['means'],'minus_m4':report['minus_m4'],'best_updated':improved})
        if STOP or (a.max_steps and cursor['step']>=a.max_steps):stopped=True
    save('latest');status='complete' if cursor['epoch']==a.epochs else 'paused'
    atomic_json(a.output/'training_summary.json',{'status':status,'cursor':cursor,'best_epoch':best_epoch,'best_validation_l1':best,'test_evaluated':False})
    emit({'phase':'finished','status':status,'cursor':cursor})

if __name__=='__main__':main()
