"""Fit fixed train-only PCA and pretrain clean spatial memory from I/P90 to GT."""
import argparse,json,random,time
from pathlib import Path
import safetensors.torch
import torch
import torch.nn.functional as F
from .sma import SMA
from .m4_cache import atomic_safetensors,atomic_json,sha256


def normalize(x):
    return F.normalize(x-x.mean(1,keepdim=True),dim=-1,eps=1e-6)


def loss(m,y,grid,scale):
    a,b=normalize(m),normalize(y)
    content=(1-(a*b).sum(-1)).mean()
    h,w=grid
    a,b=a.reshape(1,h,w,-1),b.reshape(1,h,w,-1)
    rel=0.5*(F.smooth_l1_loss((a[:,:,1:]*a[:,:,:-1]).sum(-1),(b[:,:,1:]*b[:,:,:-1]).sum(-1))+
             F.smooth_l1_loss((a[:,1:]*a[:,:-1]).sum(-1),(b[:,1:]*b[:,:-1]).sum(-1)))
    amplitude=F.smooth_l1_loss(m/scale,y/scale)
    return .7*content+.3*rel+.1*amplitude,content,rel,amplitude


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=5);ap.add_argument('--seed',type=int,default=2026)
    args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(args.seed);torch.cuda.set_device(0)
    cm=json.loads((args.cache_root/'manifest.json').read_text())
    if not cm['complete'] or len(cm['samples'])!=144: raise RuntimeError('Unfinalized cache')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'memory.safetensors').exists(): raise FileExistsError('Pretraining output exists')
    model=SMA().cuda(); sampled=[];raw=[]
    for rec in cm['samples']:
        path=args.cache_root/rec['cache']
        if sha256(path)!=rec['cache_sha256']:raise RuntimeError('Cache checksum mismatch')
        s=safetensors.torch.load_file(path)
        grid=tuple(int(x) for x in s['token_grid_hw'])
        triple=[]
        for key in ('q37_input','q37_p90','q37_gt'):
            x=F.layer_norm(s[key].float(),(3072,))
            triple.append(x)
            indices=torch.linspace(0,x.shape[0]-1,32).long()
            sampled.append(x[indices])
        raw.append((rec['id'],grid,triple))
    x=torch.cat(sampled).cuda();mean=x.mean(0);x=x-mean
    # All fit data are training split, including GT. Fixed shared target basis.
    _,singular,v=torch.pca_lowrank(x,q=256,center=False,niter=3)
    model.projection.mean.copy_(mean);model.projection.basis.copy_(v)
    model.projection.scale.copy_((singular/(x.shape[0]-1)**.5).clamp_min(.01))
    del x,sampled
    projected=[]
    for sid,grid,triple in raw:
        z=[((t.cuda()-mean)@v).unsqueeze(0).detach() for t in triple]
        projected.append((sid,grid,z))
    del raw
    opt=torch.optim.AdamW(model.memory.parameters(),lr=1e-4,weight_decay=.01)
    started=time.monotonic();events=[]
    with torch.no_grad():
        identity=sum(float(loss(z[0],z[2],g,model.projection.scale)[0]+loss(z[1],z[2],g,model.projection.scale)[0])*.5 for _,g,z in projected)/len(projected)
    for epoch in range(args.epochs):
        order=list(range(len(projected)));random.Random(args.seed+epoch).shuffle(order);values=[]
        for index in order:
            sid,grid,z=projected[index];opt.zero_grad(set_to_none=True)
            total=0
            # Two observations, one clean target. No pure R label assumed.
            for source in z[:2]:
                m=model.memory(source,grid)
                l,*_=loss(m,z[2],grid,model.projection.scale)
                (l*.5).backward();total+=float(l.detach())*.5
            norm=torch.nn.utils.clip_grad_norm_(model.memory.parameters(),1.0)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite pretraining gradient')
            opt.step();values.append(total)
        event={'phase':'memory_pretrain','epoch':epoch+1,'loss':sum(values)/len(values),'identity_loss':identity,'elapsed_seconds':time.monotonic()-started}
        events.append(event);print(json.dumps(event),flush=True)
    model.memory.eval()
    with torch.no_grad():
        trained=sum(float(loss(model.memory(z[0],g),z[2],g,model.projection.scale)[0]+loss(model.memory(z[1],g),z[2],g,model.projection.scale)[0])*.5 for _,g,z in projected)/len(projected)
    atomic_safetensors(args.output/'memory.safetensors',model.state_dict(),{'experiment':'SMA','teacher_sha256':cm['teacher']['adapter_sha256'],'cache_manifest_sha256':sha256(args.cache_root/'manifest.json')})
    atomic_json(args.output/'report.json',{'epochs':args.epochs,'train_images':144,'optimization_updates':args.epochs*144,'identity_train_loss':identity,'trained_train_loss':trained,'train_improvement':identity-trained,'validation_claim':'none: feature pretraining reports training fit only; final generator selects on 18 validation images','events':events,'label_noise':cm.get('label_noise'),'cache_manifest_sha256':sha256(args.cache_root/'manifest.json'),'teacher_sha256':cm['teacher']['adapter_sha256'],'memory_sha256':sha256(args.output/'memory.safetensors')})
    print(json.dumps({'phase':'memory_pretrain_complete','identity_loss':identity,'trained_loss':trained}),flush=True)


if __name__=='__main__':main()
