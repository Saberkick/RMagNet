"""Offline C1 + M5-R training and saved-PNG evaluation, single GPU."""
import argparse,csv,json,math,random,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import safetensors.torch as st
from safetensors import safe_open
from .m5_pixel import PixelRefiner,VERSION,losses,ssim,edge
from .m5_cache import CACHE,DATA,C1
from .m4_cache import sha256
from .sma_data import load_manifest

KEYS=('l1','psnr','ssim','edge_l1','low_change_keep_l1','high_change_restore_l1')

def read(path):
    with Image.open(path) as im:a=np.asarray(im.convert('RGB'),dtype=np.float32).copy()
    return torch.from_numpy(a).permute(2,0,1).unsqueeze(0)/255

def save_png(x,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    a=x[0].detach().float().clamp(0,1).mul(255).round().byte().permute(1,2,0).cpu().numpy()
    Image.fromarray(a).save(path)

def atomic_json(path,obj):
    p=path.with_suffix('.tmp');p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');p.replace(path)

def metrics(x,gt,image):
    err=(x-gt).abs().mean(1,keepdim=True);change=(image-gt).abs().mean(1,keepdim=True)
    lo=change<=torch.quantile(change.flatten(),.25);hi=change>=torch.quantile(change.flatten(),.75)
    return {'l1':float(F.l1_loss(x,gt)),'psnr':float(-10*torch.log10(F.mse_loss(x,gt).clamp_min(1e-12))),
            'ssim':float(ssim(x,gt)),'edge_l1':float(edge(x,gt)),
            'low_change_keep_l1':float(err[lo].mean()),'high_change_restore_l1':float(err[hi].mean())}

def audit(data,cache):
    _,records,splits=load_manifest(data);manifest=json.loads((cache/'manifest.json').read_text())
    if not manifest['complete'] or manifest['version']!='m5-c1-rgb8-v1':raise RuntimeError('Incomplete/wrong cache')
    if manifest['dataset_manifest_sha256']!=sha256(data/'manifest.json') or manifest['checkpoint_sha256']!=sha256(C1):raise RuntimeError('Cache source changed')
    if set(manifest['samples'])!=set(records):raise RuntimeError('Cache ID mismatch')
    for sid,r in records.items():
        c=manifest['samples'][sid];p=cache/'predictions'/f'{sid}.png'
        if c['input_sha256']!=r['processed']['input']['sha256'] or c['gt_sha256']!=r['processed']['gt']['sha256'] or sha256(p)!=c['prediction_sha256']:raise RuntimeError('Cache/data mismatch '+sid)
        with Image.open(p) as im:
            if im.mode!='RGB' or list(im.size)!=r['target_size']:raise RuntimeError('Cache geometry '+sid)
    return records,splits,manifest

def batch(data,cache,sid,device):
    return tuple(read(p).to(device) for p in (data/'blended'/f'{sid}.png',cache/'predictions'/f'{sid}.png',data/'transmission_layer'/f'{sid}.png'))

def save_model(path,model,identity,epoch,step):
    tmp=path.with_suffix('.tmp');st.save_file({k:v.detach().contiguous().cpu() for k,v in model.state_dict().items()},tmp,
        metadata={'architecture':VERSION,'width':str(model.width),'upstream_sha256':identity['checkpoint_sha256'],
                  'dataset_manifest_sha256':identity['dataset_manifest_sha256'],'epoch':str(epoch),'step':str(step)})
    tmp.replace(path)

def load_model(path,identity,device):
    with safe_open(path,framework='pt') as h:md=h.metadata()
    if md['architecture']!=VERSION or md['upstream_sha256']!=identity['checkpoint_sha256'] or md['dataset_manifest_sha256']!=identity['dataset_manifest_sha256']:raise RuntimeError('Checkpoint/cache identity mismatch')
    m=PixelRefiner(int(md['width'])).to(device);m.load_state_dict(st.load_file(path,device=str(device)),strict=True);return m,md

@torch.no_grad()
def evaluate(model,data,cache,ids,device,out,records):
    model.eval();rows=[];base=[];out.mkdir(parents=True,exist_ok=True)
    for sid in ids:
        image,t0,gt=batch(data,cache,sid,device);r=model(image,t0);p=out/'predictions'/f'{sid}.png';save_png(r['prediction'],p)
        pred=read(p);gt=gt.cpu();image=image.cpu();t0=t0.cpu()
        rows.append({'id':sid,'source':records[sid]['dataset_source'],**metrics(pred,gt,image)})
        base.append({'id':sid,**metrics(t0,gt,image)})
    means={k:sum(r[k] for r in rows)/len(rows) for k in KEYS};bm={k:sum(r[k] for r in base)/len(base) for k in KEYS}
    report={'count':len(rows),'means':means,'c1_baseline':bm,'minus_c1':{k:means[k]-bm[k] for k in KEYS},'per_image':rows,
            'psnr_wins':sum(r['psnr']>b['psnr'] for r,b in zip(rows,base)),'metric_domain':'macro over saved RGB8 PNG; unchanged processed aspect ratio',
            'ssim_definition':'11x11 uniform window, stride1, zero-pad5; project convention, not Gaussian-window SSIM'}
    atomic_json(out/'evaluation.json',report)
    with (out/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return report

def probe(model,data,cache,ids,device,out,identity,records):
    # Independent optimizer; formal model is reconstructed after this check.
    selected=sorted(ids,key=lambda x:records[x]['target_size'][0]*records[x]['target_size'][1],reverse=True)[:2]
    selected+= [min(ids,key=lambda x:records[x]['target_size'][0]/records[x]['target_size'][1]),max(ids,key=lambda x:records[x]['target_size'][0]/records[x]['target_size'][1])]
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4);torch.cuda.reset_peak_memory_stats()
    grads=[];zero_max=0.;before={k:v.detach().clone() for k,v in model.state_dict().items()}
    for n,sid in enumerate(dict.fromkeys(selected)):
        image,t0,gt=batch(data,cache,sid,device)
        if n==0:
            with torch.no_grad():zero_max=float((model(image,t0)['prediction']-t0).abs().max())
            if zero_max!=0:raise RuntimeError('Zero initialization failed')
        model.train();opt.zero_grad(set_to_none=True);result=model(image,t0);loss,terms=losses(result,t0,gt)
        if not all(torch.isfinite(v).all() for v in [loss,*terms.values(),result['delta'],*result['bands']]):raise RuntimeError('Nonfinite probe')
        loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0));grads.append(grad)
        if not math.isfinite(grad) or grad<=0:raise RuntimeError('No valid pixel gradients')
        opt.step()
    changed=sum(not torch.equal(v,before[k]) for k,v in model.state_dict().items())
    if not changed:raise RuntimeError('Parameters unchanged')
    temp=out/'probe.safetensors';save_model(temp,model,identity,0,0);loaded,_=load_model(temp,identity,device)
    model.eval();loaded.eval()
    with torch.no_grad():
        a=model(image,t0)['prediction'];b=loaded(image,t0)['prediction']
        if not torch.equal(a,b):raise RuntimeError('Checkpoint reload mismatch')
    png=out/'probe.png';save_png(b,png);mm=metrics(read(png),gt.cpu(),image.cpu())
    peak=torch.cuda.max_memory_allocated()/2**30
    summary={'status':'passed','tested_ids':list(dict.fromkeys(selected)),'zero_initialization_max_error':zero_max,'gradient_norms':grads,
             'changed_state_tensors':changed,'parameters':sum(v.numel() for v in model.parameters()),'peak_allocated_gib':peak,
             'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'checkpoint_reload_exact':True,'saved_png_metrics':mm,
             'upstream_loaded_in_training':False,'gt_never_passed_to_c1':True,'gt_identity_is_training_only_auxiliary_input':True}
    atomic_json(out/'preflight.json',summary);temp.unlink();png.unlink();print(json.dumps(summary),flush=True)
    if peak>18:raise RuntimeError('VRAM budget exceeded; no formal training started')
    del loaded,before,opt;torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=DATA);p.add_argument('--cache',type=Path,default=CACHE)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=30);p.add_argument('--patience',type=int,default=4)
    p.add_argument('--accumulation',type=int,default=4);p.add_argument('--min-epochs',type=int,default=5);p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--split',choices=['validation','test'],default='test');p.add_argument('--probe-only',action='store_true')
    a=p.parse_args()
    if a.epochs<1 or a.accumulation<1 or a.patience<1:raise ValueError('Positive budgets required')
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError(a.output)
    a.output.mkdir(parents=True);records,splits,identity=audit(a.data,a.cache)
    torch.set_num_threads(1);torch.cuda.set_device(0);device=torch.device('cuda:0');torch.manual_seed(2026);random.seed(2026)
    if a.checkpoint:
        model,md=load_model(a.checkpoint,identity,device)
        report=evaluate(model,a.data,a.cache,splits[a.split],device,a.output,records)
        report.update(checkpoint=str(a.checkpoint),checkpoint_sha256=sha256(a.checkpoint),checkpoint_metadata=md,split=a.split)
        atomic_json(a.output/'evaluation.json',report);print(json.dumps(report['means']),flush=True);return
    model=PixelRefiner().to(device);probe(model,a.data,a.cache,splits['train'],device,a.output,identity,records)
    if a.probe_only:return
    # Never start from probe-updated weights.
    del model;torch.cuda.empty_cache();torch.manual_seed(2026);model=PixelRefiner().to(device)
    updates=math.ceil(len(splits['train'])/a.accumulation)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(architecture=VERSION,upstream_checkpoint=str(C1),upstream_sha256=identity['checkpoint_sha256'],dataset_manifest_sha256=identity['dataset_manifest_sha256'],
        train_samples=len(splits['train']),validation_samples=len(splits['validation']),test_samples=len(splits['test']),updates_per_epoch=updates,
        trainable_parameters=sum(v.numel() for v in model.parameters()),precision='FP32',effective_batch=a.accumulation,backbone_frozen='offline C1, never loaded',
        selection='minimum saved-RGB8 validation macro L1; epoch0 baseline eligible',optimizer='AdamW weight_decay=1e-4',identity_weight=.05,identity_every_samples=10,
        scheduler='one epoch linear warmup then cosine to 0.1*LR',test_during_training=False,save_optimizer=False)
    atomic_json(a.output/'config.json',config);start=time.time()
    baseline=evaluate(model,a.data,a.cache,splits['validation'],device,a.output/'baseline_validation',records)
    best=baseline['means']['l1'];best_epoch=0;bad=0;step=0
    save_model(a.output/'best.safetensors',model,identity,0,0);atomic_json(a.output/'best_metrics.json',{'epoch':0,**baseline})
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    def factor(s):
        if s<updates:return (s+1)/updates
        return .1+.9*.5*(1+math.cos(math.pi*min(1,(s-updates)/max(1,(a.epochs-1)*updates))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,a.epochs+1):
        model.train();ids=list(splits['train']);random.Random(2026+epoch).shuffle(ids);sums={};epoch_start=time.time()
        for offset in range(0,len(ids),a.accumulation):
            group=ids[offset:offset+a.accumulation];optimizer.zero_grad(set_to_none=True)
            for j,sid in enumerate(group):
                image,t0,gt=batch(a.data,a.cache,sid,device)
                if random.random()<.5:image=image.flip(-1);t0=t0.flip(-1);gt=gt.flip(-1)
                result=model(image,t0);loss,terms=losses(result,t0,gt)
                identity_loss=loss.new_zeros(())
                if (offset+j)%10==0:
                    ir=model(gt,gt);identity_loss=ir['delta'].abs().mean();loss=loss+.05*identity_loss
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                (loss/len(group)).backward()
                values={k:float(v.detach()) for k,v in terms.items()};values.update(total=float(loss.detach()),identity=float(identity_loss.detach()),
                    correction_near_limit=float((result['delta'].detach().abs()>=.19).float().mean()))
                for k,v in values.items():sums[k]=sums.get(k,0)+v
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
            if not math.isfinite(grad):raise RuntimeError('Nonfinite gradient')
            optimizer.step();scheduler.step();step+=1
            record={'epoch':epoch,'step':step,'lr':optimizer.param_groups[0]['lr'],'gradient_norm':grad,'loss_last_sample':values,
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'samples_seen_in_epoch':min(offset+len(group),len(ids))}
            with (a.output/'train.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            if step%5==0 or offset==0:print(json.dumps(record),flush=True)
            del image,t0,gt,result,loss,terms
        validation=evaluate(model,a.data,a.cache,splits['validation'],device,a.output/'latest_validation',records)
        score=validation['means']['l1'];improved=score<best-1e-6
        save_model(a.output/'latest.safetensors',model,identity,epoch,step)
        atomic_json(a.output/'latest_metrics.json',{'epoch':epoch,'step':step,**validation})
        if improved:
            best=score;best_epoch=epoch;bad=0;save_model(a.output/'best.safetensors',model,identity,epoch,step)
            atomic_json(a.output/'best_metrics.json',{'epoch':epoch,'step':step,**validation})
            # Overwrite fixed-ID PNGs; no epoch checkpoints/prediction accumulation.
            import shutil
            shutil.copytree(a.output/'latest_validation',a.output/'best_validation',dirs_exist_ok=True)
        else:bad+=1
        er={'epoch':epoch,'step':step,'seconds':time.time()-epoch_start,'train_means':{k:v/len(ids) for k,v in sums.items()},
            'validation':validation['means'],'minus_c1':validation['minus_c1'],'best_epoch':best_epoch,'bad_epochs':bad,'improved':improved}
        with (a.output/'epochs.jsonl').open('a') as f:f.write(json.dumps(er)+'\n')
        print('EPOCH '+json.dumps(er),flush=True)
        if epoch>=a.min_epochs and bad>=a.patience:break
    summary={'status':'complete','completed_epochs':epoch,'steps':step,'best_epoch':best_epoch,'best_validation_l1':best,
             'elapsed_seconds':time.time()-start,'early_stopped':epoch<a.epochs,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
             'best_may_be_identity_baseline':best_epoch==0,'sealed_test_evaluated':False}
    atomic_json(a.output/'training_summary.json',summary)
    (a.output/'training_summary.md').write_text('# M5 on C1\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
    print('COMPLETE '+json.dumps(summary),flush=True)

if __name__=='__main__':main()
