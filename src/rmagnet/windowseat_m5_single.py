"""The unchanged M5 pixel objective with mid-epoch latest-state continuation."""
import argparse,json,math,random,signal,shutil,time
from pathlib import Path
import torch
from . import m5_train as core
from .m5_pixel import PixelRefiner,losses,VERSION
from .c1_l20_train import move_optimizer_state
from . import resume_bundle as bundle

STOP=False
def stop_request(_signal,_frame):
    global STOP
    STOP=True


def main():
    p=argparse.ArgumentParser()
    for name in ('data','cache','upstream-checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--save-every',type=int,default=10)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after-steps',type=int,default=0)
    a=p.parse_args()
    if min(a.epochs,a.accumulation,a.save_every)<1 or a.stop_after_steps<0:raise ValueError('Invalid budget')
    records,splits,cache_identity=core.audit(a.data,a.cache,a.upstream_checkpoint)
    identity={'recipe':'native-windowseat-c1-m5-single-v1','upstream_sha256':cache_identity['checkpoint_sha256'],
              'dataset_sha256':cache_identity['dataset_manifest_sha256'],'rgb_cache_sha256':bundle.sha(a.cache/'manifest.json'),
              'architecture':VERSION,'accumulation':a.accumulation,'lr':a.lr,'seed':2026}
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGINT,stop_request);signal.signal(signal.SIGTERM,stop_request)
    torch.cuda.set_device(0);torch.set_num_threads(1);torch.manual_seed(2026);random.seed(2026)
    device=torch.device('cuda:0');model=PixelRefiner().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    updates=math.ceil(len(splits['train'])/a.accumulation)
    def factor(s):
        if s<updates:return (s+1)/updates
        return .1+.9*.5*(1+math.cos(math.pi*min(1,(s-updates)/max(1,(a.epochs-1)*updates))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    cursor={'epoch':0,'batch':0,'step':0};best=float('inf');best_epoch=0;pending=False
    if a.resume:
        ck,state=bundle.load(a.output,identity)
        model,metadata=core.load_model(ck/'model.safetensors',cache_identity,device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        cursor=state['cursor'];best=state['best_score'];best_epoch=state['best_epoch'];pending=state['validation_pending']
        bm=bundle.verify(bundle.recover(a.output,'best'))['cursor']
        if bm['score']<best:best=bm['score'];best_epoch=bm['epoch']
        bundle.restore_rng(state['rng']);del state
        print(json.dumps({'kind':'resume','cursor':cursor,'optimizer_state_count':len(optimizer.state),'rng_restored':True}),flush=True)
    if cursor['epoch']>a.epochs:raise ValueError('Cumulative epoch target is below restored progress')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(identity=identity,updates_per_epoch=updates,planned_steps=updates*a.epochs,
                  train_samples=len(splits['train']),validation_samples=len(splits['validation']),test_samples=len(splits['test']),
                  early_stopping=False,precision='FP32',backbone_frozen='cached C1-best only',
                  retention='best/model; latest/model+trainer_state',trainable_parameters=sum(v.numel() for v in model.parameters()))
    bundle.atomic_json(a.output/'config.json',config)

    def save_model(path):core.save_model(path,model,cache_identity,cursor['epoch'],cursor['step'])
    def commit(pending_validation=False):
        state={'cursor':dict(cursor),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
               'rng':bundle.rng_state(),'best_score':best,'best_epoch':best_epoch,'validation_pending':pending_validation}
        bundle.save(a.output,'latest',save_model,state,identity,dict(cursor))
    def validation(baseline=False):
        nonlocal best,best_epoch
        report=core.evaluate(model,a.data,a.cache,splits['validation'],device,a.output/'latest_validation',records)
        score=report['means']['l1'];improved=score<best-1e-6
        if improved:
            best=score;best_epoch=cursor['epoch']
            bundle.save(a.output,'best',save_model,None,identity,{**cursor,'score':score})
            bundle.atomic_json(a.output/'best_metrics.json',{'cursor':dict(cursor),**report})
            target=a.output/'best_validation'
            if target.exists():shutil.rmtree(target)
            shutil.copytree(a.output/'latest_validation',target)
        bundle.atomic_json(a.output/'latest_metrics.json',{'cursor':dict(cursor),**report})
        event={'kind':'baseline' if baseline else 'validation','cursor':dict(cursor),'means':report['means'],
               'minus_c1':report['minus_c1'],'best_updated':improved}
        with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        print(json.dumps(event),flush=True)
    if not a.resume:validation(True)
    elif pending:validation();commit(False)
    torch.cuda.reset_peak_memory_stats();limit=a.stop_after_steps or updates*a.epochs
    stop=False;started=time.monotonic()
    while cursor['epoch']<a.epochs and not stop:
        epoch=cursor['epoch'];ids=list(splits['train']);random.Random(2026+epoch+1).shuffle(ids)
        for gi in range(cursor['batch'],updates):
            if STOP or cursor['step']>=limit:stop=True;break
            offset=gi*a.accumulation;group=ids[offset:offset+a.accumulation]
            optimizer.zero_grad(set_to_none=True);model.train();values=[]
            for j,sid in enumerate(group):
                image,t0,gt=core.batch(a.data,a.cache,sid,device)
                if random.random()<.5:image=image.flip(-1);t0=t0.flip(-1);gt=gt.flip(-1)
                result=model(image,t0);loss,terms=losses(result,t0,gt)
                il=loss.new_zeros(())
                if (offset+j)%10==0:
                    identity_result=model(gt,gt);il=identity_result['delta'].abs().mean();loss=loss+.05*il
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite M5 loss')
                (loss/len(group)).backward()
                values.append({**{k:float(v.detach()) for k,v in terms.items()},'total':float(loss.detach()),'identity':float(il.detach())})
                del image,t0,gt,result,loss,terms,il
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            if not math.isfinite(grad) or grad<=0:raise RuntimeError('Invalid M5 gradient')
            optimizer.step();scheduler.step()
            cursor={'epoch':epoch,'batch':gi+1,'step':cursor['step']+1}
            boundary=cursor['batch']==updates
            if boundary:cursor.update(epoch=epoch+1,batch=0)
            record={'kind':'train','cursor':dict(cursor),'ids':group,'lr':optimizer.param_groups[0]['lr'],'gradient_norm':grad,
                    'terms':{k:sum(v[k] for v in values)/len(values) for k in values[0]},
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'elapsed_this_process_seconds':time.monotonic()-started}
            with (a.output/'train.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            if cursor['step']%5==0 or gi==0:print(json.dumps(record),flush=True)
            should_stop=STOP or cursor['step']>=limit
            if boundary or should_stop or cursor['step']==1 or cursor['step']%a.save_every==0:commit(boundary)
            if boundary:validation();commit(False)
            if should_stop:stop=True;break
    complete=cursor['epoch']==a.epochs and cursor['batch']==0
    if not complete:commit(False)
    summary={'status':'complete' if complete else 'paused','cursor':cursor,'completed_epochs':cursor['epoch'],'steps':cursor['step'],
             'best_epoch':best_epoch,'best_validation_l1':best,'early_stopped':False,'resumable_latest':True,'sealed_test_evaluated':False}
    bundle.atomic_json(a.output/'training_summary.json',summary);print(json.dumps(summary),flush=True)
    if not complete:raise SystemExit(130)


if __name__=='__main__':main()
