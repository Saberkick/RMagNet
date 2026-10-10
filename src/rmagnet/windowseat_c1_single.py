"""Native WindowSeat C1, single GPU, effective batch four, resumable latest."""
import argparse,json,math,signal,shutil,time
from contextlib import nullcontext
from pathlib import Path
import bitsandbytes as bnb
import lpips
import safetensors.torch
import torch
from torch.utils.data import DataLoader,default_collate
from . import sma_c_train as core
from . import resume_bundle as bundle
from .sma_conditioned import ConditionedSMA,install,load_joint,save_joint,calibrate,diagnostics
from .sma_joint import install_teacher,lora_parameters,teacher_context
from .sma_data import load_manifest
from .qwen_backend import QwenSharedBackend
from .stage1_train import seed_everything,make_scheduler,append_jsonl
from .m2a_data_baseline import grouped_global_batches,validate
from .c1_l20_train import move_optimizer_state
from .qwen_layer_probe import deterministic_encode

STOP=False
def request_stop(_signum,_frame):
    global STOP
    STOP=True
    print('Graceful stop requested; finish this optimizer update and save latest',flush=True)


def arguments():
    p=argparse.ArgumentParser()
    for name in ('data-root','cache-root','memory-file','initial','run-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--initial-sha256',default='2f04225c5be1cac69a5d76faf4c95515d5503210cfee73c0bb1e79309f5a74e9')
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--resume',action='store_true');p.add_argument('--save-every',type=int,default=10)
    p.add_argument('--stop-after-steps',type=int,default=0,help='Verification budget; global cumulative optimizer steps')
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    if min(a.epochs,a.accumulation,a.save_every)<1 or a.stop_after_steps<0:raise ValueError('Invalid budget')
    # Pin the same objective, rates and gradual output-gradient controls as C1.
    a.ssim_weight=.2;a.edge_weight=.1;a.consistency_coefficient=.10
    a.cpu_offload_activations=True;a.learning_rate=1e-4;a.gamma_learning_rate=5e-3
    a.lora_learning_rate=5e-6;a.warmup_steps=20;a.gradient_ramp_steps=36
    a.calibration_start_epoch=1;a.calibration_interval=8;a.seed=2026
    return a


def harvest(parameters,accumulator,weight):
    for i,p in enumerate(parameters):
        if p.grad is not None:
            g=p.grad.detach().to('cpu')
            if i not in accumulator:accumulator[i]=g.clone().mul_(weight)
            else:accumulator[i].add_(g,alpha=weight)
            p.grad=None


def restore_gradients(parameters,accumulator,device):
    for i,p in enumerate(parameters):p.grad=accumulator[i].to(device) if i in accumulator else None


def gradient_norm(parameters):
    values=[p.grad.float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def main():
    a=arguments();device=torch.device('cuda:0')
    manifest,records,splits=load_manifest(a.data_root)
    cm,cr=core.load_cache(a.cache_root,a.data_root,splits['train'])
    initial_sha=bundle.sha(a.initial)
    if initial_sha!=a.initial_sha256 or cm['teacher']['adapter_sha256']!=initial_sha:raise RuntimeError('Teacher/initial identity mismatch')
    mr=json.loads((a.memory_file.parent/'report.json').read_text())
    if mr['teacher_sha256']!=initial_sha or mr['cache_manifest_sha256']!=bundle.sha(a.cache_root/'manifest.json') or mr['memory_sha256']!=bundle.sha(a.memory_file):raise RuntimeError('Memory identity mismatch')
    data=core.M4Dataset(a.data_root,a.cache_root,records,cr,splits['train'],False)
    if len(data)%a.accumulation:raise RuntimeError('Use a whole-group training size divisible by effective batch')
    updates=len(data)//a.accumulation;planned=updates*a.epochs
    identity={'recipe':'native-windowseat-c1-single-v1','dataset_sha256':bundle.sha(a.data_root/'manifest.json'),
              'cache_sha256':bundle.sha(a.cache_root/'manifest.json'),'memory_sha256':bundle.sha(a.memory_file),
              'initial_sha256':initial_sha,'effective_batch':a.accumulation,'seed':a.seed}
    config={'args':vars(a),'identity':identity,'world_size':1,'physical_gpu':__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
            'effective_batch':a.accumulation,'updates_per_epoch':updates,'planned_steps':planned,
            'counts':{k:len(v) for k,v in splits.items()},'early_stopping':False,'retention':'best/model; latest/model+trainer_state',
            'activation_offload':'save_on_cpu pinned; gradient accumulator and optimizer state on CPU between expensive forwards',
            'gradient_control':{'spatial':.08,'texture':.08,'semantic':.08,'combined_cap':.25,'ramp_updates':36,'ema':.9},
            'resume':{'cursor':'next group in deterministic aspect-bucket epoch','save_every_updates':a.save_every,'saves_first_update':True,
                      'state':'AdamW, paged8bit AdamW, both schedules, RNG, gradient controller, gate tau, best selection'}}
    if a.preflight_only:print(json.dumps(config,indent=2,default=str));return
    torch.cuda.set_device(0);seed_everything(a.seed)
    prop=torch.cuda.get_device_properties(0)
    free,total=torch.cuda.mem_get_info(0)
    config['gpu_identity']={'visible':__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
                            'uuid':str(prop.uuid),'free_gib_before_load':free/2**30,'total_gib':total/2**30}
    print(json.dumps({'kind':'gpu_identity',**config['gpu_identity']}),flush=True)
    if free<22_000*1024**2:raise RuntimeError('Selected CUDA device has insufficient free memory before model load')
    if a.run_dir.exists() and any(a.run_dir.iterdir()) and not a.resume:raise FileExistsError(a.run_dir)
    a.run_dir.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    backend=QwenSharedBackend.from_local(device)
    backend.initialization_family='WindowSeat official';backend.initialization_sha256=initial_sha
    backend.set_trainable_branch('transmission');core.load_initial(backend,a.initial,device);backend.set_trainable_branch(None)
    backend.joint_training=True;config['teacher_audit']=install_teacher(backend)
    sma=ConditionedSMA().to(device)
    sma.load_memory(safetensors.torch.load_file(a.memory_file,device='cpu'))
    saved=None;cursor={'epoch':0,'batch':0,'step':0};best_score=float('inf');best_step=0
    if a.resume:
        if not (a.run_dir/'latest').exists() and not (a.run_dir/'.previous_latest').exists():raise FileNotFoundError('No committed resumable latest bundle')
        ck,saved=bundle.load(a.run_dir,identity)
        load_joint(ck/'model.safetensors',backend,sma,device,initial_sha)
        cursor=dict(saved['cursor']);best_score=saved['best_score'];best_step=saved['best_step']
        best_meta=bundle.verify(bundle.recover(a.run_dir,'best'))
        if best_meta['cursor']['score']<best_score:best_score=best_meta['cursor']['score'];best_step=best_meta['cursor']['step']
        if cursor['epoch']>a.epochs or cursor['step']>planned:raise ValueError('Cumulative epoch target is below restored progress')
    sma.freeze_memory();runtime=install(backend,sma)
    backend.transformer.train();backend.vae.eval()
    readers=list(sma.readers.parameters());student=lora_parameters(backend);parameters=readers+student
    student_ids={id(p) for p in student}
    if any(p.requires_grad and id(p) not in student_ids for p in backend.transformer.parameters()) or any(p.requires_grad for p in backend.vae.parameters()):raise RuntimeError('Frozen backbone has trainable parameters')
    optimizer=torch.optim.AdamW([{'params':[p for n,p in sma.readers.named_parameters() if not n.endswith('gamma')],'lr':a.learning_rate},
                                {'params':[p for n,p in sma.readers.named_parameters() if n.endswith('gamma')],'lr':a.gamma_learning_rate,'weight_decay':0.}],weight_decay=.01)
    lo=bnb.optim.PagedAdamW8bit(student,lr=a.lora_learning_rate,weight_decay=.01)
    scheduler=make_scheduler(optimizer,min(a.warmup_steps,planned),planned)
    ls=make_scheduler(lo,min(a.warmup_steps,planned),planned)
    controller=core.GradientController({'spatial':.08,'texture':.08,'semantic':.08},36,total_cap=.25)
    pending=False
    if saved:
        optimizer.load_state_dict(saved['optimizer']);lo.load_state_dict(saved['lora_optimizer'],move_to_device=False)
        move_optimizer_state(lo,torch.device('cpu'))
        scheduler.load_state_dict(saved['scheduler']);ls.load_state_dict(saved['lora_scheduler'])
        controller.ema=saved['controller_ema'];pending=saved['validation_pending']
        if saved['gate_tau'] is not None:runtime.tau=saved['gate_tau']
    config['student_lora_parameters']=sum(p.numel() for p in student);config['reader_parameters']=sum(p.numel() for p in readers)
    bundle.atomic_json(a.run_dir/'run_config.json',config)
    lp=lpips.LPIPS(net='squeeze',verbose=False).eval().cpu().requires_grad_(False)
    vl=DataLoader(core.M2ValidationDataset(a.data_root,records,splits['validation']),batch_size=1,num_workers=0)
    if saved:
        bundle.restore_rng(saved['rng']);del saved
        event={'kind':'resume','cursor':cursor,'target_epochs':a.epochs,'optimizer_state_count':len(lo.state),
               'rng_restored':True,'scheduler_step':ls.last_epoch,'controller_ema_restored':dict(controller.ema)}
        append_jsonl(a.run_dir/'metrics.jsonl',event);print(json.dumps(event),flush=True)
    torch.cuda.reset_peak_memory_stats()

    def commit(pending_validation=False):
        move_optimizer_state(lo,torch.device('cpu'))
        state={'cursor':dict(cursor),'optimizer':optimizer.state_dict(),'lora_optimizer':lo.state_dict(),
               'scheduler':scheduler.state_dict(),'lora_scheduler':ls.state_dict(),'rng':bundle.rng_state(),
               'controller_ema':dict(controller.ema),'gate_tau':getattr(runtime,'tau',None),
               'best_score':best_score,'best_step':best_step,'validation_pending':pending_validation,'target_epochs':a.epochs}
        # Reader state is small; student optimizer state already lives on CPU.
        bundle.save(a.run_dir,'latest',lambda p:save_joint(p,backend,initial_sha),state,identity,dict(cursor))

    def validation(baseline=False):
        nonlocal best_score,best_step
        path=a.run_dir/'validation'/f"step_{cursor['step']:06d}"
        if path.exists():shutil.rmtree(path)
        report=validate(backend,vl,device,lp,a.run_dir,cursor['step'],a.seed)
        score=report['means']['l1'];improved=score<best_score
        if improved:
            best_score=score;best_step=cursor['step']
            bundle.save(a.run_dir,'best',lambda p:save_joint(p,backend,initial_sha),None,identity,{**cursor,'score':score})
            bundle.atomic_json(a.run_dir/'best_metrics.json',{'step':best_step,'means':report['means']})
            best_images=a.run_dir/'best_validation'
            if best_images.exists():shutil.rmtree(best_images)
            shutil.copytree(path,best_images)
        latest_images=a.run_dir/'latest_validation'
        if latest_images.exists():shutil.rmtree(latest_images)
        shutil.copytree(path,latest_images);shutil.rmtree(path)
        bundle.atomic_json(a.run_dir/'latest_metrics.json',{'cursor':dict(cursor),'means':report['means']})
        event={'kind':'baseline' if baseline else 'validation','cursor':dict(cursor),'means':report['means'],'best_updated':improved}
        append_jsonl(a.run_dir/'metrics.jsonl',event);print(json.dumps(event),flush=True)

    if not a.resume:validation(True)
    elif pending:validation();commit(False)
    started=time.monotonic();limit=a.stop_after_steps or planned
    stop=False
    while cursor['epoch']<a.epochs and not stop:
        epoch=cursor['epoch'];groups=grouped_global_batches(data,a.seed,epoch,a.accumulation)
        for gi in range(cursor['batch'],len(groups)):
            if cursor['step']>=limit or STOP:stop=True;break
            group=groups[gi];accumulator={};logs=[];conditions=[]
            old_ema=dict(controller.ema);new_emas=[]
            do_calibration=epoch>=1 and (cursor['step']+1)%8==0
            optimizer.zero_grad(set_to_none=True);lo.zero_grad(set_to_none=True)
            for index in group:
                controller.ema=dict(old_ema)
                batch=default_collate([data[index]])
                record,condition=micro_step(backend,batch,device,a,controller,cursor['step']+1)
                new_emas.append(dict(controller.ema));logs.append(record)
                if do_calibration:conditions.append(condition)
                harvest(parameters,accumulator,1/len(group))
                del condition,batch
                torch.cuda.empty_cache()
            controller.ema={k:sum(e[k] for e in new_emas)/len(new_emas) for k in new_emas[0]}
            restore_gradients(parameters,accumulator,device);del accumulator
            calibration={'calibration':False}
            if do_calibration:
                gates=sma.gate_parameters();original=[p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in gates]
                extra=[torch.zeros_like(p) for p in gates];cal=[];old_tau=getattr(runtime,'tau',None);taus=[]
                for latent,target in conditions:
                    for p,g in zip(gates,original):p.grad=g.clone()
                    if old_tau is None:
                        if hasattr(runtime,'tau'):del runtime.tau
                    else:runtime.tau=old_tau
                    cal.append(calibrate(backend,latent,target,cursor['step']+1,epoch));taus.append(runtime.tau)
                    for e,p,g in zip(extra,gates,original):e.add_(p.grad-g,alpha=1/len(conditions))
                for p,g,e in zip(gates,original,extra):p.grad=g+e
                runtime.tau=sum(taus)/len(taus)
                calibration={'calibration':True,'per_micro':cal,'gate_tau':runtime.tau}
                del original,extra,conditions
            ln=gradient_norm(student);rn=gradient_norm(readers)
            if not math.isfinite(ln) or ln<=0:raise RuntimeError('Student LoRA has no finite gradient')
            for n,p in backend.transformer.named_parameters():
                if id(p) not in student_ids and p.grad is not None:raise RuntimeError('Frozen backbone received gradient')
            if any(p.grad is not None for p in backend.vae.parameters()) or any(p.grad is not None for p in sma.memory.parameters()):raise RuntimeError('Frozen VAE/memory received gradient')
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.))
            if not math.isfinite(norm) or norm<=0:raise RuntimeError('Invalid parameter gradient norm')
            gamma_before=sma.readers['39_in'].gamma.detach().clone() if cursor['step']==0 else None
            probe=max(student,key=lambda p:float(p.grad.float().square().sum()) if p.grad is not None else -1) if cursor['step']==0 else None
            before=probe.detach().clone() if probe is not None else None
            optimizer.step();move_optimizer_state(lo,device);lo.step();lo.zero_grad(set_to_none=True);move_optimizer_state(lo,torch.device('cpu'))
            optimizer.zero_grad(set_to_none=True);scheduler.step();ls.step()
            if before is not None and torch.equal(before,probe):raise RuntimeError('First student LoRA update failed')
            if gamma_before is not None and torch.equal(gamma_before,sma.readers['39_in'].gamma):raise RuntimeError('First condition gamma update failed')
            del before,probe,gamma_before
            cursor={'epoch':epoch,'batch':gi+1,'step':cursor['step']+1}
            boundary=cursor['batch']==len(groups)
            if boundary:cursor.update(epoch=epoch+1,batch=0)
            event={'kind':'train','cursor':dict(cursor),'micro_samples':len(group),'ids':[data.sample_ids[i] for i in group],
                   'mean_terms':{k:sum(x[k] for x in logs)/len(logs) for k in logs[0]},'micro_terms':logs,
                   'student_lora_grad_norm':ln,'reader_grad_norm':rn,'parameter_grad_norm':norm,
                   'lora_lr':ls.get_last_lr()[0],'condition_lr':scheduler.get_last_lr()[0],
                   'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                   'backbone_vae_memory_frozen':True,'first_update_verified':True,'calibration':calibration,
                   'elapsed_this_process_seconds':time.monotonic()-started,**diagnostics(sma,runtime)}
            append_jsonl(a.run_dir/'metrics.jsonl',event);print(json.dumps(event),flush=True)
            should_stop=STOP or cursor['step']>=limit
            if boundary or should_stop or cursor['step']==1 or cursor['step']%a.save_every==0:commit(boundary)
            if boundary:validation();commit(False)
            if should_stop:stop=True;break
    complete=cursor['epoch']==a.epochs and cursor['batch']==0
    if not complete:commit(False)
    summary={'status':'complete' if complete else 'paused','cursor':cursor,'epochs_requested':a.epochs,'epochs_completed':cursor['epoch'],
             'optimizer_updates':cursor['step'],'best_step':best_step,'best_validation_l1':best_score,'early_stopped':False,
             'resumable_latest':True,'sealed_test_evaluated':False}
    bundle.atomic_json(a.run_dir/'training_summary.json',summary);print(json.dumps(summary),flush=True)
    if not complete:raise SystemExit(130)


def micro_step(backend,batch,device,args,controller,step):
    runtime=backend.sma_runtime
    from .sma_c_train import (token_weights,transmission_loss,consistency_loss,spatial_loss,
        online_prediction_features,texture_loss,semantic_loss,finite,tensor_norm,forward_from_latent)
    from torch.nn import functional as F
    image = batch["image"].to(device, non_blocking=True)
    p90 = batch["p90"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    grid_h, grid_w = (int(value) for value in batch["token_grid"][0])
    expected_tokens = grid_h * grid_w
    gate, keep, restore = token_weights(batch, device)
    confidence = 0.5 + 0.5 * batch["late_agreement"].to(
        device, non_blocking=True
    ).float()

    with torch.no_grad():
        latent_i = deterministic_encode(backend, image)
        latent_90 = deterministic_encode(backend, p90)
        prediction_i_probe = forward_from_latent(
            backend, latent_i
        ).detach()
        prediction_90_reference = forward_from_latent(
            backend, latent_90
        ).detach()

    leaf = prediction_i_probe.detach().requires_grad_(True)
    rec_i_probe, rec_i_parts = transmission_loss(
        leaf, target, args.ssim_weight, args.edge_weight
    )
    gate_pixel_i = F.interpolate(
        gate.reshape(1, 1, grid_h, grid_w),
        size=leaf.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    cons_i = consistency_loss(
        leaf, prediction_90_reference, gate_pixel_i
    )
    base_i = 0.5 * (rec_i_probe + args.consistency_coefficient * cons_i)
    spatial, spatial_restore, spatial_keep = spatial_loss(
        leaf, target, image, gate, grid_h, grid_w
    )
    # Same fixed M4 teacher as the cache; only new modules are bypassed.
    offload = lambda: torch.autograd.graph.save_on_cpu(pin_memory=True) if args.cpu_offload_activations else nullcontext()
    with offload(), teacher_context(backend, runtime):
      try:
          features = online_prediction_features(
              backend, leaf, expected_tokens
          )
          texture = texture_loss(
              features, batch, device, keep, restore
          )
          semantic, semantic_content, semantic_relation = semantic_loss(
              features,
              batch,
              device,
              confidence,
              grid_h,
              grid_w,
          )
          for name, value in {
              "base_i": base_i,
              "spatial": spatial,
              "texture": texture,
              "semantic": semantic,
          }.items():
              finite(name, value)
          base_gradient = torch.autograd.grad(
              base_i, leaf, retain_graph=True
          )[0]
          spatial_gradient = torch.autograd.grad(
              spatial, leaf, retain_graph=True
          )[0]
          texture_gradient = torch.autograd.grad(
              texture, leaf, retain_graph=True
          )[0]
          semantic_gradient = torch.autograd.grad(
              semantic, leaf
          )[0]
      finally:
          pass
    output_gradient_i, gradient_log = controller.combine(
        base_gradient,
        {
            "spatial": spatial_gradient,
            "texture": texture_gradient,
            "semantic": semantic_gradient,
        },
        step,
    )
    if not torch.isfinite(output_gradient_i).all() or float(tensor_norm(output_gradient_i)) <= 0:
        raise RuntimeError("Invalid M4 I output gradient")
    rec_i_log = rec_i_probe.detach()
    cons_i_log = cons_i.detach()
    spatial_log = spatial.detach()
    spatial_restore_log = spatial_restore.detach()
    spatial_keep_log = spatial_keep.detach()
    texture_log = texture.detach()
    semantic_log = semantic.detach()
    semantic_content_log = semantic_content.detach()
    semantic_relation_log = semantic_relation.detach()
    del features, leaf, base_gradient, spatial_gradient, texture_gradient, semantic_gradient
    del base_i, rec_i_probe, rec_i_parts, cons_i, gate_pixel_i
    del spatial, spatial_restore, spatial_keep
    del texture, semantic, semantic_content, semantic_relation
    torch.cuda.empty_cache()

    with offload():
        prediction_i = forward_from_latent(backend, latent_i)
        prediction_i.backward(output_gradient_i)
    prediction_i_reference = prediction_i.detach()
    del prediction_i, output_gradient_i
    torch.cuda.empty_cache()

    with offload():
        prediction_90 = forward_from_latent(backend, latent_90)
    rec_90, rec_90_parts = transmission_loss(
        prediction_90, target, args.ssim_weight, args.edge_weight
    )
    cons_90 = consistency_loss(
        prediction_90,
        prediction_i_reference,
        F.interpolate(
            gate.reshape(1, 1, grid_h, grid_w),
            size=prediction_90.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ),
    )
    base_90 = 0.5 * (rec_90 + args.consistency_coefficient * cons_90)
    finite("base_90", base_90)
    output_gradient_90 = torch.autograd.grad(
        base_90, prediction_90
    )[0]
    prediction_90.backward(output_gradient_90)

    if not torch.isfinite(output_gradient_90).all():raise RuntimeError('Nonfinite P90 output gradient')
    stats={'rec_i':float(rec_i_log),'rec_p90':float(rec_90.detach()),
           'consistency_i':float(cons_i_log),'consistency_p90':float(cons_90.detach()),
           'spatial':float(spatial_log),'spatial_restore':float(spatial_restore_log),'spatial_keep':float(spatial_keep_log),
           'texture':float(texture_log),'semantic':float(semantic_log),'semantic_content':float(semantic_content_log),
           'semantic_relation':float(semantic_relation_log),**gradient_log}
    if not all(__import__('math').isfinite(v) for v in stats.values()):raise RuntimeError('Nonfinite micro-step log')
    return stats,(latent_i.detach(),target.detach())


if __name__=='__main__':main()
