"""Input-only M7 best sealed evaluation, paired with fresh frozen M4 predictions."""
import argparse,csv,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
import safetensors.torch as st
from safetensors import safe_open
from .m7_latent_tail import (identity,sha,atomic_json,emit,TailRestorer,restore_trainable,decode)
from .qwen_backend import QwenSharedBackend
from .m1b_train import load_initial
from .qwen_layer_probe import deterministic_encode
from .sma_data import load_manifest
from .m5_train import read,save_png,metrics,KEYS

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for k in ('data','initial','tail-cache','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(2026)
    if torch.cuda.device_count()!=2:raise RuntimeError('Select exactly two free GPUs: frozen M4, restoration')
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError(a.output)
    _,records,splits=load_manifest(a.data);ids=splits['test']
    ident=identity(SimpleNamespace(data=a.data,initial=a.initial))
    with safe_open(a.checkpoint,framework='pt') as f:md=f.metadata()
    if json.loads(md['identity'])!=ident:raise RuntimeError('Checkpoint provenance mismatch')
    cache=json.loads((a.tail_cache/'manifest.json').read_text())
    if not cache['complete'] or cache['identity']!=ident:raise RuntimeError('Tail identity mismatch')
    if sha(a.tail_cache/'frozen_tail.safetensors')!=cache['tail_sha256']:raise RuntimeError('Tail hash mismatch')
    a.output.mkdir(parents=True)
    atomic_json(a.output/'config.json',{'identity':ident,'checkpoint':str(a.checkpoint),
        'checkpoint_sha256':sha(a.checkpoint),'checkpoint_metadata':md,'test_ids':ids,'test_count':len(ids),
        'inference_input':'ordinary input I only; no GT/P90/DoLP conditioning',
        'protocol':'native processed geometry; deterministic posterior mode; NF4 M4, BF16 restoration; RGB8 macro metrics',
        'ssim':'uniform11x11, zero padding 5, existing project convention','fresh_matched_m4_baseline':True})
    primary=torch.device('cuda:0');secondary=torch.device('cuda:1');torch.cuda.set_device(primary)
    backend=QwenSharedBackend.from_local(primary)
    backend.set_trainable_branch('transmission');load_initial(backend,a.initial,primary)
    backend.set_trainable_branch(None);backend.eval()
    if any(x.requires_grad for x in backend.parameters()):raise RuntimeError('M4 must be frozen')
    model=TailRestorer(a.tail_cache/'frozen_tail.safetensors',int(md['rank'])).to(secondary)
    restore_trainable(model,st.load_file(a.checkpoint));model.requires_grad_(False).eval()
    captured={}
    def hook(_m,_args,kwargs,out):
        captured.update(hidden=out[1].detach().to(secondary),text=out[0].detach().to(secondary),
            temb=kwargs['temb'].detach().to(secondary),
            mask=kwargs['encoder_hidden_states_mask'].detach().to(secondary))
        for k,r in enumerate(kwargs['image_rotary_emb']):captured[f'rope_{k}']=r.detach().to(secondary)
    handle=backend.transformer.transformer_blocks[-1].register_forward_hook(hook,with_kwargs=True)
    rows={'M4 final best':[],'M7 best':[]};started=time.monotonic()
    try:
        for n,sid in enumerate(ids,1):
            captured.clear();image=read(a.data/'blended'/f'{sid}.png')
            z=deterministic_encode(backend,image*2-1)
            zm4=backend.upstream.flow_step(z,backend.transformer,backend.vae,backend.embeddings)
            c={**captured,'z_input':z.to(secondary),'z_m4':zm4.to(secondary)}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                final,delta=model(c,image.to(secondary))
                predictions={'M4 final best':decode(backend.vae,zm4),
                             'M7 best':decode(backend.vae,final.to(primary))}
            if not all(torch.isfinite(x).all() for x in [final,delta,*predictions.values()]):
                raise RuntimeError('Nonfinite sealed prediction '+sid)
            # GT is loaded only after all model forwards have finished.
            gt=read(a.data/'transmission_layer'/f'{sid}.png')
            for name,folder in [('M4 final best','m4'),('M7 best','m7')]:
                path=a.output/folder/'predictions'/f'{sid}.png';save_png(predictions[name],path)
                saved=read(path)
                if saved.shape!=gt.shape:raise RuntimeError('Saved image geometry mismatch')
                values=metrics(saved,gt,image)
                row={'id':sid,**values};rows[name].append(row)
            emit({'phase':'sealed_sample','done':n,'total':len(ids),'id':sid,
                  'm4_psnr':rows['M4 final best'][-1]['psnr'],'m7_psnr':rows['M7 best'][-1]['psnr'],
                  'delta_abs_mean':float(delta.abs().mean()),'elapsed_seconds':time.monotonic()-started})
            del image,z,zm4,c,final,delta,predictions,gt,saved;captured.clear()
    finally:handle.remove()
    reports={}
    for name,folder in [('M4 final best','m4'),('M7 best','m7')]:
        values=rows[name];means={k:sum(v[k] for v in values)/len(values) for k in KEYS}
        reports[name]={'count':len(values),'means':means,'per_image':values}
        atomic_json(a.output/folder/'metrics.json',reports[name])
        with open(a.output/folder/'metrics.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    delta={k:reports['M7 best']['means'][k]-reports['M4 final best']['means'][k] for k in KEYS}
    wins={k:sum(m[k]>b[k] for m,b in zip(rows['M7 best'],rows['M4 final best'])) for k in ('psnr','ssim')}
    summary={'complete':True,'split':'sealed test','count':len(ids),'reports':reports,
        'minus_m4':delta,'wins':wins,'elapsed_inference_seconds':time.monotonic()-started}
    atomic_json(a.output/'summary.json',summary)
    table='\n'.join(f"| {name} | {r['count']} | {r['means']['psnr']:.4f} | {r['means']['ssim']:.6f} | {r['means']['l1']:.6f} | {r['means']['edge_l1']:.6f} |" for name,r in reports.items())
    (a.output/'REPORT.md').write_text(f'''# M7 best 封存测试集评估

同一组 {len(ids)} 张封存测试图；M7 best 来自第3 epoch。M4 与 M7 同次生成、相同输入/GT/尺寸，模型只输入普通 I。
指标基于保存后的 RGB8 PNG，逐图宏平均；SSIM 沿用项目 uniform 11×11、zero padding 5。

| 模型 | 图数 | PSNR / dB | SSIM | L1 | 边缘 L1 |
|---|---:|---:|---:|---:|---:|
{table}

M7 相对 M4：PSNR {delta['psnr']:+.4f} dB，SSIM {delta['ssim']:+.6f}。
逐图 PSNR 胜出 {wins['psnr']}/{len(ids)}，SSIM 胜出 {wins['ssim']}/{len(ids)}。
低变化区 L1 差 {delta['low_change_keep_l1']:+.6f}，高变化区 L1 差 {delta['high_change_restore_l1']:+.6f}（误差下降为改善）。

预测位于 `m4/predictions/`、`m7/predictions/`，逐图指标位于对应 `metrics.csv`；模型与数据哈希见 `config.json`。
没有训练、修改训练缓存或依据封存测试选择 checkpoint。
''')
    emit({'phase':'complete','count':len(ids),'means':{k:v['means'] for k,v in reports.items()},'minus_m4':delta,'wins':wins})

if __name__=='__main__':main()
