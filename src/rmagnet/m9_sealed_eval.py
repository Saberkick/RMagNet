"""Fixed M9 best/latest and M4 sealed evaluation; no training or GT conditioning."""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open

from .m9_latent_only import LatentEditor, identity
from .m1b_train import load_initial
from .qwen_layer_probe import deterministic_encode
from .sma_data import load_manifest
from .m5_train import read, save_png, metrics, KEYS
from .m7_latent_tail import EXPECTED_M4, sha, atomic_json, emit


def equal_tensors(left, right):
    with safe_open(left, framework='pt',device='cpu') as a, safe_open(right, framework='pt',device='cpu') as b:
        if set(a.keys()) != set(b.keys()): return False
        for key in a.keys():
            if not torch.equal(a.get_tensor(key), b.get_tensor(key)): return False
    return True


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for name in ('data','initial','run','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(2026);torch.cuda.set_device(0)
    status=json.loads((a.run/'training_summary.json').read_text())
    if status['status']!='complete' or status['cursor']!={'epoch':4,'batch':0,'step':136}:
        raise RuntimeError('Expected completed four-epoch M9')
    if (a.run/'pipeline.exit_code').read_text().strip()!='0':raise RuntimeError('Training did not exit successfully')
    if sha(a.initial)!=EXPECTED_M4:raise RuntimeError('Wrong M4 final best')
    _,records,splits=load_manifest(a.data);ids=splits['test'];ident=identity(a)
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError(a.output)
    checkpoints={'m4':a.initial,'best':a.run/'best/transmission_lora.safetensors',
                 'latest':a.run/'latest/transmission_lora.safetensors'}
    info={'m4':{'path':str(a.initial),'sha256':EXPECTED_M4,'cursor':None}}
    for label in ('best','latest'):
        path=checkpoints[label]
        with safe_open(path,framework='pt') as f:md=f.metadata()
        if json.loads(md['identity'])!=ident:raise RuntimeError('Checkpoint identity differs '+label)
        state=json.loads((path.parent/'state.json').read_text())
        cursor=json.loads(md['cursor'])
        if cursor!=state['cursor'] or state['identity']!=ident:raise RuntimeError('Checkpoint state mismatch')
        if label=='latest' and cursor!=status['cursor']:raise RuntimeError('Latest is not final')
        info[label]={'path':str(path),'sha256':sha(path),'cursor':cursor}
    best_is_m4=equal_tensors(checkpoints['m4'],checkpoints['best'])
    a.output.mkdir(parents=True)
    config={'identity':ident,'training_summary':status,'checkpoint_info':info,
            'best_tensors_equal_m4':best_is_m4,'test_ids':ids,'count':len(ids),
            'physical_gpus':os.environ.get('CUDA_VISIBLE_DEVICES'),'model_parallel_cards':torch.cuda.device_count(),
            'metric_domain':'saved RGB8 PNG; macro mean; project uniform11x11 SSIM zero-pad5',
            'input':'ordinary I only; GT only read after predictions for metrics',
            'resize_crop':False,'training':False,'test_weight_selection':False}
    atomic_json(a.output/'config.json',config)
    runner=LatentEditor(a.initial,torch.cuda.device_count())
    runner.backend.set_trainable_branch(None);runner.backend.eval()
    if any(p.requires_grad for p in runner.backend.parameters()):raise RuntimeError('Inference must be frozen')
    started=time.monotonic();latents={}
    for sid in ids:
        x=read(a.data/'blended'/f'{sid}.png').to(runner.primary)
        latents[sid]=deterministic_encode(runner.backend,x*2-1).cpu()
    predictions={}
    for key in ('m4','best','latest'):
        if key=='best' and best_is_m4:
            predictions[key]=predictions['m4']
            emit({'phase':'best_reused_m4','reason':'all checkpoint tensor keys and values verified equal'})
            continue
        if key!='m4':
            runner.backend.set_trainable_branch('transmission')
            load_initial(runner.backend,checkpoints[key],runner.primary)
            runner.backend.set_trainable_branch(None);runner.backend.eval()
        predictions[key]={}
        for i,sid in enumerate(ids,1):
            edited=runner.forward(latents[sid].to(runner.primary))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=(runner.backend.upstream.decode(edited.to(runner.backend.vae.dtype),runner.backend.vae).float()+1)*.5
            if not torch.isfinite(pred).all():raise RuntimeError('Nonfinite prediction '+sid)
            path=a.output/key/'predictions'/f'{sid}.png';save_png(pred,path);predictions[key][sid]=path
            emit({'phase':'prediction','model':key,'done':i,'total':len(ids),'id':sid})
            del pred,edited
    reports={}
    for key in ('m4','best','latest'):
        rows=[]
        for sid in ids:
            image=read(a.data/'blended'/f'{sid}.png')
            gt=read(a.data/'transmission_layer'/f'{sid}.png')
            saved=read(predictions[key][sid])
            if saved.shape!=gt.shape:raise RuntimeError('Prediction geometry differs')
            rows.append({'id':sid,**metrics(saved,gt,image)})
        means={k:sum(r[k] for r in rows)/len(rows) for k in KEYS}
        label='M4 final best' if key=='m4' else 'M9 '+key+', epoch '+str(info[key]['cursor']['epoch'])
        report={'model':label,'count':len(rows),'means':means,'per_image':rows,
                'prediction_directory':str(a.output/('m4' if key=='best' and best_is_m4 else key)/'predictions')}
        if key!='m4':
            report['minus_m4']={k:means[k]-reports['m4']['means'][k] for k in KEYS}
            report['wins']={k:sum(r[k]>base[k] for r,base in zip(rows,reports['m4']['per_image'])) for k in ('psnr','ssim')}
        atomic_json(a.output/key/'metrics.json',report)
        with open(a.output/key/'metrics.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        reports[key]=report
    summary={'complete':True,'split':'sealed test','count':len(ids),'reports':reports,
             'best_tensors_equal_m4':best_is_m4,'checkpoint_info':info,'elapsed_seconds':time.monotonic()-started}
    atomic_json(a.output/'summary.json',summary)
    with open(a.output/'comparison.csv','w',newline='') as f:
        fields=['id']+[key+'_'+metric for key in reports for metric in ('psnr','ssim','l1')]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,sid in enumerate(ids):
            w.writerow({'id':sid,**{key+'_'+metric:r['per_image'][i][metric] for key,r in reports.items() for metric in ('psnr','ssim','l1')}})
    vals=[json.loads(line) for line in (a.run/'metrics.jsonl').read_text().splitlines()]
    lines=['# M9 latent-only 四 epoch：验证与封存测试结果','',
           '从 M4 final best 小幅微调已有编辑 LoRA，训练只用归一化最终 latent L1，学习率1e-5。主干和VAE冻结。',
           '', '训练完成4 epoch / 136次更新，正常退出。没有使用封存测试选择权重。',
           '', '## 25张验证集', '', '| Epoch | latent L1 | PSNR / dB | SSIM |', '|---:|---:|---:|---:|']
    for v in vals:
        m=v['means'];lines.append(f"| {v['cursor']['epoch']} | {m['latent_l1']:.6f} | {m['psnr']:.4f} | {m['ssim']:.6f} |")
    lines += ['', '## 24张封存测试集', '', '| 模型 | PSNR / dB | SSIM | L1 | Edge L1 |', '|---|---:|---:|---:|---:|']
    for r in reports.values():
        m=r['means'];lines.append(f"| {r['model']} | {m['psnr']:.4f} | {m['ssim']:.6f} | {m['l1']:.6f} | {m['edge_l1']:.6f} |")
    lines += ['', '所有模型使用同一24张清单、同一推理实现、原有动态尺寸；保存RGB8 PNG后沿用项目指标计算逐图算术均值。SSIM为RGB、11×11均匀窗口、边界零填充5。',
              '', f'best 是否逐张量等于 M4：{best_is_m4}。若相等，best复用本轮M4预测；其metrics.csv和metrics.json独立保存并明确标记预测位置。', '']
    for key in ('best','latest'):
        r=reports[key];d=r['minus_m4'];lines.append(f"- {r['model']} 相对 M4：PSNR {d['psnr']:+.4f} dB；SSIM {d['ssim']:+.6f}；逐图PSNR胜出 {r['wins']['psnr']}/24，SSIM胜出 {r['wins']['ssim']}/24。")
    lines += ['', '## 解读边界', '',
              'latent L1下降与最终RGB图像质量提升是不同结果，需同时查看PSNR/SSIM、边缘误差及局部图像。单次4 epoch实验不能证明所有latent监督都无效，也不能证明VAE是唯一瓶颈。',
              '', f'结果目录：`{a.output}`。各模型逐图表见子目录metrics.csv，固定权重身份见config.json。']
    (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    emit({'phase':'complete','results':{k:{x:r[x] for x in ('means','minus_m4','wins') if x in r} for k,r in reports.items()}})


if __name__=='__main__':main()
