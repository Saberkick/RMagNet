"""Measure saved 8-bit real20 predictions and build comparison panels/report."""
from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from .stage1_train import ssim

ROOT=Path('/share/linmingheng-local/xuke')
DATA=ROOT/'datasets/liyucs_RAGNet/testsets/real20'
RUN=ROOT/'RMagNet/runs/real20_windowseat_m4'
VARIANTS={'Input':DATA/'blended','WindowSeat':RUN/'windowseat','M0':RUN/'m0','M4-best':RUN/'m4best'}

def read(path:Path)->torch.Tensor:
    with Image.open(path) as im: a=np.asarray(im.convert('RGB'),dtype=np.float32).copy()
    return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).div_(255)

def metric(pred,gt):
    pred=pred.clamp(0,1).mul(255).round().div(255); gt=gt.clamp(0,1).mul(255).round().div(255)
    mse=F.mse_loss(pred,gt)
    return {'l1':float(F.l1_loss(pred,gt)),'psnr':float(-10*torch.log10(mse.clamp_min(1e-12))),'ssim':float(ssim(pred,gt))}

def fit(path:Path,size=(480,360)):
    with Image.open(path) as im: x=im.convert('RGB').copy()
    x.thumbnail(size,Image.Resampling.LANCZOS); c=Image.new('RGB',size,(28,28,28)); c.paste(x,((size[0]-x.width)//2,(size[1]-x.height)//2)); return c

def err_img(path:Path,gt_path:Path,size=(480,360)):
    with Image.open(path) as a, Image.open(gt_path) as b:
        x=np.asarray(a.convert('RGB'),dtype=np.float32)/255; y=np.asarray(b.convert('RGB'),dtype=np.float32)/255
    e=np.abs(x-y).mean(2); heat=np.stack((np.clip(e*4,0,1),np.clip(np.sqrt(e)*.55*4**.5,0,1),np.zeros_like(e)),2)
    im=Image.fromarray((heat*255).round().astype(np.uint8)); im.thumbnail(size,Image.Resampling.LANCZOS); c=Image.new('RGB',size,(28,28,28)); c.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2)); return c

def main():
    ids=sorted((p.stem for p in (DATA/'blended').glob('*.jpg')),key=int)
    rows=[]; by=defaultdict(list); panel_dir=RUN/'panels'; panel_dir.mkdir(exist_ok=True)
    for n,sid in enumerate(ids,1):
        gt_path=DATA/'transmission_layer'/f'{sid}.jpg'; gt=read(gt_path)
        paths={'Input':DATA/'blended'/f'{sid}.jpg','WindowSeat':RUN/'windowseat'/f'{sid}_windowseat_output.png','M0':RUN/'m0'/f'{sid}_windowseat_output.png','M4-best':RUN/'m4best'/f'{sid}_windowseat_output.png'}
        sample={}
        for name,path in paths.items():
            if not path.is_file(): raise FileNotFoundError(path)
            pred=read(path)
            if pred.shape!=gt.shape: raise RuntimeError(f'shape mismatch {sid} {name}: {pred.shape} vs {gt.shape}')
            m=metric(pred,gt); row={'id':sid,'variant':name,'width':pred.shape[-1],'height':pred.shape[-2],**m}; rows.append(row); by[name].append(row); sample[name]=m
        cw,ch,hh=480,360,58; canvas=Image.new('RGB',(cw*5,(ch+hh)*2),'white'); draw=ImageDraw.Draw(canvas); font=ImageFont.load_default()
        columns=[('Input',paths['Input']),('GT',gt_path),('WindowSeat',paths['WindowSeat']),('M0',paths['M0']),('M4-best',paths['M4-best'])]
        for i,(name,path) in enumerate(columns):
            x=i*cw; subtitle='' if name=='GT' else f"PSNR {sample[name]['psnr']:.2f}  SSIM {sample[name]['ssim']:.4f}"
            draw.text((x+8,8),f'{sid}  {name}',fill='black',font=font); draw.text((x+8,29),subtitle,fill=(60,60,60),font=font); canvas.paste(fit(path),(x,hh))
            y=hh+ch; draw.text((x+8,y+8),'Reference' if name=='GT' else 'Absolute error x4',fill='black',font=font)
            if name!='GT': canvas.paste(err_img(path,gt_path),(x,y+hh))
        canvas.save(panel_dir/f'{sid}.png',compress_level=6)
        print(json.dumps({'sample':n,'total':len(ids),'id':sid}),flush=True)
    with (RUN/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    means={name:{k:sum(r[k] for r in rs)/len(rs) for k in ('l1','psnr','ssim')} for name,rs in by.items()}
    comparisons={}
    for name in ('M0','M4-best'):
        delta={k:means[name][k]-means['WindowSeat'][k] for k in ('l1','psnr','ssim')}
        wins={k:sum(1 for sid in ids if next(r[k] for r in by[name] if r['id']==sid) > next(r[k] for r in by['WindowSeat'] if r['id']==sid)) for k in ('psnr','ssim')}
        wins['l1']=sum(1 for sid in ids if next(r['l1'] for r in by[name] if r['id']==sid) < next(r['l1'] for r in by['WindowSeat'] if r['id']==sid))
        comparisons[name]={'minus_windowseat':delta,'wins_out_of_20':wins}
    m4_minus_m0={k:means['M4-best'][k]-means['M0'][k] for k in ('l1','psnr','ssim')}
    summary={'status':'complete','dataset':'RAGNet real20','dataset_commit':'75467a50dcbf1558dbb6b7b70cdcbefc78a4d242','sample_count':20,'metric_domain':'native-resolution saved 8-bit RGB; macro average over images','means':means,'comparisons_to_windowseat':comparisons,'m4_minus_m0':m4_minus_m0}
    (RUN/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# RAGNet real20：原生 WindowSeat、M0 与 M4-best','',f"- 数据集提交：`{summary['dataset_commit']}`。",'- 数据：20 对真实图像，按原始分辨率评测；输入和 GT 为仓库 JPEG，预测为保存后的 8 位 RGB PNG。','- 推理：三个模型均使用 WindowSeat 官方短边分块、Lanczos 拼接和随机种子 2026。','- 汇总：逐图宏平均。','', '## 汇总','', '| 版本 | L1 ↓ | PSNR ↑ | SSIM ↑ |','|---|---:|---:|---:|']
    for name in ('Input','WindowSeat','M0','M4-best'):
        m=means[name]; lines.append(f"| {name} | {m['l1']:.6f} | {m['psnr']:.4f} | {m['ssim']:.6f} |")
    for name in ('M0','M4-best'):
        delta=comparisons[name]['minus_windowseat']; wins=comparisons[name]['wins_out_of_20']
        lines += ['',f'## {name} 相对原生 WindowSeat','',f"- ΔL1：`{delta['l1']:+.6f}`",f"- ΔPSNR：`{delta['psnr']:+.4f} dB`",f"- ΔSSIM：`{delta['ssim']:+.6f}`",f"- 逐图胜出：PSNR {wins['psnr']}/20，SSIM {wins['ssim']}/20，L1 {wins['l1']}/20。"]
    lines += ['', '## M4-best 相对 M0','',f"- ΔL1：`{m4_minus_m0['l1']:+.6f}`",f"- ΔPSNR：`{m4_minus_m0['psnr']:+.4f} dB`",f"- ΔSSIM：`{m4_minus_m0['ssim']:+.6f}`",'', '逐图数值见 `metrics.csv`，20 张五列对比图见 `panels/`。','']
    (RUN/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__': main()
