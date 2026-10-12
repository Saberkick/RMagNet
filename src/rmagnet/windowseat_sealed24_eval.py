"""Official unmodified WindowSeat weights, current sealed split and M9 protocol."""
import csv,json,os,time
from pathlib import Path
import torch
from .qwen_backend import check_snapshots
from .m9_latent_only import LatentEditor
from .qwen_layer_probe import deterministic_encode
from .m5_train import read,save_png,metrics,KEYS
from .m7_latent_tail import sha,atomic_json,emit
from .sma_data import load_manifest

ROOT=Path('/share/linmingheng-local/xuke')
PROJECT=Path(__file__).resolve().parents[2]
DATA=ROOT/'datasets/rmagnet_sma_dataset3'
OUT=PROJECT/'results_archive/WindowSeat/windowseat_official_sealed24_mode'


@torch.no_grad()
def main():
    torch.set_num_threads(1);torch.manual_seed(2026);torch.cuda.set_device(0)
    if OUT.exists() and any(OUT.iterdir()):raise FileExistsError(OUT)
    _,records,splits=load_manifest(DATA);ids=splits['test']
    if len(ids)!=24:raise RuntimeError('Expected fixed current sealed24')
    qwen,official=check_snapshots()
    lora=official/'transformer_lora/pytorch_lora_weights.safetensors'
    config={'model':'Official WindowSeat; no personal training',
            'qwen_snapshot':str(qwen),'windowseat_snapshot':str(official),
            'official_lora_sha256':sha(lora),'dataset_manifest_sha256':sha(DATA/'manifest.json'),
            'test_ids':ids,'count':24,'input':'I only; GT/P90/DoLP not supplied to model',
            'physical_gpus':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'dtype':'NF4 backbone, BF16 activations/VAE, FP32 official LoRA',
            'protocol':'same native full forward as M4/M9; BF16(499)/1000; final zI-velocity',
            'encoder':'deterministic posterior mode with fixed latent mean/std',
            'geometry':'existing processed dynamic sizes, no resize/crop/padding/tiling',
            'metrics':'saved RGB8 PNG; RGB PSNR range1; project uniform11x11 SSIM zero-pad5; macro means',
            'training':False,'claim':'official weights under unified protocol, not official CLI tiling/stochastic replication'}
    OUT.mkdir(parents=True)
    atomic_json(OUT/'config.json',config)
    emit({'phase':'verified','count':24,'official_lora_sha256':config['official_lora_sha256']})
    runner=LatentEditor(lora,torch.cuda.device_count())
    runner.backend.set_trainable_branch(None);runner.backend.eval()
    # Official checkpoint omits native projection LoRA. Zero-B means exact identity residual.
    projection=runner.backend.transformer.proj_out
    if torch.count_nonzero(projection.lora_B['default'].weight)!=0:
        raise RuntimeError('Official omitted proj_out adapter must have zero B')
    if any(p.requires_grad for p in runner.backend.parameters()):raise RuntimeError('Expected frozen inference')
    emit({'phase':'loaded','official_proj_out_zero_residual':True})
    started=time.monotonic()
    for n,sid in enumerate(ids,1):
        image=read(DATA/'blended'/f'{sid}.png').to(runner.primary)
        z=deterministic_encode(runner.backend,image*2-1)
        edited=runner.forward(z)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            pred=(runner.backend.upstream.decode(edited.to(runner.backend.vae.dtype),runner.backend.vae).float()+1)*.5
        if not torch.isfinite(pred).all():raise RuntimeError('Nonfinite output '+sid)
        save_png(pred,OUT/'predictions'/f'{sid}.png')
        emit({'phase':'prediction','done':n,'total':24,'id':sid})
        del image,z,edited,pred
    rows=[]
    for sid in ids:
        prediction=read(OUT/'predictions'/f'{sid}.png')
        gt=read(DATA/'transmission_layer'/f'{sid}.png')
        source=read(DATA/'blended'/f'{sid}.png')
        if prediction.shape!=gt.shape:raise RuntimeError('Geometry mismatch')
        rows.append({'id':sid,'width':gt.shape[-1],'height':gt.shape[-2],
                     'input_sha256':records[sid]['processed']['input']['sha256'],
                     'gt_sha256':records[sid]['processed']['gt']['sha256'],
                     'prediction_sha256':sha(OUT/'predictions'/f'{sid}.png'),**metrics(prediction,gt,source)})
    means={k:sum(r[k] for r in rows)/24 for k in KEYS}
    with open(OUT/'metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'complete':True,'count':24,'split':'sealed test','model':'Official WindowSeat',
            'means':means,'per_image':rows,'config':config,'elapsed_seconds':time.monotonic()-started}
    atomic_json(OUT/'metrics.json',report)
    comparisons={'WindowSeat official':{'means':means,'delta':{k:0 for k in KEYS}}}
    source=PROJECT/'results_archive/M9/m9_latent_only_e4_sealed_test/summary.json'
    prior_config=json.loads((source.parent/'config.json').read_text())
    if prior_config['identity']['dataset_sha256']!=config['dataset_manifest_sha256']:
        raise RuntimeError('M4/M9 dataset identity differs')
    prior=json.loads(source.read_text())
    for key,label in (('m4','M4 final best'),('best','M9 best (M4 initialization)'),('latest','M9 latest epoch4')):
        old=prior['reports'][key]
        if {r['id'] for r in old['per_image']}!=set(ids):raise RuntimeError('Different comparison sample set')
        comparisons[label]={'means':old['means'],'delta':{k:old['means'][k]-means[k] for k in KEYS}}
    summary={'complete':True,'count':24,'protocol':config,'comparisons':comparisons}
    atomic_json(OUT/'comparison.json',summary)
    lines=['# 原生 WindowSeat：当前24张封存测试','',
           '官方发布权重，无个人微调。与M4/M9使用相同处理后输入、固定文本、NF4主干、BF16激活、posterior mode VAE编码、整图一步更新及保存后RGB8指标。',
           '不重新缩放、裁剪、填充或切块；GT只用于评价。本轮目标是统一模型对照，不是复现官方CLI切块和随机采样协议。',
           '', '| 模型 | PSNR / dB | SSIM | 相对原生ΔPSNR | 相对原生ΔSSIM |', '|---|---:|---:|---:|---:|']
    for label,r in comparisons.items():
        m,d=r['means'],r['delta'];lines.append(f"| {label} | {m['psnr']:.4f} | {m['ssim']:.6f} | {d['psnr']:+.4f} | {d['ssim']:+.6f} |")
    lines+=['', '本表覆盖同一24张，包括现有清单中的55_1672_1162；不使用之前历史9张或12张过滤子集的均值。',
            '', '全部24张预测PNG：predictions/。逐图metrics.csv；权重、数据、输入GT输出哈希和协议见config.json、metrics.json。',
            '', f'结果目录：`{OUT}`。M4/M9为前次已完成的同协议结果，本次没有重新训练或重测这些模型。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    emit({'phase':'complete','means':means,'comparisons':comparisons})


if __name__=='__main__':main()
