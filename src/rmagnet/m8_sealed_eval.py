"""Evaluate fixed M8 best/latest and fresh M4 on the same sealed test IDs."""
import argparse,csv,json,os,time
from pathlib import Path
import torch
import safetensors.torch as st
from safetensors import safe_open
from .m8_spatial_gates import FrozenM4Gates,VERSION,LAYERS,gate_previews
from .m7_latent_tail import EXPECTED_M4,sha,atomic_json,emit,event
from .m5_train import read,save_png,metrics,KEYS
from .sma_data import load_manifest

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    for k in ('data','initial','best','latest','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(2026);torch.cuda.set_device(0)
    cards=torch.cuda.device_count()
    if cards not in (2,3,4):raise RuntimeError('Use 2-4 model-parallel GPUs')
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError(a.output)
    if sha(a.initial)!=EXPECTED_M4:raise RuntimeError('Wrong frozen M4')
    _,records,splits=load_manifest(a.data);ids=splits['test']
    ident={'version':VERSION,'m4_sha256':EXPECTED_M4,'dataset_sha256':sha(a.data/'manifest.json'),'layers':list(LAYERS)}
    states={};checkpoint_info={}
    for label,path,expected_epoch in [('best',a.best,2),('latest',a.latest,5)]:
        with safe_open(path,framework='pt') as f:md=f.metadata()
        if json.loads(md['identity'])!=ident:raise RuntimeError('Checkpoint identity mismatch '+label)
        cursor=json.loads(md['cursor'])
        if cursor['epoch']!=expected_epoch:raise RuntimeError('Unexpected fixed checkpoint epoch '+label)
        states[label]=st.load_file(path,device='cpu')
        checkpoint_info[label]={'path':str(path),'sha256':sha(path),'cursor':cursor}
    a.output.mkdir(parents=True)
    atomic_json(a.output/'config.json',{'identity':ident,'checkpoint_info':checkpoint_info,
        'physical_gpus':os.environ.get('CUDA_VISIBLE_DEVICES'),'cards':cards,'test_ids':ids,'count':len(ids),
        'input':'I only; GT/P90/DoLP never supplied to model','metric_domain':'saved RGB8 PNG, macro means',
        'ssim':'project uniform11x11, zero-pad5','resolution':'existing processed dynamic sizes; no resize/crop',
        'M4_baseline':'fresh same-model forward with all spatial gates disabled','training':False})
    runner=FrozenM4Gates(a.initial,cards);runner.gates.requires_grad_(False).eval()
    if any(p.requires_grad for p in runner.backend.parameters()):raise RuntimeError('Frozen model changed')
    versions=[('m4','M4 final best'),('best','M8 best, epoch2'),('latest','M8 latest, epoch5')]
    rows={key:[] for key,_ in versions};started=time.monotonic()
    for n,sid in enumerate(ids,1):
        image=read(a.data/'blended'/f'{sid}.png').to(runner.primary)
        for key,label in versions:
            runner.enabled=key!='m4'
            if runner.enabled:runner.gates.load_state_dict(states[key],strict=True)
            prediction,latent=runner.forward(image)
            if not torch.isfinite(prediction).all() or not torch.isfinite(latent).all():raise RuntimeError('Nonfinite prediction')
            path=a.output/key/'predictions'/f'{sid}.png';save_png(prediction,path)
            if runner.enabled:
                event(a.output/key/'gate_statistics.jsonl',{'id':sid,'stats':runner.stats()})
                if n<=3:gate_previews(runner,a.output/key,sid,(image.shape[-1],image.shape[-2]))
            runner.clear();del prediction,latent
        # Load GT only after all three predictions are generated and saved.
        gt=read(a.data/'transmission_layer'/f'{sid}.png');source=image.cpu()
        for key,label in versions:
            saved=read(a.output/key/'predictions'/f'{sid}.png')
            if saved.shape!=gt.shape:raise RuntimeError('Prediction/GT geometry mismatch '+sid)
            values=metrics(saved,gt,source)
            rows[key].append({'id':sid,**values})
        emit({'phase':'sealed_sample','done':n,'total':len(ids),'id':sid,
              'psnr':{key:rows[key][-1]['psnr'] for key,_ in versions},'elapsed_seconds':time.monotonic()-started})
        del image,gt,source,saved
    reports={}
    for key,label in versions:
        values=rows[key];means={k:sum(v[k] for v in values)/len(values) for k in KEYS}
        reports[key]={'model':label,'count':len(values),'means':means,'per_image':values}
        if key!='m4':
            reports[key]['minus_m4']={k:means[k]-reports['m4']['means'][k] for k in KEYS}
            reports[key]['wins']={k:sum(v[k]>base[k] for v,base in zip(values,rows['m4'])) for k in ('psnr','ssim')}
        atomic_json(a.output/key/'metrics.json',reports[key])
        with open(a.output/key/'metrics.csv','w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    summary={'complete':True,'split':'sealed test','count':len(ids),'reports':reports,
             'checkpoint_info':checkpoint_info,'elapsed_seconds':time.monotonic()-started,'training':False}
    atomic_json(a.output/'summary.json',summary)
    table='\n'.join(f"| {r['model']} | {r['count']} | {r['means']['psnr']:.4f} | {r['means']['ssim']:.6f} | {r['means']['l1']:.6f} | {r['means']['edge_l1']:.6f} |" for r in reports.values())
    comparison='\n'.join(f"- {reports[k]['model']}：PSNR {reports[k]['minus_m4']['psnr']:+.4f} dB、SSIM {reports[k]['minus_m4']['ssim']:+.6f}；逐图PSNR胜出 {reports[k]['wins']['psnr']}/{len(ids)}，SSIM胜出 {reports[k]['wins']['ssim']}/{len(ids)}。" for k in ('best','latest'))
    (a.output/'REPORT.md').write_text(f'''# M8 best/latest 封存测试

同一组{len(ids)}张纠正标签后的封存图。best由验证集L1选定为epoch2，latest为epoch5；均在读取本轮封存结果前固定。
输入仅普通I，保留现有动态尺寸；GT只用于评价。冻结M4、NF4主干、BF16激活、FP32门控；无训练、无教师缓存。
M4、best、latest在同一模型/同次任务下生成；模型输出保存成RGB8 PNG后统一计算逐图宏平均。SSIM为项目uniform11×11、zero-pad5。

| 模型 | 图数 | PSNR / dB | SSIM | L1 | 边缘L1 |
|---|---:|---:|---:|---:|---:|
{table}

{comparison}

所有模型均保留24张PNG、逐图metrics.csv；best/latest还记录门控统计和固定前三张灰度门控图。
运行目录：`{a.output}`。预测位于`m4/predictions/`、`best/predictions/`、`latest/predictions/`；身份与哈希见config.json。
本轮未根据封存测试结果更换checkpoint，未运行real20。
''')
    emit({'phase':'complete','count':len(ids),'results':{k:{x:r[x] for x in ('means','minus_m4','wins') if x in r} for k,r in reports.items()}})

if __name__=='__main__':main()
