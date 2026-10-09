"""Summarize completed B4 and matched fixed-model evaluations without retraining."""
import argparse,csv,json,math
from pathlib import Path
from .sma_eval import ROOT

def read(path):return json.loads(path.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);a=ap.parse_args()
    run=a.run_dir;summary=read(run/'training_summary.json');config=read(run/'run_config.json')
    assert summary['status']=='complete' and summary['joint_lora']
    assert summary['epochs_completed']==config['args']['epochs'] and summary['optimizer_updates']==51*config['args']['epochs']
    events=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
    train=[x for x in events if x['kind']=='train'];vals=[x for x in events if x['kind']=='validation']
    assert len(train)==summary['optimizer_updates'] and len(vals)==config['args']['epochs']
    assert all(x['student_lora_grad_norm']>0 and x['reader_grad_norm']>0 and x['backbone_vae_memory_frozen'] and x['noisy_gt_samples_in_update']==0 for x in train)
    curves=[]
    for v in vals:
        ts=[x for x in train if x['epoch']==v['epoch']]
        curves.append({'epoch':v['epoch']+1,'step':v['step'],**v['means'],
            **{k:sum(x[k] for x in ts)/len(ts) for k in ('rec_i','rec_p90','student_lora_grad_norm','reader_grad_norm','actual_aux_base_ratio')}})
    with (run/'epoch_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    project=ROOT/'RMagNet';old=project/'runs/sma_dataset2_e50'
    test={'M4-best':read(old/'test_m4best/evaluation.json'),
        'SMA50-best':read(old/'test_best/evaluation.json'),
        'B4-best':read(run/'test_best/evaluation.json'),'B4-latest':read(run/'test_latest/evaluation.json')}
    real={'M4-best':read(project/'runs/real20_sma_dataset2_e50/m4best/evaluation.json'),
        'SMA50-best':read(project/'runs/real20_sma_dataset2_e50/best/evaluation.json'),
        'B4-best':read(run/'real20_best/evaluation.json'),'B4-latest':read(run/'real20_latest/evaluation.json')}
    ids={x['id'] for x in test['M4-best']['per_image']}
    for report in test.values():
        assert {x['id'] for x in report['per_image']}==ids and len(ids)==26
        assert report['dataset_manifest_sha256']==test['M4-best']['dataset_manifest_sha256']
    for report in real.values():assert len(report['per_image'])==20 and report['vae_posterior']=='mode'
    result={'experiment':'SMA-B4-joint','status':'complete','training':summary,'epoch_metrics':curves,
        'test26':{k:v['means'] for k,v in test.items()},
        'test_by_source':{k:v['by_source'] for k,v in test.items()},
        'real20':{k:v['means'] for k,v in real.items()},
        'controls':'M4 fixed; SMA50-best is historical reference, not matched 4-epoch schedule control',
        'peak_cuda_gib':max(x['peak_allocated_gib'] for x in train)}
    rows=[]
    for dataset,reports in [('test26',test),('real20',real)]:
        for model,report in reports.items():
            for item in report['per_image']:
                rows.append({'dataset':dataset,'model':model,'id':item['id'],**{k:item[k] for k in ('l1','psnr','ssim')}})
    with (run/'comparison_per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 实验 B4：LoRA 与 SMA 联合训练结果','',
        f"- 完成 {summary['epochs_completed']} epoch / {summary['optimizer_updates']} 更新；初始化为 M4-best 与 fresh SMA readers。",
        f"- 可训练 LoRA {config['student_lora_parameters']:,} 参数，读取模块 {config['reader_parameters']:,} 参数。",
        f"- best step {summary['best_metrics']['step']}；latest step {summary['latest_metrics']['step']}。",
        f"- 四卡训练，主rank显存峰值 {result['peak_cuda_gib']:.3f} GiB。固定教师/主干/VAE/记忆梯度检查通过。",'']
    for title,reports in [('26张封存测试',test),('real20（20张）',real)]:
        lines += ['## '+title,'','| 模型 | L1 ↓ | PSNR ↑ | SSIM ↑ |','|---|---:|---:|---:|']
        for name,r in reports.items():
            m=r['means'];lines.append(f"| {name} | {m['l1']:.6f} | {m['psnr']:.4f} | {m['ssim']:.6f} |")
        lines.append('')
    lines+=['## 验证曲线','', '| Epoch | PSNR | SSIM | L1 | LPIPS |','|---|---:|---:|---:|---:|']
    for c in curves:lines.append(f"| {c['epoch']} | {c['psnr']:.4f} | {c['ssim']:.6f} | {c['l1']:.6f} | {c['lpips_squeeze']:.6f} |")
    lines+=['','## 结论边界','']
    for title,reports in [('封存测试',test),('real20',real)]:
        m=reports['M4-best']['means']
        for choice in ('B4-best','B4-latest'):
            v=reports[choice]['means'];lines.append(f"- {title} {choice} 相对固定 M4：PSNR {v['psnr']-m['psnr']:+.4f} dB，SSIM {v['ssim']-m['ssim']:+.6f}，L1 {v['l1']-m['l1']:+.6f}。")
    lines+=['','本轮只实际训练 B4；SMA50-best 为历史参考，cosine轨迹与本轮不同。未运行严格S4/A4，不能把任何收益全部归因于语义与编辑协同，也不能把负结果完全归因于语义层选择。',
        '',f'结果目录：`{run}`。联合权重位于 `best_sma.safetensors` 和 `latest_sma.safetensors`，同时包含可训练学生LoRA与完整SMA，不能使用旧版仅SMA载入方式。',
        '', '`epoch_metrics.csv` 为逐epoch训练/验证记录；`comparison_per_image.csv` 为逐图对比；`test_best/`、`test_latest/`、`real20_best/`、`real20_latest/` 含预测与评价身份记录。','']
    (run/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    (run/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    docs=project/'docx/SMA_JointBCbranch'
    (docs/'B4_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    (docs/'materials/B4_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    import shutil
    shutil.copy2(run/'epoch_metrics.csv',docs/'materials/B4_EPOCH_METRICS.csv')
    shutil.copy2(run/'comparison_per_image.csv',docs/'materials/B4_COMPARISON_PER_IMAGE.csv')
    print(json.dumps({k:result[k] for k in ('status','test26','real20','peak_cuda_gib')},indent=2))
if __name__=='__main__':main()
