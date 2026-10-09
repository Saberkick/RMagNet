"""Report completed C runs, without selection on sealed tests."""
import argparse, json, csv
from pathlib import Path
from .sma_eval import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--allow-interrupted',action='store_true');a=p.parse_args()
    config=json.loads((a.run_dir/'run_config.json').read_text())
    logs=[json.loads(x) for x in (a.run_dir/'metrics.jsonl').read_text().splitlines()]
    train=[r for r in logs if r.get('kind')=='train'];val=[r for r in logs if r.get('kind')=='validation']
    summary_path=a.run_dir/'training_summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text())
        if summary['status']!='complete' or summary['optimizer_updates']!=51*config['args']['epochs']:
            raise RuntimeError('C full training budget incomplete')
    elif a.allow_interrupted:
        best=json.loads((a.run_dir/'best_metrics.json').read_text())
        latest=json.loads((a.run_dir/'latest_metrics.json').read_text())
        console=(ROOT/'RMagNet/runs/sma_launch/c1_e10.console.log').read_text()
        code=(ROOT/'RMagNet/runs/sma_launch/c1_e10.exit_code').read_text().strip()
        if code!='1' or 'torch.OutOfMemoryError' not in console or not train or latest['step']!=val[-1]['step']:
            raise RuntimeError('Interrupted-training evidence mismatch')
        summary={'status':'interrupted_OOM','epochs_requested':config['args']['epochs'],
                 'epochs_completed':latest['step']//51,'optimizer_updates':train[-1]['step'],
                 'latest_checkpoint_step':latest['step'],'best_metrics':best,'latest_metrics':latest,
                 'uncheckpointed_updates':train[-1]['step']-latest['step'],
                 'note':'No resumed training; evaluation uses saved best/latest, not last unsaved update'}
        (a.run_dir/'interrupted_training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    else:
        raise RuntimeError('No completed training summary; use explicit --allow-interrupted for saved checkpoints')
    if len(train)!=summary['optimizer_updates'] or not all(r['condition_hooks_active'] and r['student_lora_grad_norm']>0 and r['backbone_vae_memory_frozen'] for r in train):
        raise RuntimeError('C trainable/frozen audit failed')
    calibration=[r for r in train if r['calibration']]
    if not calibration or not all(r['counterfactual_memory_identical'] and r['reliability_gate_gradient_ratio']<=.10001 for r in calibration):
        raise RuntimeError('C calibration missing/invalid')
    b=json.loads((ROOT/'RMagNet/runs/sma_joint_b_e4/comparison.json').read_text())
    tables={};reports={}
    for ds,prefix in [('test26','test'),('real20','real20')]:
        tables[ds]={k:v for k,v in b[ds].items() if k in ('M4-best','SMA50-best','B4-best')}
        for choice in ('best','latest'):
            r=json.loads((a.run_dir/f'{prefix}_{choice}/evaluation.json').read_text());reports[f'{ds}_{choice}']=r
            if r['sample_count']!=(26 if ds=='test26' else 20):raise RuntimeError('Evaluation sample mismatch')
            tables[ds][f'C1-{choice}']=r['means']
    interventions={mode:json.loads((a.run_dir/f'validation_condition_{mode}/evaluation.json').read_text())['means'] for mode in ('off','on')}
    out={'status':'complete','training':summary,'comparisons':tables,'validation_interventions':interventions,'calibration_updates':len(calibration),
         'last_condition_diagnostics':{k:train[-1][k] for k in ('condition_gamma','gate_mean')},
         'last_calibration':calibration[-1], 'controls':'B4 and SMA50 are historical unequal-schedule references; C0 not run'}
    (a.run_dir/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
    lines=['# C1：语义条件 LoRA 与编辑收益门控实验结果','',f"训练状态 {summary['status']}；已记录 {len(train)} 更新，latest保存于step {summary.get('latest_checkpoint_step', len(train))}；best step {summary['best_metrics']['step']}。",'']
    for ds,models in tables.items():
        lines += [f'## {ds}','','| 模型 | PSNR ↑ | SSIM ↑ | L1 ↓ |','|---|---:|---:|---:|']
        for name,m in models.items():lines.append(f"| {name} | {m['psnr']:.4f} | {m['ssim']:.6f} | {m['l1']:.6f} |")
        lines += ['']
    lines += ['## 验证曲线','','| Epoch | PSNR | SSIM | L1 |','|---|---:|---:|---:|']
    for r in val:
        m=r['means'];lines.append(f"| {r['step']//51} | {m['psnr']:.4f} | {m['ssim']:.6f} | {m['l1']:.6f} |")
    lines += ['', '## 同一best模型的验证集条件干预', '', '| 条件 | PSNR | SSIM | L1 |', '|---|---:|---:|---:|']
    best=summary['best_metrics']; lines.append(f"| learned | {best['val_psnr']:.4f} | {best['val_ssim']:.6f} | {best['val_l1']:.6f} |")
    for mode,m in interventions.items(): lines.append(f"| {mode} | {m['psnr']:.4f} | {m['ssim']:.6f} | {m['l1']:.6f} |")
    lines += ['',f'有效校准更新：{len(calibration)}。最后门控均值：{train[-1]["gate_mean"]:.6f}。',
              '同一best模型的验证集off/on对照检验条件路径贡献，不等同于独立训练的C0；门控不代表已校准的语义正确概率。',
              '本轮计划10 epoch（中断时只评估已保存权重），B4为4 epoch；没有C0或同预算LoRA-only对照，不能单凭指标将收益归因于门控。',
              '预测结果与逐图指标位于 test_best/、test_latest/、real20_best/、real20_latest/。','']
    text='\n'.join(lines);(a.run_dir/'REPORT.md').write_text(text)
    doc=ROOT/'RMagNet/docx/SMA_Cbranch';doc.mkdir(exist_ok=True)
    (doc/'C1_RESULTS.md').write_text(text)
    (doc/'C1_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':'complete','comparisons':tables},indent=2))
if __name__=='__main__':main()
