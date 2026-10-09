"""Validate three real20 evaluations and report the matched comparison."""
import csv,json,math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from .real20_infer import ROOT,DATA
from .real20_report import fit

RUN=ROOT/'RMagNet/runs/real20_sma_dataset2_e50'

def main():
    names={'m4best':'M4-best','best':'SMA-best (epoch 2)','latest':'SMA-latest (epoch 50)'}
    reports={k:json.loads((RUN/k/'evaluation.json').read_text()) for k in names}
    ids=sorted([p.stem for p in (DATA/'blended').glob('*.jpg')],key=int)
    rows=[]; per={}
    for key,r in reports.items():
        assert r['status']=='complete' and r['sample_count']==20 and r['vae_posterior']=='mode'
        per[key]={x['id']:x for x in r['per_image']}
        assert set(per[key])==set(ids)
        for metric in ('l1','psnr','ssim'):
            assert all(math.isfinite(float(x[metric])) for x in r['per_image'])
            assert abs(sum(x[metric] for x in r['per_image'])/20-r['means'][metric])<1e-10
    for sid in ids:
        m4=per['m4best'][sid]
        for key in names:
            item=per[key][sid]
            assert item['input_sha256']==m4['input_sha256'] and item['gt_sha256']==m4['gt_sha256']
            rows.append({'id':sid,'variant':names[key],**{k:item[k] for k in ('width','height','l1','psnr','ssim')}})
    with (RUN/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    comparison={}
    for key in ('best','latest'):
        comparison[key]={'minus_m4':{k:reports[key]['means'][k]-reports['m4best']['means'][k] for k in ('l1','psnr','ssim')},
            'wins_out_of_20':{k:sum((per[key][sid][k]<per['m4best'][sid][k]) if k=='l1' else (per[key][sid][k]>per['m4best'][sid][k]) for sid in ids) for k in ('l1','psnr','ssim')}}
    summary={'status':'complete','dataset':'real20','sample_count':20,'means':{names[k]:r['means'] for k,r in reports.items()},'comparison_to_m4':comparison,
        'protocol':{k:reports['m4best'][k] for k in ('dataset_commit','vae_posterior','seed','processing_resolution','tiling','metric_domain','inference_inputs')},
        'checkpoints':{names[k]:{field:r[field] for field in ('checkpoint','checkpoint_sha256','checkpoint_training','peak_cuda_gib','tile_checks')} for k,r in reports.items()}}
    (RUN/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    panels=RUN/'panels';panels.mkdir(exist_ok=True)
    for sid in ids:
        columns=[('Input',DATA/'blended'/f'{sid}.jpg'),('GT',DATA/'transmission_layer'/f'{sid}.jpg')]+[(names[k],RUN/k/f'{sid}_windowseat_output.png') for k in names]
        panel=Image.new('RGB',(480*5,418),'white');draw=ImageDraw.Draw(panel)
        for n,(name,path) in enumerate(columns):
            draw.text((n*480+8,8),f'{sid}  {name}',fill='black')
            if n>=2:
                item=per[list(names)[n-2]][sid]
                draw.text((n*480+8,29),f"PSNR {item['psnr']:.3f} SSIM {item['ssim']:.5f}",fill='black')
            panel.paste(fit(path),(n*480,58))
        panel.save(panels/f'{sid}.png')
    lines=['# real20：合并数据 SMA-best / SMA-latest 与 M4-best','',
        '- SMA 来自 `runs/sma_dataset2_e50`：best 为 epoch 2 / step 102，latest 为 epoch 50 / step 2550。',
        '- 三个模型均采用确定性 VAE posterior mode、种子 2026、官方短边分块和 Lanczos 拼接。',
        '- 对原始尺寸的 20 张保存后 RGB 8 位 PNG 与 GT JPEG 计算逐图指标，再取宏平均。',
        '- 测试时只输入普通图 I；GT 仅评分，不输入 P90、DoLP 或训练语义缓存。','',
        '| 模型 | L1 ↓ | PSNR ↑ | SSIM ↑ |','|---|---:|---:|---:|']
    for key in names:
        m=reports[key]['means'];lines.append(f"| {names[key]} | {m['l1']:.6f} | {m['psnr']:.4f} | {m['ssim']:.6f} |")
    lines+=['','## 相对同口径 M4-best','']
    for key in ('best','latest'):
        c=comparison[key];d=c['minus_m4'];w=c['wins_out_of_20']
        lines.append(f"- {names[key]}：PSNR {d['psnr']:+.4f} dB，SSIM {d['ssim']:+.6f}，L1 {d['l1']:+.6f}；逐图 PSNR 胜出 {w['psnr']}/20，SSIM 胜出 {w['ssim']}/20。")
    lines+=['','## 结论','',
        'SMA-best 相对同口径 M4-best 的平均 PSNR 下降 0.1208 dB、SSIM 下降 0.001248；20 张中有 7 张 PSNR 更高。SMA-latest 平均 PSNR 下降 1.1269 dB、SSIM 下降 0.013387，仅 4 张 PSNR 更高。',
        '本轮未显示 real20 上的整体泛化收益。训练到 50 epoch 的版本退化更明显，与此前合并封存测试的退化方向一致；这不能单独确定是 SMA 结构、训练数据分布还是训练时长造成。',
        'SSIM 沿用项目原 real20 实现：RGB [0,1]，11×11 均值窗口、边界补零，C1=0.01²、C2=0.03²。并非另换一种 SSIM 实现。',
        '', '## 历史口径说明','',
        '本次 M4-best 重新推理，VAE 改用与 SMA 相同的 posterior mode。历史 `runs/real20_windowseat_m4` 使用 posterior sample，不能将两次差异完全归因于 SMA。历史 WindowSeat / M0 / M4 表仍保留在原目录，未覆盖。',
        '', '## 文件','',
        '- 服务器根目录：`/share/linmingheng-local/xuke/RMagNet/runs/real20_sma_dataset2_e50/`。',
        '- `best/`、`latest/`、`m4best/`：20 张预测 PNG、逐图 `metrics.csv`、含权重和数据哈希的 `evaluation.json`。',
        '- `summary.json`：宏平均、差值、逐图胜出数与模型身份。',
        '- `metrics.csv`：三个模型的 60 行逐图指标。',
        '- `panels/`：20 张 Input / GT / M4-best / SMA-best / SMA-latest 五列对比图，图上标注 PSNR/SSIM。',
        '- 启动脚本：`bash scripts/eval_sma_real20.sh`；已有输出时拒绝覆盖。','']
    (RUN/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    doc=ROOT/'RMagNet/docx/SMAbranch'
    (doc/'SMA_DATASET2_REAL20_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    (doc/'materials/SMA_DATASET2_REAL20_RESULTS.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
