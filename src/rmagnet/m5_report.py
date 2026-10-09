"""Summarize the completed C1+M5 sealed-test evaluation, without inference."""
import json,csv,math
from pathlib import Path
from PIL import Image,ImageDraw
from .m5_cache import DATA,CACHE,PROJECT

RUN=PROJECT/'runs/m5_c1_pixel_e30'
def main():
 reports={k:json.loads((RUN/f'eval_test_{k}/evaluation.json').read_text()) for k in ('best','latest')}
 ids=(DATA/'splits/test.txt').read_text().split()
 prior=json.loads((PROJECT/'runs/sma_c1_e10/comparison.json').read_text())['comparisons']['test26']
 for choice,r in reports.items():
  assert r['count']==len(ids)==26 and {x['id'] for x in r['per_image']}==set(ids)
  assert all(math.isfinite(x[k]) for x in r['per_image'] for k in ('l1','psnr','ssim'))
  for k in ('l1','psnr','ssim'):assert abs(r['c1_baseline'][k]-prior['C1-best'][k])<1e-6
 means={'C1-best':reports['best']['c1_baseline'],'C1+M5-best':reports['best']['means'],'C1+M5-latest':reports['latest']['means'],'M4-best(reference)':prior['M4-best']}
 summary={'dataset':'corrected merged sealed test26; retrospective evaluation','means':means,
          'm5_best_minus_c1':reports['best']['minus_c1'],'m5_latest_minus_c1':reports['latest']['minus_c1'],
          'psnr_wins':{k:r['psnr_wins'] for k,r in reports.items()},'best_epoch':3,'latest_epoch':7,'upstream':'Both M5 checkpoints use the same fixed C1-best; not C1-latest',
          'metric_domain':'saved RGB8 PNG, macro per image; existing aspect-preserved processed sizes',
          'history_note':'Previously used research test split with historical initialization scene overlaps; not blind generalization'}
 (RUN/'test_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
 rows=[]
 for choice,r in reports.items():
  rows += [dict(model='C1+M5-'+choice,**x) for x in r['per_image']]
 with (RUN/'test_comparison_per_image.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 selected=[ids[0],ids[len(ids)//2],ids[-1]];paneldir=RUN/'test_panels';paneldir.mkdir(exist_ok=True)
 for sid in selected:
  paths=[('Input',DATA/'blended'/f'{sid}.png'),('GT',DATA/'transmission_layer'/f'{sid}.png'),('C1-best',CACHE/'predictions'/f'{sid}.png')]
  paths += [('M5-'+k,RUN/f'eval_test_{k}/predictions'/f'{sid}.png') for k in reports]
  canvas=Image.new('RGB',(2400,412),'white');draw=ImageDraw.Draw(canvas)
  for col,(name,path) in enumerate(paths):
   with Image.open(path) as im:im=im.convert('RGB');im.thumbnail((480,360));canvas.paste(im,(col*480+(480-im.width)//2,52+(360-im.height)//2))
   draw.text((col*480+8,8),sid+' '+name,fill='black')
   if name.startswith('M5-'):
    m=next(x for x in reports[name[3:]]['per_image'] if x['id']==sid)
    draw.text((col*480+8,28),f"PSNR {m['psnr']:.4f} SSIM {m['ssim']:.6f}",fill='black')
  canvas.save(paneldir/f'{sid}.png')
 lines=['# C1 + M5-R: sealed-test results','', 'Training completed: 7 epochs / 357 updates; early stopping after 4 epochs without improved validation L1; best epoch3, latest epoch7. Both use fixed C1-best.','', '| Model | L1 down | PSNR up | SSIM up |','|---|---:|---:|---:|']
 for name,m in means.items():lines.append(f"| {name} | {m['l1']:.6f} | {m['psnr']:.4f} | {m['ssim']:.6f} |")
 for choice,r in reports.items():
  d=r['minus_c1'];lines += ['',f"M5-{choice} minus C1: PSNR {d['psnr']:+.4f} dB, SSIM {d['ssim']:+.6f}; PSNR wins {r['psnr_wins']}/26."]
 lines+=['','Best is selected on validation macro L1, not test PSNR. This was a small residual restorer, not joint C1 training. No real20 evaluation in this run.','', 'Metrics use saved 8-bit PNG at unchanged processed sizes. SSIM uses the existing uniform-window project convention. Test26 has been used before and historical initialization has scene overlap; these results are retrospective.','', 'Full rows: test_comparison_per_image.csv. Three fixed index-selected panels (first/middle/last test IDs, not selected for gains): test_panels/.','']
 (RUN/'TEST_REPORT.md').write_text('\n'.join(lines))
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
