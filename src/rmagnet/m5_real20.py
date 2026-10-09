"""Native real20 evaluation using audited C1-best PNG intermediates."""
import csv,json,math,subprocess,time
import torch
from PIL import Image,ImageDraw
from .m5_cache import CACHE,C1,PROJECT
from .m5_train import read,save_png,load_model,atomic_json
from .m4_cache import sha256
from .real20_report import metric

DATA=PROJECT.parent/'datasets/liyucs_RAGNet/testsets/real20'
RUN=PROJECT/'runs/m5_c1_pixel_e30'
UPSTREAM=PROJECT/'runs/sma_c1_e10/real20_best'
KEYS=('l1','psnr','ssim')

def main():
 torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda:0');started=time.time()
 identity=json.loads((CACHE/'manifest.json').read_text());up=json.loads((UPSTREAM/'evaluation.json').read_text())
 if sha256(C1)!=identity['checkpoint_sha256'] or up['checkpoint_sha256']!=identity['checkpoint_sha256']:raise RuntimeError('C1 source mismatch')
 commit=subprocess.check_output(['git','-C',str(DATA.parents[1]),'rev-parse','HEAD'],text=True).strip()
 if commit!=up['dataset_commit'] or up['vae_posterior']!='mode':raise RuntimeError('Dataset/inference mismatch')
 records={r['id']:r for r in up['per_image']};ids=sorted(records,key=int)
 if len(ids)!=20 or {p.stem for p in (DATA/'blended').glob('*.jpg')}!=set(ids):raise RuntimeError('real20 IDs changed')
 sizes={};baseline=[]
 for sid in ids:
  r=records[sid];i=DATA/'blended'/f'{sid}.jpg';g=DATA/'transmission_layer'/f'{sid}.jpg';t=UPSTREAM/f'{sid}_windowseat_output.png'
  for path,key in ((i,'input_sha256'),(g,'gt_sha256'),(t,'prediction_sha256')):
   if sha256(path)!=r[key]:raise RuntimeError('real20 source file changed '+sid+'/'+key)
  with Image.open(i) as im,Image.open(g) as gt,Image.open(t) as pred:
   if im.size!=gt.size or im.size!=pred.size or pred.mode!='RGB':raise RuntimeError('Geometry '+sid)
   sizes[sid]=im.size
  score=metric(read(t),read(g));baseline.append({'id':sid,**score})
 base={k:sum(r[k] for r in baseline)/20 for k in KEYS}
 for k in KEYS:
  if abs(base[k]-up['means'][k])>1e-6:raise RuntimeError('C1 metric reproduction mismatch '+k)
 # Largest first; memory failure stops evaluation without changing image scale.
 order=sorted(ids,key=lambda sid:sizes[sid][0]*sizes[sid][1],reverse=True)
 reports={};torch.cuda.reset_peak_memory_stats()
 for choice in ('best','latest'):
  out=RUN/f'real20_{choice}'
  if out.exists():raise FileExistsError(out)
  out.mkdir();model,md=load_model(RUN/f'{choice}.safetensors',identity,device);model.eval();rows=[]
  for n,sid in enumerate(order,1):
   image=read(DATA/'blended'/f'{sid}.jpg').to(device);t0=read(UPSTREAM/f'{sid}_windowseat_output.png').to(device)
   with torch.inference_mode():result=model(image,t0);save_png(result['prediction'],out/'predictions'/f'{sid}.png')
   if not torch.isfinite(result['prediction']).all():raise RuntimeError('Nonfinite prediction '+sid)
   pred=read(out/'predictions'/f'{sid}.png');gt=read(DATA/'transmission_layer'/f'{sid}.jpg')
   if pred.shape!=gt.shape:raise RuntimeError('Output geometry '+sid)
   score=metric(pred,gt)
   if not all(math.isfinite(v) for v in score.values()):raise RuntimeError('Nonfinite metric')
   rows.append({'id':sid,'width':sizes[sid][0],'height':sizes[sid][1],**score,'prediction_sha256':sha256(out/'predictions'/f'{sid}.png')})
   print(json.dumps({'choice':choice,'completed':n,'total':20,'id':sid,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}),flush=True)
   del image,t0,result,pred,gt;torch.cuda.empty_cache()
  rows.sort(key=lambda r:int(r['id']));means={k:sum(r[k] for r in rows)/20 for k in KEYS}
  report={'status':'complete','sample_count':20,'means':means,'c1_baseline':base,'minus_c1':{k:means[k]-base[k] for k in KEYS},
   'psnr_wins':sum(r['psnr']>b['psnr'] for r,b in zip(rows,baseline)),'checkpoint_metadata':md,'checkpoint_sha256':sha256(RUN/f'{choice}.safetensors'),
   'upstream_checkpoint_sha256':identity['checkpoint_sha256'],'dataset_commit':commit,'per_image':rows,
   'upstream_processing':up['tiling'],'m5_processing':'full native-size FP32 forward, no resize/crop/tile, no GT input',
   'metric_domain':'saved RGB8 PNG vs native GT JPEG; per-image macro; existing project SSIM',
   'resolution_shift':'M5 trained at ~200k pixels; native real20 up to 7962624 pixels is cross-resolution evaluation',
   'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
  atomic_json(out/'evaluation.json',report)
  with (out/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  reports[choice]=report;del model;torch.cuda.empty_cache()
 prior=json.loads((PROJECT/'runs/sma_c1_e10/comparison.json').read_text())['comparisons']['real20']
 means={'C1-best':base,**{'C1+M5-'+k:r['means'] for k,r in reports.items()},'M4-best(reference)':prior['M4-best']}
 summary={'status':'complete','count':20,'means':means,'comparisons':{k:{'minus_c1':r['minus_c1'],'psnr_wins':r['psnr_wins']} for k,r in reports.items()},
  'elapsed_seconds':time.time()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
  'note':'Fixed C1-best for both M5 choices; no Qwen inference rerun, no training; historical research dataset, not blind test; native M5 is outside training scale'}
 atomic_json(RUN/'real20_comparison.json',summary)
 rows=[{'model':'C1-best',**r} for r in baseline]
 for k,r in reports.items():rows.extend({'model':'C1+M5-'+k,**{key:x[key] for key in ('id',*KEYS)}} for x in r['per_image'])
 with (RUN/'real20_comparison_per_image.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 panels=RUN/'real20_panels';panels.mkdir()
 for sid in ('22','47','86'):
  paths=[('Input',DATA/'blended'/f'{sid}.jpg'),('GT',DATA/'transmission_layer'/f'{sid}.jpg'),('C1-best',UPSTREAM/f'{sid}_windowseat_output.png')]
  paths += [('M5-'+k,RUN/f'real20_{k}/predictions'/f'{sid}.png') for k in reports]
  canvas=Image.new('RGB',(2400,412),'white');draw=ImageDraw.Draw(canvas)
  for col,(name,path) in enumerate(paths):
   with Image.open(path) as im:im=im.convert('RGB');im.thumbnail((480,360));canvas.paste(im,(col*480+(480-im.width)//2,52+(360-im.height)//2))
   draw.text((col*480+8,8),sid+' '+name,fill='black')
   if name.startswith('M5-'):
    m=next(x for x in reports[name[3:]]['per_image'] if x['id']==sid)
    draw.text((col*480+8,28),f"PSNR {m['psnr']:.4f} SSIM {m['ssim']:.6f}",fill='black')
  canvas.save(panels/f'{sid}.png')
 lines=['# C1 + M5: native real20 results','','| Model | L1 down | PSNR up | SSIM up |','|---|---:|---:|---:|']
 for name,m in means.items():lines.append(f"| {name} | {m['l1']:.6f} | {m['psnr']:.4f} | {m['ssim']:.6f} |")
 lines+=['',summary['note'],'','Upstream PNG/input/GT hashes verified; C1 metrics reproduced within 1e-6. M5 uses complete native images in FP32 without resizing or tiling; no Qwen loaded.','', 'Panels: real20_panels/{22,47,86}.png. Image86 is the largest; 22/47 are the previously discussed car/glare examples.','']
 (RUN/'REAL20_REPORT.md').write_text('\n'.join(lines));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
