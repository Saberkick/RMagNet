"""Evaluate AfterFullM2-B1 best and latest on the sealed corrected M2 test split."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import lpips,torch
from .m2_compare_report import load_adapter,measure,pil_to_tensor,save_png,sha256,mean_metrics
from .m2a_data_baseline import load_m2_manifest
from .qwen_backend import QwenSharedBackend
from .stage1_train import image_tensor
ROOT=Path('/share/linmingheng-local/xuke'); PROJECT=ROOT/'RMagNet'; DATA=ROOT/'datasets/rmagnet_m2_aspect'; RUN=PROJECT/'runs/AfterFullM2-B1'
VARIANTS={'best':RUN/'best_transmission_lora.safetensors','latest':RUN/'latest_transmission_lora.safetensors'}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=DATA); p.add_argument('--output-dir',type=Path,default=RUN/'sealed_test_best_latest'); p.add_argument('--device',default='cuda:0'); p.add_argument('--seed',type=int,default=2026); a=p.parse_args()
 if a.output_dir.exists(): raise FileExistsError(a.output_dir)
 a.output_dir.mkdir(parents=True)
 _,records,splits=load_m2_manifest(a.data_root); ids=splits['test']
 if len(ids)!=18: raise RuntimeError('sealed test split must contain 18 images')
 for x in VARIANTS.values():
  if not x.is_file(): raise FileNotFoundError(x)
 device=torch.device(a.device); backend=QwenSharedBackend.from_local(device); backend.set_trainable_branch('transmission'); backend.transformer.eval(); backend.vae.eval()
 percept=lpips.LPIPS(net='squeeze',verbose=False).eval().cpu()
 for x in percept.parameters(): x.requires_grad_(False)
 rows=[]; hashes={}
 with torch.inference_mode():
  for name,ckpt in VARIANTS.items():
   load_adapter(backend,ckpt,device); hashes[name]={'path':str(ckpt),'sha256':sha256(ckpt)}; torch.manual_seed(a.seed); torch.cuda.manual_seed(a.seed)
   for n,sid in enumerate(ids,1):
    norm=image_tensor(a.data_root/'blended'/f'{sid}.png').unsqueeze(0).to(device); pred=((backend.forward_normalized(norm,'transmission').float()+1)*.5).clamp(0,1).cpu(); out=a.output_dir/'predictions'/name/f'{sid}.png'; save_png(pred,out); saved=pil_to_tensor(out)
    inp=((image_tensor(a.data_root/'blended'/f'{sid}.png').unsqueeze(0)+1)*.5); gt=((image_tensor(a.data_root/'transmission_layer'/f'{sid}.png').unsqueeze(0)+1)*.5); m=measure(saved,gt,inp,percept); rows.append({'id':sid,'variant':name,'bucket':records[sid]['aspect_bucket'],'width':saved.shape[-1],'height':saved.shape[-2],**m}); print(json.dumps({'variant':name,'sample':n,'total':len(ids),'id':sid}),flush=True); del norm,pred,saved; torch.cuda.empty_cache()
 with (a.output_dir/'metrics.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 grouped=defaultdict(list)
 for r in rows: grouped[r['variant']].append(r)
 summary={'status':'complete','split':'corrected sealed test','sample_count':len(ids),'test_ids':ids,'seed':a.seed,'metric_domain':'saved 8-bit RGB PNG; macro average over images','weights':hashes,'means':{k:mean_metrics(grouped[k]) for k in VARIANTS}}
 (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
