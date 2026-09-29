"""Rebuild the M2 Q20 cache from the compatible base-teacher M4 cache."""
from __future__ import annotations
import argparse, hashlib, json, os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import safetensors.torch
import torch
from PIL import Image
from .c1_l20_prepare import resize_float, robust_unit
from .qwen_layer_probe import cosine_map

PROJECT=Path(__file__).resolve().parents[2]
DEFAULT_DATA=Path('/share/linmingheng-local/xuke/datasets/rmagnet_m2_aspect')
DEFAULT_SOURCE=PROJECT/'data_cache/m4_multilayer_v1'
DEFAULT_OUTPUT=PROJECT/'data_cache/m2a_q20'
FORMULA_VERSION='m2a-q20-variable-aspect-v1'

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(1024*1024): h.update(b)
 return h.hexdigest()

def atomic_json(path:Path,obj:dict)->None:
 tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)+'\n'); os.replace(tmp,path)

def main()->None:
 p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=DEFAULT_DATA); p.add_argument('--source',type=Path,default=DEFAULT_SOURCE); p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT); a=p.parse_args()
 data=a.data_root.resolve(); source=a.source.resolve(); output=a.output.resolve()
 dm=json.loads((data/'manifest.json').read_text()); sm=json.loads((source/'manifest.json').read_text())
 ids=(data/'splits/train.txt').read_text().split(); byid={r['id']:r for r in dm['samples']}; sbyid={r['id']:r for r in sm['samples']}
 dsha=sha256(data/'manifest.json')
 if not dm.get('complete') or dm.get('version')!='m2-variable-aspect-v2-corrected-labels': raise RuntimeError('incompatible M2 dataset')
 if not sm.get('complete') or sm.get('cache_version')!='m4-multilayer-v1': raise RuntimeError('source is not base-teacher M4 cache')
 if sm['source_dataset']['manifest_sha256']!=dsha or sm['source_dataset']['train_ids']!=ids: raise RuntimeError('source cache dataset/split differs')
 if sm['teacher']['adapter']!='all LoRA disabled' or sm['teacher']['flow_timestep']!=499 or 20 not in sm['teacher']['early_blocks']: raise RuntimeError('source Q20 teacher identity differs')
 if output.exists() and any(output.iterdir()): raise FileExistsError(output)
 (output/'gt_features').mkdir(parents=True); (output/'weights').mkdir(); (output/'previews').mkdir()
 records=[]
 for n,sid in enumerate(ids,1):
  sr=sbyid[sid]; dr=byid[sid]; src=source/sr['cache']; tensors=safetensors.torch.load_file(src)
  qi=tensors['q20_input']; qg=tensors['q20_gt']; gh,gw=map(int,tensors['token_grid_hw'].tolist()); w,h=dr['target_size']
  if (gh,gw)!=(h//16,w//16) or qi.shape!=qg.shape or qi.shape!=(gh*gw,3072): raise RuntimeError(f'shape mismatch {sid}')
  dqraw=cosine_map(qi[None].float(),qg[None].float(),(gh,gw)); dq,qlo,qhi=robust_unit(dqraw,.02,.98)
  with Image.open(data/'dolp'/f'{sid}.png') as im: dolp=np.asarray(im,dtype=np.float32)/255.0
  dt=np.clip(resize_float(dolp,(gw,gh)),0,1); score=dq*(.7+.3*dt); raw=np.clip(1+2*score,1,3); wt=raw/float(raw.mean()); wp=resize_float(wt,(w,h)); wp/=float(wp.mean())
  if not all(np.isfinite(x).all() for x in (dq,dt,score,wt,wp)) or not np.isclose(wt.mean(),1,atol=2e-6) or not np.isclose(wp.mean(),1,atol=2e-6): raise RuntimeError(f'invalid weight {sid}')
  frel=Path('gt_features')/f'{sid}.safetensors'; wrel=Path('weights')/f'{sid}.npz'
  safetensors.torch.save_file({'q20_gt':qg.contiguous()},output/frel)
  with (output/wrel).open('wb') as f: np.savez_compressed(f,weight_pixel=wp.astype(np.float16),weight_token=wt.astype(np.float16))
  paths={'input':data/'blended'/f'{sid}.png','gt':data/'transmission_layer'/f'{sid}.png','dolp':data/'dolp'/f'{sid}.png'}
  if sha256(paths['input'])!=sr['source_sha256']['input'] or sha256(paths['gt'])!=sr['source_sha256']['gt']: raise RuntimeError(f'source hash mismatch {sid}')
  records.append({'id':sid,'group':dr['group'],'aspect_bucket':dr['aspect_bucket'],'source':{k:{'path':str(v),'sha256':sha256(v)} for k,v in paths.items()},'image_size_wh':[w,h],'token_grid_hw':[gh,gw],'q20_feature_shape':[gh*gw,3072],'q_difference':{'normalization_low':qlo,'normalization_high':qhi},'weight_stats':{'token_mean':float(wt.mean()),'pixel_min':float(wp.min()),'pixel_max':float(wp.max()),'pixel_mean':float(wp.mean())},'memory_gib':{'peak_allocated':0.0,'peak_reserved':0.0},'cache':{'gt_feature':str(frel),'gt_feature_sha256':sha256(output/frel),'weight':str(wrel),'weight_sha256':sha256(output/wrel),'preview_panel':None}})
  if n%12==0 or n==len(ids): print(json.dumps({'converted':n,'total':len(ids)}),flush=True)
 identity={'formula_version':FORMULA_VERSION,'data_root':str(data),'data_manifest_sha256':dsha,'selected_ids':ids,'selection':'full-train-split','q_low_quantile':.02,'q_high_quantile':.98,'block_zero_based_index':19,'flow_timestep':499}
 manifest={'schema_version':1,'complete':True,'identity':identity,'created_at_utc':datetime.now(timezone.utc).isoformat(),'completed_at_utc':datetime.now(timezone.utc).isoformat(),'project_git_commit':'reused-compatible-m4-cache','source_dataset':{'manifest':str(data/'manifest.json'),'manifest_sha256':dsha,'split':'train','full_train_count':len(ids),'cached_count':len(records)},'qwen_feature':{'block_one_based':20,'block_zero_based_index':19,'flow_timestep':499,'hidden_size':3072,'adapters':'all LoRA adapters disabled','stored_gt_dtype':'bfloat16','reuse_source':str(source)},'weighting':{'score_formula':'S = D_Q * (0.7 + 0.3 * D_DoLP)','raw_weight_formula':'W_raw = clip(1 + 2*S, 1, 3)','final_weight_formula':'mean-one token W; bilinear pixel W; mean-one again'},'aspect_bucket_counts':dict(Counter(r['aspect_bucket'] for r in records)),'memory':{'batch_size':1,'single_gpu':False,'max_peak_allocated_gib':0.0,'max_peak_reserved_gib':0.0},'samples':records}
 atomic_json(output/'manifest.json',manifest); print(json.dumps({'status':'complete','samples':len(records),'output':str(output),'manifest_sha256':sha256(output/'manifest.json')},indent=2))
if __name__=='__main__': main()
