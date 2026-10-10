"""Crash-safe best/latest bundles for this project's locally generated states."""
import hashlib,json,os,random,shutil
from pathlib import Path
import numpy as np
import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')
    os.replace(tmp,path)


def recover(root,name='latest'):
    root=Path(root);target=root/name;previous=root/f'.previous_{name}'
    if not target.exists() and previous.exists():os.replace(previous,target)
    if target.exists():
        verify(target)
        if previous.exists():shutil.rmtree(previous)
    tmp=root/f'.pending_{name}'
    if tmp.exists():shutil.rmtree(tmp)
    return target


def verify(bundle):
    m=json.loads((bundle/'manifest.json').read_text())
    for filename,digest in m['files'].items():
        if sha(bundle/filename)!=digest:raise RuntimeError(f'Corrupt checkpoint bundle: {bundle}/{filename}')
    return m


def save(root,name,write_model,state,identity,cursor):
    root=Path(root);target=recover(root,name);tmp=root/f'.pending_{name}';tmp.mkdir()
    write_model(tmp/'model.safetensors')
    if state is not None:torch.save(state,tmp/'trainer_state.pt')
    files={p.name:sha(p) for p in tmp.iterdir() if p.is_file()}
    atomic_json(tmp/'manifest.json',{'format':'rmagnet-resume-bundle-v1','identity':identity,'cursor':cursor,'files':files})
    # Old complete checkpoint survives until the replacement has all its files.
    previous=root/f'.previous_{name}'
    if target.exists():os.replace(target,previous)
    os.replace(tmp,target)
    if previous.exists():shutil.rmtree(previous)


def load(root,identity):
    bundle=recover(root);m=verify(bundle)
    if m['identity']!=identity:raise RuntimeError('Resume data/model/recipe identity changed')
    state=torch.load(bundle/'trainer_state.pt',map_location='cpu',weights_only=False)
    if state['cursor']!=m['cursor']:raise RuntimeError('Resume cursor mismatch')
    return bundle,state


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(0)}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda'],0)
