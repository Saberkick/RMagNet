"""Archive the specific blocked native run; remove only validated redundant paths."""
import argparse,json,shutil
from pathlib import Path
from .m4_cache import sha256

PROJECT=Path('/share/linmingheng-local/xuke/RMagNet')
def main():
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');a=p.parse_args()
    old=PROJECT/'runs/ws_c1_m5_filtered_e50'
    oldcache=PROJECT/'data_cache/ws_c1_m5_filtered_e50_layers'
    newcache=PROJECT/'data_cache/ws_c1_m5_dataset3_s1_e20_layers'
    archive=PROJECT/'results_archive/WindowSeat_C1_M5/blocked_e50_20261010'
    cm=json.loads((newcache/'manifest.json').read_text())
    assert cm['complete'] and cm['source_dataset']['sample_count']==204
    for r in cm['samples']:assert sha256(newcache/r['cache'])==r['cache_sha256']
    archive.mkdir(parents=True,exist_ok=True);report={'reason':'old pipeline stopped while waiting for four GPUs; no C1 training log or trained checkpoint','removed':[]}
    if old.exists():
        assert old.resolve()==PROJECT/'runs/ws_c1_m5_filtered_e50' and not old.is_symlink()
        assert not (old/'c1').exists() and not (old/'m5').exists()
        for f in old.rglob('*'):
            if f.is_file() and f.suffix!='.safetensors':
                dst=archive/'run_records'/f.relative_to(old);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
        report['removed'].append(str(old))
        if a.apply:shutil.rmtree(old)
    if oldcache.exists():
        assert oldcache.resolve()==PROJECT/'data_cache/ws_c1_m5_filtered_e50_layers' and not oldcache.is_symlink()
        shutil.copy2(oldcache/'manifest.json',archive/'old_cache_manifest.json')
        if (oldcache/'logs').exists():shutil.copytree(oldcache/'logs',archive/'cache_logs',dirs_exist_ok=True)
        report['removed'].append(str(oldcache))
        if a.apply:shutil.rmtree(oldcache)
    report['applied']=a.apply;report['new_cache_sha256']=sha256(newcache/'manifest.json')
    (archive/'cleanup.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
