"""Train-only identity and filtered dataset audit before native WindowSeat C1."""
import argparse,json
from pathlib import Path
from .sma_data import load_manifest
from .m4_cache import sha256
from .qwen_backend import check_snapshots
from safetensors import safe_open

OFFICIAL_SHA='2f04225c5be1cac69a5d76faf4c95515d5503210cfee73c0bb1e79309f5a74e9'

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest,records,splits=load_manifest(a.data_root)
    assert len(splits['train']) % 4 == 0 and all(splits.values())
    base,lora=check_snapshots();adapter=lora/'transformer_lora/pytorch_lora_weights.safetensors'
    assert sha256(adapter)==OFFICIAL_SHA
    with safe_open(adapter,framework='pt') as f:count=len(f.keys())
    assert count==1446
    r={'status':'passed','dataset_root':str(a.data_root),'dataset_manifest_sha256':sha256(a.data_root/'manifest.json'),'split_counts':{k:len(v) for k,v in splits.items()},'native_windowseat_adapter':str(adapter),'native_windowseat_sha256':OFFICIAL_SHA,'official_adapter_tensor_count':count,'proj_out_policy':'loader must verify zero residual at runtime','cache_blocks_1_based':[16,20,37,39,41,52,54,56],'cache_timestep':499,'cache_train_only':True,'images_resized_again':False}
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

if __name__=='__main__':main()
