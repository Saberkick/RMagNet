"""Checkpoint parity and temporary-condition sensitivity; no optimizer step."""
import argparse,json
from pathlib import Path
import torch,lpips
from torch.utils.data import DataLoader
from .sma_eval import ROOT,M4_SHA
from .qwen_backend import QwenSharedBackend
from .m1b_train import load_initial
from .sma_conditioned import ConditionedSMA,install,load_joint,forward,calibrate
from .sma_data import load_manifest
from .m2b1_q20 import M2ValidationDataset
from .m2a_data_baseline import validate
from .stage1_train import image_tensor
from .qwen_layer_probe import deterministic_encode
from .stage2_train import transmission_loss
from .m4_cache import sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    out=a.run_dir/'reload_audit';out.mkdir()
    device=torch.device('cuda:0');torch.cuda.set_device(device);torch.manual_seed(2026)
    backend=QwenSharedBackend.from_local(device)
    m4=ROOT/'RMagNet/runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors'
    backend.set_trainable_branch('transmission');load_initial(backend,m4,device);backend.set_trainable_branch(None)
    sma=ConditionedSMA().to(device)
    load_joint(a.run_dir/'latest_sma.safetensors',backend,sma,device,M4_SHA)
    sma.requires_grad_(False).eval();runtime=install(backend,sma)
    data=ROOT/'datasets/rmagnet_sma_dataset2';_,records,splits=load_manifest(data)
    model=lpips.LPIPS(net='squeeze',verbose=False).cpu().requires_grad_(False)
    report=validate(backend,DataLoader(M2ValidationDataset(data,records,splits['validation']),batch_size=1),device,model,out,0,2026)
    original=a.run_dir/'latest_validation/predictions';loaded=out/'validation/step_000000/predictions'
    hashes={f.name:sha256(f) for f in loaded.glob('*.png')}
    expected={f.name:sha256(f) for f in original.glob('*.png')}
    if hashes!=expected or len(hashes)!=26:raise RuntimeError('C checkpoint PNG reload mismatch')
    identity=json.loads((a.run_dir/'initial_prediction_hashes.json').read_text())
    m4identity=json.loads((ROOT/'RMagNet/runs/sma_joint_b_e4/initial_prediction_hashes.json').read_text())
    if identity!=m4identity:raise RuntimeError('C zero-init differs from M4')
    sample=splits['train'][0]
    image=image_tensor(data/'blended'/f'{sample}.png').unsqueeze(0).to(device)
    target=image_tensor(data/'transmission_layer'/f'{sample}.png').unsqueeze(0).to(device)
    with torch.no_grad():
        latent=deterministic_encode(backend,image)
        with runtime.mode('off'):t0=forward(backend,latent)
        with runtime.mode('on'):t1=forward(backend,latent)
        current_delta=float((t1-t0).abs().mean())
    saved={k:m.gamma.detach().clone() for k,m in sma.readers.items()}
    # Temporary, documented sensitivity intervention, never saved/trained.
    with torch.no_grad():
        for m in sma.readers.values():m.gamma.fill_(.2)
    for p in sma.gate_parameters():p.requires_grad_(True)
    prediction=forward(backend,latent)
    loss,_=transmission_loss(prediction,target,.2,.1)
    loss.backward();del prediction,loss
    calibration=calibrate(backend,latent,target,0,0)
    if calibration['condition_output_l1']<=0 or calibration['reliability_loss']<=0:
        raise RuntimeError('C finite condition fails to change actual image / create calibration signal')
    if calibration['reliability_gate_gradient_ratio']>.10001:raise RuntimeError('Reliability cap violated')
    if any(p.grad is not None for p in backend.transformer.parameters()) or any(p.grad is not None for p in backend.vae.parameters()):raise RuntimeError('Frozen audit teacher has gradient')
    if any(p.grad is not None for p in sma.memory.parameters()):raise RuntimeError('Memory has gradient')
    with torch.no_grad():
        for k,m in sma.readers.items():m.gamma.copy_(saved[k])
    result={'status':'pass','reload_png_hashes_match':26,'zero_init_m4_png_match':26,
            'current_smoke_condition_output_l1':current_delta,'temporary_sensitivity_raw_gamma':.2,
            'temporary_sensitivity_calibration':calibration,'validation':report['means'],
            'note':'Sensitivity gamma is temporary, no optimizer update, not included in formal initialization/checkpoint'}
    (a.run_dir/'reload_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
