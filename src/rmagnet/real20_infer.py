"""Run official WindowSeat or M4-best on RAGNet real20 with upstream tiling."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import safetensors.torch
import torch
from .qwen_backend import QwenSharedBackend

ROOT = Path('/share/linmingheng-local/xuke')
DATA = ROOT / 'datasets/liyucs_RAGNet/testsets/real20'
M4 = ROOT / 'RMagNet/runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors'
M0 = ROOT / 'RMagNet/runs/M0_windowseat_m2_e18/best_transmission_lora.safetensors'

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=('windowseat', 'm4best', 'm0'), required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--data-root', type=Path, default=DATA)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args()
    inputs = sorted((a.data_root / 'blended').glob('*.jpg'))
    gts = sorted((a.data_root / 'transmission_layer').glob('*.jpg'))
    if len(inputs) != 20 or {x.name for x in inputs} != {x.name for x in gts}:
        raise RuntimeError(f'real20 pair check failed: {len(inputs)} inputs, {len(gts)} GTs')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed(a.seed)
    backend = QwenSharedBackend.from_local(device)
    weight = {'path': 'official WindowSeat adapter', 'sha256': None}
    selected = M4 if a.variant == 'm4best' else M0 if a.variant == 'm0' else None
    if selected is not None:
        state = safetensors.torch.load_file(str(selected), device=str(device))
        _, unexpected = backend.transformer.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f'unexpected M4 keys: {unexpected[:5]}')
        weight = {'path': str(selected), 'sha256': sha256(selected)}
    backend.set_trainable_branch(None)
    backend.transformer.eval(); backend.vae.eval()
    backend.upstream.run_inference(
        backend.vae, backend.transformer, backend.embeddings, backend.resolution,
        str(a.data_root / 'blended'), str(a.output_dir),
        use_short_edge_tile=True, save_comparison=False, save_alternating=False,
        batch_size=1, num_workers=0,
    )
    outputs = sorted(a.output_dir.glob('*_windowseat_output.png'))
    if len(outputs) != 20:
        raise RuntimeError(f'expected 20 outputs, found {len(outputs)}')
    repo = ROOT / 'datasets/liyucs_RAGNet'
    manifest = {
        'status': 'complete', 'variant': a.variant, 'seed': a.seed,
        'dataset': str(a.data_root),
        'dataset_commit': subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip(),
        'sample_count': len(outputs), 'tiling': 'official WindowSeat short-edge tiles',
        'weight': weight,
    }
    (a.output_dir / 'inference_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)
if __name__ == '__main__':
    main()
