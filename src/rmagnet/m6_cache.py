"""Resumable three-view M6 references with a consistent LoRA-disabled teacher."""
from __future__ import annotations
import argparse
import gc
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import safetensors.torch
import torch
import torch.nn.functional as F
from .m4_cache import atomic_json, atomic_safetensors, build_gate, sha256
from .m6_common import (CACHE, DATA, PROJECT, ROOT, VERSION, RULE, RULE_SHA,
                       MID, EARLY, LATE, STRICT_RULE_SHA, load_dataset, load_backend, source_paths,
                       audit_sources, require_space, check_lineage, teacher_identity)
from .m6_losses import build_directions
from .qwen_layer_probe import deterministic_encode
from .stage1_train import image_tensor

class EndCapture(Exception):
    pass

@torch.inference_mode()
def capture(backend, image, layers):
    values, handles = {}, []
    for block in layers:
        def hook(_module, _inputs, output, block=block):
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError(f'Invalid block output {block}')
            values[block] = output[1].detach().to('cpu', torch.bfloat16)
            if block == max(layers):
                raise EndCapture
        handles.append(backend.transformer.transformer_blocks[block - 1].register_forward_hook(hook))
    backend.transformer.disable_lora()
    try:
        latent = deterministic_encode(backend, image)
        try:
            backend.upstream.flow_step(latent, backend.transformer, backend.vae, backend.embeddings)
        except EndCapture:
            pass
        else:
            raise RuntimeError('Teacher did not stop at requested block')
    finally:
        for handle in handles:
            handle.remove()
        backend.transformer.enable_lora()
    if set(values) != set(layers):
        raise RuntimeError('Missing captured teacher features')
    return values

def valid_record(output, sid, hashes=None):
    try:
        p = output / 'records' / f'{sid}.json'
        r = json.loads(p.read_text())
        f = output / r['cache']
        return (r['rule_sha256'] == RULE_SHA and r['id'] == sid
                and (hashes is None or r['source_sha256'] == hashes)
                and f.stat().st_size == r['cache_bytes'] and sha256(f) == r['cache_sha256'])
    except (OSError, KeyError, ValueError):
        return False

def old_reference(root, sid, hashes):
    # The failed strict preflight already computed consistent M4 references.
    # Reuse only its GT/early/gate tensors; recompute I/P90 and all new masks.
    record = root / 'records' / f'{sid}.json'
    if record.is_file() and not (root / 'manifest.json').is_file():
        r = json.loads(record.read_text())
        if r.get('rule_sha256') == STRICT_RULE_SHA and r.get('source_sha256') == hashes:
            f = root / r['cache']
            if f.stat().st_size != r['cache_bytes'] or sha256(f) != r['cache_sha256']:
                raise RuntimeError(f'Strict candidate cache hash failure: {sid}')
            return safetensors.torch.load_file(str(f))
        return None
    manifest = root / 'manifest.json'
    if not manifest.is_file():
        return None
    m = json.loads(manifest.read_text())
    t = m.get('teacher', {})
    if t.get('adapter') != 'all LoRA disabled' or t.get('flow_timestep') != 499:
        return None
    if t.get('early_blocks') != list(EARLY) or t.get('mid_blocks') != list(MID) or t.get('late_blocks') != list(LATE):
        return None
    r = next((x for x in m['samples'] if x['id'] == sid), None)
    if r is None or any(r.get('source_sha256', {}).get(k) != hashes[k] for k in ('input', 'gt')):
        return None
    f = root / r['cache']
    if not f.is_file() or sha256(f) != r['cache_sha256']:
        raise RuntimeError(f'Original M4 cache hash failure: {sid}')
    return safetensors.torch.load_file(str(f))

def unsaturated(images, gh, gw):
    if RULE.get('direction_mode') == 'unfiltered':
        return torch.ones(gh * gw, dtype=torch.bool)
    fractions = []
    for image in images:
        sat = ((image[0].float() + 1) * 127.5 >= 250).any(0).float()[None, None]
        fraction = F.avg_pool2d(sat, 16, 16).reshape(-1)
        if fraction.numel() != gh * gw:
            raise RuntimeError('Saturation token grid mismatch')
        fractions.append(fraction)
    return torch.stack(fractions).amax(0) <= RULE['max_saturation_fraction']

def preview(output, sid, paths, maps, gh, gw):
    items = []
    for name in ('input', 'gt', 'p90'):
        with Image.open(paths[name]) as image:
            im = image.convert('RGB').copy()
        im.thumbnail((320, 240))
        items.append((name, im))
    for name, value in maps.items():
        a = value.reshape(gh, gw).float().clamp(0, 1).mul(255).round().byte().numpy()
        im = Image.fromarray(a).convert('RGB').resize((320, 240), Image.Resampling.NEAREST)
        items.append((name, im))
    canvas = Image.new('RGB', (320 * len(items), 280), 'white')
    draw = ImageDraw.Draw(canvas)
    for j, (label, im) in enumerate(items):
        draw.text((j * 320 + 8, 10), f'{sid} {label}', fill='black')
        canvas.paste(im, (j * 320, 35))
    folder = output / 'previews'
    folder.mkdir(parents=True, exist_ok=True)
    canvas.save(folder / f'{sid}.png')

def prepare(args):
    manifest, records, splits = load_dataset(args.data_root)
    ids = splits['train'][args.shard_index::args.num_shards]
    hashes = audit_sources(args.data_root, records, ids)
    remaining = [sid for sid in ids if not valid_record(args.output, sid, hashes[sid])]
    print(json.dumps({'shard': args.shard_index, 'remaining': len(remaining), 'total': len(ids)}), flush=True)
    if not remaining:
        return
    require_space(16)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(2026)
    backend = load_backend(device)
    backend.set_trainable_branch(None)
    backend.transformer.eval()
    backend.vae.eval()
    reused_verified = False
    for position, sid in enumerate(remaining, 1):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        paths = source_paths(args.data_root, sid)
        images = {k: image_tensor(p)[None] for k, p in paths.items()}
        w, h = records[sid]['target_size']
        gh, gw = h // 16, w // 16
        old = old_reference(args.reuse_root, sid, hashes[sid])
        if old is not None and not reused_verified:
            fresh_gt = capture(backend, images['gt'], MID)
            errors = {str(l): float((fresh_gt[l][0].float() - old[f'q{l}_gt'].float()).norm()
                                   / fresh_gt[l][0].float().norm().clamp_min(1e-12)) for l in MID}
            if max(errors.values()) > 1e-4:
                raise RuntimeError(f'Legacy/fresh teacher disagreement: {sid}: {errors}')
            print(json.dumps({'event': 'legacy_teacher_verified', 'sample': sid,
                              'relative_errors': errors}), flush=True)
            reused_verified = True
            del fresh_gt
        qi = capture(backend, images['input'], MID if old is not None else EARLY + MID + LATE)
        if old is None:
            qg = capture(backend, images['gt'], EARLY + MID + LATE)
            gate, agreement, late_statistics = build_gate(qi, qg)
            tensors = {'late_gate': gate, 'late_agreement': agreement}
            for l in EARLY:
                tensors[f'q{l}_input'], tensors[f'q{l}_gt'] = qi[l][0], qg[l][0]
            for l in MID:
                tensors[f'q{l}_gt'] = qg[l][0]
        else:
            tensors = {k: v for k, v in old.items() if k != 'token_grid_hw'}
            gate, agreement = old['late_gate'], old['late_agreement']
            qg = {l: old[f'q{l}_gt'][None] for l in MID}
            late_statistics = {'reused_teacher': 'all LoRA disabled'}
        qp = capture(backend, images['p90'], MID)
        for role, features in (('input', qi), ('gt', qg), ('p90', qp)):
            if any(q.shape != (1, gh * gw, 3072) for q in features.values()):
                raise RuntimeError(f'Feature/grid mismatch: {sid}/{role}')
        valid = unsaturated(list(images.values()), gh, gw)
        directions, weights, diagnostic = build_directions(
            {l: qi[l][0] for l in MID}, {l: qg[l][0] for l in MID},
            {l: qp[l][0] for l in MID}, gate, valid)
        for l in MID:
            tensors[f'q{l}_negative_direction'] = directions[l].to(torch.bfloat16)
            tensors[f'q{l}_negative_weight'] = weights[l].to(torch.float16)
        tensors['token_grid_hw'] = torch.tensor([gh, gw], dtype=torch.int16)
        if not all(torch.isfinite(t.float()).all() for t in tensors.values()):
            raise RuntimeError(f'Nonfinite cache: {sid}')
        f = args.output / 'samples' / f'{sid}.safetensors'
        f.parent.mkdir(parents=True, exist_ok=True)
        atomic_safetensors(f, tensors, {'sample_id': sid, 'cache_version': VERSION, 'rule_sha256': RULE_SHA})
        row = {'id': sid, 'cache': str(f.relative_to(args.output)), 'cache_bytes': f.stat().st_size,
               'cache_sha256': sha256(f), 'source_sha256': hashes[sid], 'rule_sha256': RULE_SHA,
               'image_size_wh': [w, h], 'token_grid_hw': [gh, gw],
               'reused_original_m4': old is not None, 'reuse_root': str(args.reuse_root) if old is not None else None,
               'negative': diagnostic,
               'late_statistics': late_statistics,
               'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
               'completed_at_utc': datetime.now(timezone.utc).isoformat()}
        atomic_json(args.output / 'records' / f'{sid}.json', row)
        if sid in splits['train'][:5] or (diagnostic['active'] and position <= 2):
            preview(args.output, sid, paths, {'late-G': gate,
                    'direction-confidence': torch.stack(list(weights.values())).mean(0)}, gh, gw)
        print(json.dumps({'sample': sid, 'progress': f'{position}/{len(remaining)}',
                          'shard': args.shard_index, 'negative_active': diagnostic['active'],
                          'confidence_mass': diagnostic['confidence_mass_before_skip'],
                          'peak_allocated_gib': row['peak_allocated_gib']}), flush=True)
        del old, qi, qg, qp, tensors, directions, weights, images
    del backend
    gc.collect()
    torch.cuda.empty_cache()

def finalize(args):
    dataset, records, splits = load_dataset(args.data_root)
    # A rejected scientific precondition needs no second multi-GB checksum scan.
    # Successful caches still undergo the complete file/source validation below.
    quick = []
    for sid in splits['train']:
        path = args.output / 'records' / f'{sid}.json'
        if not path.is_file():
            break
        row = json.loads(path.read_text())
        if row.get('id') != sid or row.get('rule_sha256') != RULE_SHA:
            break
        quick.append(row)
    if len(quick) == len(splits['train']) and not any(r['negative']['active'] for r in quick):
        atomic_json(args.output / 'rejected_preflight.json', {
            'cache_version': VERSION, 'rule_sha256': RULE_SHA, 'sample_count': len(quick),
            'negative_active_samples': 0, 'training_allowed': False,
            'all_sample_records_written': True, 'final_checksum_scan_complete': False,
            'reason': 'No sample passed the new directional-supervision reliability condition'})
        raise RuntimeError('No usable directional supervision; cache preflight rejected before redundant checksum scan')
    hashes = audit_sources(args.data_root, records, splits['train'])
    rows = []
    for sid in splits['train']:
        if not valid_record(args.output, sid, hashes[sid]):
            raise RuntimeError(f'Invalid/missing M6 reference: {sid}')
        row = json.loads((args.output / 'records' / f'{sid}.json').read_text())
        with safetensors.safe_open(str(args.output / row['cache']), framework='pt') as f:
            for l in MID:
                m = f.get_tensor(f'q{l}_negative_weight')
                if not torch.isfinite(m).all() or float(m.min()) < 0 or float(m.max()) > 1:
                    raise RuntimeError(f'Invalid negative mask: {sid}/{l}')
        rows.append(row)
    active = sum(row['negative']['active'] for row in rows)
    if active == 0:
        raise RuntimeError('No training sample has usable P90-I45 directional evidence; do not launch training')
    total = sum(row['cache_bytes'] for row in rows)
    result = {'complete': True, 'cache_version': VERSION, 'rule_sha256': RULE_SHA, 'formula': RULE,
              'project_git_commit': subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip(),
              'source_dataset': {'manifest_sha256': sha256(args.data_root / 'manifest.json'),
                                 'train_ids': splits['train'], 'sample_count': len(rows)},
              'teacher': {'model': 'Qwen/Qwen-Image-Edit-2509', **teacher_identity()},
              'lineage_audit': check_lineage(records, splits),
              'storage': {'total_bytes': total, 'total_gib': total / 2**30},
              'negative_active_samples': active, 'reused_original_m4_samples': sum(r['reused_original_m4'] for r in rows),
              'samples': rows, 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    atomic_json(args.output / 'manifest.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('samples', 'source_dataset')}, indent=2), flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['extract', 'finalize'])
    p.add_argument('--data-root', type=Path, default=DATA)
    p.add_argument('--output', type=Path, default=CACHE)
    p.add_argument('--reuse-root', type=Path, default=ROOT / 'RMagNet/data_cache/m4_multilayer_v1')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    args = p.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        p.error('Invalid shard configuration')
    (prepare if args.mode == 'extract' else finalize)(args)

if __name__ == '__main__':
    main()
