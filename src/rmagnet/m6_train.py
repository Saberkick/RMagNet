"""M6-B: five-epoch M4-initial continuation, strict auxiliary gradient budgets."""
from __future__ import annotations
import argparse
import gc
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import bitsandbytes as bnb
import lpips
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .c1_l20_train import move_optimizer_state
from .m1b_train import load_initial
from .m2a_data_baseline import (AspectGroupedDistributedSampler, atomic_save_adapter,
    check_initial_hash, grouped_global_batches, maybe_save_best, validate)
from .m2b1_q20 import M2ValidationDataset
from .m3_with_lrec_100e import forward_from_latent, consistency_loss
from .m4_train import (M4Dataset, token_weights, texture_loss, semantic_loss,
                       spatial_loss, online_prediction_features)
from .m4_cache import atomic_json, sha256
from .m6_common import (CACHE, DATA, PROJECT, INITIAL, INITIAL_SHA, VERSION, RULE_SHA,
    MID, load_dataset, load_backend, audit_sources, require_space, check_lineage, teacher_identity)
from .m6_losses import M6GradientController, negative_loss
from .qwen_backend import ADAPTER_NAMES
from .qwen_layer_probe import deterministic_encode
from .stage1_train import (append_jsonl, rank, world_size, seed_everything,
    setup_distributed, sync_initial_parameters, sync_gradients, trainable_parameters, make_scheduler)
from .stage2_train import transmission_loss

class M6Dataset(M4Dataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        row = self.cache_records[item['id']]
        with safetensors.safe_open(str(self.cache_root / row['cache']), framework='pt') as f:
            for l in MID:
                for role in ('negative_direction', 'negative_weight'):
                    key = f'q{l}_{role}'
                    item[key] = f.get_tensor(key)
                    if not torch.isfinite(item[key].float()).all():
                        raise RuntimeError(f'Nonfinite negative reference {item["id"]}/{key}')
        return item

def load_references(cache, data, train_ids):
    m = json.loads((cache / 'manifest.json').read_text())
    if not m.get('complete') or m.get('cache_version') != VERSION or m.get('rule_sha256') != RULE_SHA:
        raise RuntimeError('M6 cache identity incomplete or incompatible')
    if m['source_dataset']['train_ids'] != train_ids or m['source_dataset']['manifest_sha256'] != sha256(data / 'manifest.json'):
        raise RuntimeError('M6 cache data/split mismatch')
    if m['teacher']['adapter'] != 'all LoRA disabled' or any(m['teacher'].get(k) != v for k, v in teacher_identity().items()):
        raise RuntimeError('Wrong M6 teacher adapter')
    rows = {s['id']: s for s in m['samples']}
    if set(rows) != set(train_ids):
        raise RuntimeError('M6 reference coverage differs from train')
    for sid, row in rows.items():
        path = cache / row['cache']
        if sha256(path) != row['cache_sha256']:
            raise RuntimeError(f'M6 reference hash mismatch {sid}')
    return m, rows

def barrier():
    if dist.is_initialized():
        dist.barrier()

def initialize_distributed():
    local = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        # Model reads may be slow on this shared filesystem; serialize CPU load
        # without making the earliest rank time out at its first broadcast.
        dist.init_process_group(backend='nccl', device_id=device, timeout=timedelta(hours=2))
    return device

def finish_validation(backend, loader, device, lpips_model, run_dir, step, epoch, args):
    report = validate(backend, loader, device, lpips_model, run_dir, step, args.seed)
    updated = maybe_save_best(run_dir, report, step, backend, args.initial)
    directory = run_dir / 'validation' / f'step_{step:06d}'
    if updated:
        dest = run_dir / 'validation/best'
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(directory, dest)
    # Validation CSV/JSON for every epoch are tiny and preserve the complete curve.
    history = run_dir / 'validation/history'
    history.mkdir(exist_ok=True)
    for name in ('metrics.json', 'metrics.csv'):
        shutil.copy2(directory / name, history / f'epoch_{epoch:02d}_{name}')
    latest_images = run_dir / 'validation/latest'
    if latest_images.exists():
        shutil.rmtree(latest_images)
    directory.rename(latest_images)
    if step > 0:
        require_space(8)
        atomic_save_adapter(run_dir / 'latest_transmission_lora.safetensors', backend)
    atomic_json(run_dir / 'latest_metrics.json', {'step': step, 'epoch': epoch, 'means': report['means']})
    event = {'kind': 'validation', 'step': step, 'epoch': epoch, 'means': report['means'], 'best_updated': updated}
    append_jsonl(run_dir / 'metrics.jsonl', event)
    print(json.dumps(event), flush=True)
    return report

def one_update(backend, batch, device, parameters, optimizer, scheduler, controller, step):
    image, p90, target = [batch[k].to(device) for k in ('image', 'p90', 'target')]
    gh, gw = (int(v) for v in batch['token_grid'][0])
    gate, keep, restore = token_weights(batch, device)
    confidence = 0.5 + 0.5 * batch['late_agreement'].to(device).float()
    with torch.no_grad():
        latent_i = deterministic_encode(backend, image)
        latent_90 = deterministic_encode(backend, p90)
        probe_i = forward_from_latent(backend, latent_i).detach()
        probe_p = forward_from_latent(backend, latent_90).detach()
    leaf = probe_i.detach().requires_grad_(True)
    rec, _ = transmission_loss(leaf, target, 0.2, 0.1)
    pixel_gate = F.interpolate(gate.reshape(1, 1, gh, gw), size=leaf.shape[-2:], mode='bilinear', align_corners=False)
    cons = consistency_loss(leaf, probe_p, pixel_gate)
    base = 0.5 * (rec + 0.10 * cons)
    spatial, restore_loss, keep_loss = spatial_loss(leaf, target, image, gate, gh, gw)
    backend.transformer.disable_lora()
    try:
        features = online_prediction_features(backend, leaf, gh * gw)
        texture = texture_loss(features, batch, device, keep, restore)
        positive, content, relation = semantic_loss(features, batch, device, confidence, gh, gw)
        negative, negative_logs = negative_loss(features, batch, device)
        losses = {'base': base, 'spatial': spatial, 'texture': texture,
                  'semantic': positive, 'negative': negative}
        if not all(bool(torch.isfinite(v).all()) for v in losses.values()):
            raise RuntimeError('Nonfinite M6 loss')
        gradients = {k: torch.autograd.grad(v, leaf, retain_graph=True)[0]
                     for k, v in losses.items() if k != 'negative'}
        gradients['negative'] = torch.autograd.grad(negative, leaf)[0]
    finally:
        backend.transformer.enable_lora()
        backend.transformer.set_adapter(ADAPTER_NAMES['transmission'])
    gi, controller_logs = controller.combine(gradients['base'], {k: v for k, v in gradients.items() if k != 'base'}, step)
    logs = {k: float(v.detach()) for k, v in losses.items()}
    logs.update({'rec_i': float(rec.detach()), 'consistency_i': float(cons.detach()),
                 'spatial_restore': float(restore_loss.detach()), 'spatial_keep': float(keep_loss.detach()),
                 'semantic_content': float(content.detach()), 'semantic_relation': float(relation.detach()),
                 **negative_logs, **controller_logs})
    del losses, features, gradients, leaf, base, rec, cons, spatial, texture, positive, negative, content, relation
    del restore_loss, keep_loss
    torch.cuda.empty_cache()
    prediction = forward_from_latent(backend, latent_i)
    if not torch.isfinite(gi).all() or float(gi.float().norm()) <= 0:
        raise RuntimeError('Invalid M6 output gradient')
    prediction.backward(gi)
    pi_ref = prediction.detach()
    del prediction, gi
    torch.cuda.empty_cache()
    prediction_p = forward_from_latent(backend, latent_90)
    rec_p, _ = transmission_loss(prediction_p, target, 0.2, 0.1)
    cons_p = consistency_loss(prediction_p, pi_ref, pixel_gate)
    bp = 0.5 * (rec_p + 0.10 * cons_p)
    if not torch.isfinite(bp):
        raise RuntimeError('Invalid P90 base loss')
    gp = torch.autograd.grad(bp, prediction_p)[0]
    prediction_p.backward(gp)
    logs.update(rec_p90=float(rec_p.detach()), consistency_p90=float(cons_p.detach()))
    if any(p.grad is not None for name, p in backend.transformer.named_parameters()
           if not ('.lora_' in name and '.default.' in name)) or any(p.grad is not None for p in backend.vae.parameters()):
        raise RuntimeError('Frozen backbone/VAE/reflection adapter received gradients')
    logs['active_gradient_tensors'] = sync_gradients(parameters, device)
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
        raise RuntimeError('Invalid LoRA_T parameter gradient')
    before = parameters[0].detach().flatten()[:256].clone() if step <= 5 else None
    move_optimizer_state(optimizer, device)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    move_optimizer_state(optimizer, torch.device('cpu'))
    if before is not None:
        logs['first_parameter_delta_max'] = float((parameters[0].detach().flatten()[:256] - before).abs().max())
    logs['grad_norm'] = float(grad_norm)
    return logs

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=DATA)
    p.add_argument('--cache-root', type=Path, default=CACHE)
    p.add_argument('--run-dir', type=Path, default=PROJECT / 'runs/m6_b_e5')
    p.add_argument('--initial', type=Path, default=INITIAL)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--num-workers', type=int, default=1)
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--skip-initial-validation', action='store_true')
    args = p.parse_args()
    if args.epochs < 1 or args.max_steps < 0:
        p.error('Invalid epoch/step budget')
    m, records, splits = load_dataset(args.data_root)
    cache_m, cache_records = load_references(args.cache_root, args.data_root, splits['train'])
    audit_sources(args.data_root, records, splits['train'])
    dataset = M6Dataset(args.data_root, args.cache_root, records, cache_records, splits['train'], False)
    replicas = int(os.environ.get('WORLD_SIZE', '1'))
    steps_per_epoch = len(grouped_global_batches(dataset, args.seed, 0, replicas))
    planned = steps_per_epoch * args.epochs
    if args.max_steps:
        planned = min(planned, args.max_steps)
    preflight = {'experiment': 'M6-B', 'train': len(dataset), 'validation': len(splits['validation']),
                 'test': len(splits['test']), 'world_size': replicas, 'effective_batch': replicas,
                 'updates_per_epoch': steps_per_epoch, 'planned_updates': planned,
                 'negative_active_samples': cache_m['negative_active_samples'],
                 'lineage': check_lineage(records, splits), 'cache_gib': cache_m['storage']['total_gib']}
    if args.preflight_only:
        print(json.dumps(preflight, indent=2)); return
    require_space(16)
    device = initialize_distributed()
    main_rank = rank() == 0
    if main_rank:
        if args.run_dir.exists() and any(args.run_dir.iterdir()):
            raise FileExistsError(f'Nonempty M6 run: {args.run_dir}')
        args.run_dir.mkdir(parents=True, exist_ok=True)
    barrier()
    sampler = AspectGroupedDistributedSampler(dataset, args.seed)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    val_data = M2ValidationDataset(args.data_root, records, splits['validation'])
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
    seed_everything(args.seed)
    initial_sha = check_initial_hash(args.initial, INITIAL_SHA, device)
    backend = load_backend(device)
    backend.transformer.enable_gradient_checkpointing()
    backend.set_trainable_branch('transmission')
    if main_rank:
        load_initial(backend, args.initial, device)
    parameters = trainable_parameters(backend)
    sync_initial_parameters(parameters)
    if any('.lora_' not in n or '.default.' not in n for n, v in backend.transformer.named_parameters() if v.requires_grad):
        raise RuntimeError('Only transmission LoRA may be trainable')
    backend.transformer.train(); backend.vae.eval()
    seed_everything(args.seed + rank())
    optimizer = bnb.optim.PagedAdamW8bit(parameters, lr=5e-6, weight_decay=0.01)
    scheduler = make_scheduler(optimizer, min(20, planned), planned)
    controller = M6GradientController()
    lpips_model = None
    if main_rank:
        lpips_model = lpips.LPIPS(net='squeeze', verbose=False).eval().cpu()
        for value in lpips_model.parameters():
            value.requires_grad_(False)
        atomic_json(args.run_dir / 'run_config.json', {
            **preflight, 'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'initial_sha256': initial_sha, 'dataset_manifest_sha256': sha256(args.data_root / 'manifest.json'),
            'cache_manifest_sha256': sha256(args.cache_root / 'manifest.json'),
            'git_commit': subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip(),
            'gpu_binding': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'learning_rate': 5e-6, 'semantic_targets': {'positive': 0.06, 'negative': 0.02},
            'semantic_cap': 0.08, 'auxiliary_cap': 0.25,
            'checkpoint_policy': 'best/latest adapters only; no optimizer state', 'early_stopping': False})
    barrier()
    if main_rank and not args.skip_initial_validation:
        finish_validation(backend, val_loader, device, lpips_model, args.run_dir, 0, 0, args)
    barrier()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    start = time.monotonic()
    stop = False
    final = None
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            result = one_update(backend, batch, device, parameters, optimizer, scheduler, controller, step + 1)
            step += 1
            keys = sorted(result)
            values = torch.tensor([result[k] for k in keys], device=device, dtype=torch.float64)
            if dist.is_initialized():
                dist.all_reduce(values); values.div_(world_size())
            if main_rank:
                event = {'kind': 'train', 'epoch': epoch + 1, 'step': step,
                         **{k: float(v) for k, v in zip(keys, values)},
                         'lr': scheduler.get_last_lr()[0], 'elapsed_seconds': time.monotonic() - start,
                         'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
                         'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30}
                append_jsonl(args.run_dir / 'metrics.jsonl', event)
                atomic_json(args.run_dir / 'status.json', {'phase': 'training', 'epoch': epoch + 1,
                    'epochs': args.epochs, 'step': step, 'planned_updates': planned,
                    'updated_at_utc': datetime.now(timezone.utc).isoformat()})
                print(json.dumps(event), flush=True)
            del batch, result, values
            gc.collect(); torch.cuda.empty_cache()
            if step >= planned:
                stop = True; break
        barrier()
        if main_rank:
            final = finish_validation(backend, val_loader, device, lpips_model, args.run_dir, step, epoch + 1, args)
        barrier()
        if stop:
            break
    if main_rank:
        atomic_json(args.run_dir / 'training_summary.json', {'status': 'complete', 'epochs_requested': args.epochs,
            'epochs_completed': epoch + 1, 'optimizer_updates': step, 'validation': final['means'],
            'wall_seconds_training_and_validation': time.monotonic() - start,
            'retention': 'best/latest LoRA only', 'early_stopped': False})
        atomic_json(args.run_dir / 'status.json', {'phase': 'complete', 'epoch': epoch + 1, 'step': step})
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
