"""Audit completed SMA runs for an explicitly weights-only continuation."""
import json
from pathlib import Path
from safetensors import safe_open
from .m4_cache import sha256
from .sma import SMA_VERSION


def audit_parent(parent, data_root, cache_root, initial_sha, updates_per_epoch, target_epochs):
    parent = Path(parent).resolve()
    config = json.loads((parent / 'run_config.json').read_text())
    summary = json.loads((parent / 'training_summary.json').read_text())
    latest = json.loads((parent / 'latest_metrics.json').read_text())
    checkpoint = parent / 'latest_sma.safetensors'
    if summary['status'] != 'complete' or summary.get('early_stopped'):
        raise RuntimeError('Continue only a normally completed full-epoch SMA run')
    completed, step = summary['epochs_completed'], summary['optimizer_updates']
    if step != completed * updates_per_epoch or latest['step'] != step:
        raise RuntimeError('Parent latest is not the final full-epoch checkpoint')
    if latest['epoch'] + 1 != completed or not completed < target_epochs <= 20:
        raise RuntimeError('Target must exceed completed epochs and must be at most 20')
    if config['dataset_manifest_sha256'] != sha256(data_root / 'manifest.json'):
        raise RuntimeError('Continuation dataset changed')
    if config['cache_manifest_sha256'] != sha256(cache_root / 'manifest.json'):
        raise RuntimeError('Continuation cache changed')
    if config['initial_sha256'] != initial_sha or config['sma_architecture'] != SMA_VERSION:
        raise RuntimeError('Continuation fixed M4 / SMA architecture changed')
    if config['world_size'] != 4 or config['updates_per_epoch'] != updates_per_epoch:
        raise RuntimeError('Continuation batch accounting changed')
    with safe_open(checkpoint, framework='pt') as f:
        md = f.metadata()
        if md.get('architecture') != SMA_VERSION or md.get('base_m4_sha256') != initial_sha:
            raise RuntimeError('Parent SMA checkpoint identity mismatch')
    expected_eval = parent / 'test_latest/evaluation.json'
    checkpoint_sha = sha256(checkpoint)
    if expected_eval.exists() and json.loads(expected_eval.read_text())['checkpoint_sha256'] != checkpoint_sha:
        raise RuntimeError('Parent latest changed after its evaluation')
    return {'mode': 'weights-only-continuation', 'parent_run': str(parent),
            'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha,
            'completed_epochs': completed, 'initial_global_step': step,
            'new_epochs': target_epochs-completed,
            'new_optimizer_updates': (target_epochs-completed)*updates_per_epoch,
            'optimizer_restored': False, 'scheduler_restored': False,
            'gradient_controller_ema_restored': False,
            'parent_best_step': json.loads((parent/'best_metrics.json').read_text())['step'],
            'epochs_without_improvement': summary.get('epochs_without_improvement', 0)}
