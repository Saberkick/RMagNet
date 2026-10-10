"""M6 identity, dataset validation and serialized model loading."""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import torch
from .m4_cache import atomic_json, sha256
from .qwen_backend import QwenSharedBackend

ROOT = Path('/share/linmingheng-local/xuke')
PROJECT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'datasets/rmagnet_sma_dataset3'
DIRECTION_MODE = os.environ.get('M6_DIRECTION_MODE', 'unfiltered')
if DIRECTION_MODE not in ('strict', 'gt-calibrated', 'unfiltered'):
    raise ValueError('Invalid M6_DIRECTION_MODE')
CACHE = PROJECT / ('data_cache/m6_polar_negative_v1' if DIRECTION_MODE == 'strict'
                   else 'data_cache/m6_gt_calibrated_v2' if DIRECTION_MODE == 'gt-calibrated'
                   else 'data_cache/m6_unfiltered_v3')
INITIAL = ROOT / 'RMagNet/runs/m4_e30_p4/best_transmission_lora.safetensors'
INITIAL_SHA = '5725d32b04e1271d51a33f7512174f1035ff0acf5e5427bdd3b179e98e1a13eb'
DATA_SHA = 'bb211d1de9d399c685d70e80ef292d9a86b9e6733fb78cf1c1724d0d591bb77c'
VERSION = 'm6-polar-negative-v1'
MID = (37, 39, 41)
EARLY = (16, 20)
LATE = (52, 54, 56)
RULE = {'version': VERSION, 'teacher': 'all LoRA disabled', 'timestep': 499,
        'qwen_revision': 'd3968ef930e841f4c73640fb8afa3b306a78167e',
        'windowseat_revision': 'c1f59ca02bff68535c976e5e17147b3d9323309e',
        'mid': MID, 'min_norm': 0.02, 'align_low': 0.2, 'align_width': 0.6,
        'min_layers': 2, 'min_mass': 0.01, 'saturation_u8': 250,
        'max_saturation_fraction': 0.1}
RULE_SHA = hashlib.sha256(json.dumps(RULE, sort_keys=True).encode()).hexdigest()
STRICT_RULE_SHA = RULE_SHA
if DIRECTION_MODE == 'gt-calibrated':
    VERSION = 'm6-gt-calibrated-v2'
    RULE.update(version=VERSION, direction_mode=DIRECTION_MODE,
                reliability='absolute GT alignment', polarity='sign of GT alignment')
    RULE_SHA = hashlib.sha256(json.dumps(RULE, sort_keys=True).encode()).hexdigest()
elif DIRECTION_MODE == 'unfiltered':
    VERSION = 'm6-unfiltered-v3'
    RULE = {k: v for k, v in RULE.items() if k not in (
        'min_norm', 'align_low', 'align_width', 'min_layers', 'min_mass',
        'saturation_u8', 'max_saturation_fraction')}
    RULE.update(version=VERSION, direction_mode=DIRECTION_MODE,
                reliability='none; finite nonzero vectors only',
                polarity='raw P90-I45; no flipping', loss='GT-anchored squared projection, both signs')
    RULE_SHA = hashlib.sha256(json.dumps(RULE, sort_keys=True).encode()).hexdigest()

def load_dataset(root: Path):
    root = root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if not manifest.get('complete') or sha256(root / 'manifest.json') != DATA_SHA:
        raise RuntimeError('M6 dataset identity differs from the authorized manifest')
    records = {s['id']: s for s in manifest['samples']}
    splits = {k: (root / 'splits' / f'{k}.txt').read_text().split()
              for k in ('train', 'validation', 'test')}
    if {k: len(v) for k, v in splits.items()} != {'train': 204, 'validation': 25, 'test': 24}:
        raise RuntimeError('Unexpected M6 split sizes')
    if sum(map(len, splits.values())) != len(set(sum(splits.values(), []))):
        raise RuntimeError('Split overlap or duplicate IDs')
    if set(sum(splits.values(), [])) != set(records):
        raise RuntimeError('Split union differs from manifest')
    groups = [{records[i]['group'] for i in splits[k]} for k in splits]
    if any(groups[a] & groups[b] for a in range(3) for b in range(a + 1, 3)):
        raise RuntimeError('Capture group leakage')
    for split, ids in splits.items():
        for sid in ids:
            r = records[sid]
            if r['split'] != split or any(int(d) % 16 for d in r['target_size']):
                raise RuntimeError(f'Invalid split or grid: {sid}')
    return manifest, records, splits

def source_paths(root: Path, sid: str):
    return {'input': root / 'blended' / f'{sid}.png',
            'gt': root / 'transmission_layer' / f'{sid}.png',
            'p90': root / 'reflection_90' / f'{sid}.png'}

def source_hashes(root: Path, sid: str):
    return {k: sha256(p) for k, p in source_paths(root, sid).items()}

def manifest_processed_sha(record: dict, role: str):
    aliases = {'input': ('input', 'blended'), 'gt': ('gt', 'transmission_layer'),
               'p90': ('reflection_90', 'p90')}
    for key in aliases[role]:
        v = record.get('processed', {}).get(key)
        if isinstance(v, dict) and 'sha256' in v:
            return v['sha256']
    return None

def audit_sources(root: Path, records: dict, ids: list[str]):
    from PIL import Image
    from collections import Counter
    hashes = {}
    for sid in ids:
        row = {}
        for role, p in source_paths(root, sid).items():
            with Image.open(p) as im:
                if list(im.size) != list(records[sid]['target_size']) or im.mode != 'RGB':
                    raise RuntimeError(f'RGB shape/mode mismatch: {sid}/{role}')
            row[role] = sha256(p)
            expected = manifest_processed_sha(records[sid], role)
            if expected is None or row[role] != expected:
                raise RuntimeError(f'Processed role/hash mismatch: {sid}/{role}')
        hashes[sid] = row
    return hashes

@contextlib.contextmanager
def model_load_lock():
    """Serialize transient CPU loading peaks; inference/training remain independent."""
    lock = ROOT / 'tmp/m6_model_load.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)

def load_backend(device: torch.device):
    with model_load_lock():
        free, total = torch.cuda.mem_get_info(device)
        info = {'event': 'model_load_binding', 'pid': os.getpid(),
                'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'local_device': str(device),
                'uuid': str(torch.cuda.get_device_properties(device).uuid),
                'cuda_free_gib': free / 2**30}
        print(json.dumps(info), flush=True)
        if free < 20 * 2**30:
            raise RuntimeError('GPU became busy before model loading; leave its other process running')
        backend = QwenSharedBackend.from_local(device)
    return backend

def require_space(minimum_gib: float):
    import shutil
    available = shutil.disk_usage(ROOT).free / 2**30
    if available < minimum_gib:
        raise RuntimeError(f'Free space {available:.2f} GiB < {minimum_gib:.2f} GiB')
    return available

def teacher_identity():
    from .qwen_backend import check_snapshots, REPO
    base, lora = check_snapshots()
    if base.name != RULE['qwen_revision'] or lora.name != RULE['windowseat_revision']:
        raise RuntimeError('M6 teacher revision differs from locked design')
    return {'qwen_revision': base.name, 'windowseat_revision': lora.name,
            'text_embeddings_sha256': sha256(lora / 'text_embeddings/state_dict.safetensors'),
            'upstream_code_sha256': sha256(REPO / 'windowseat_inference.py'),
            'vae': 'deterministic posterior mode', 'adapter': 'all LoRA disabled',
            'flow_timestep': 499, 'mid_blocks': list(MID)}

def check_lineage(records: dict, splits: dict):
    old = ROOT / 'RMagNet/data_cache/m4_multilayer_v1/manifest.json'
    if not old.is_file():
        old = PROJECT / 'docx/M6branch/M6_HISTORICAL_M4_CACHE_MANIFEST.json'
    m = json.loads(old.read_text())
    old_ids = {s['id'] for s in m['samples']} | set(m.get('local_filter', {}).get('removed_ids', []))
    old_groups = {s.split('_')[0] for s in old_ids}
    held_groups = {records[s]['group'] for k in ('validation', 'test') for s in splits[k]}
    if old_groups & held_groups:
        raise RuntimeError('Historical M4 training capture groups overlap evaluation')
    # Stage 2 filenames are recoverable from its initial data directory/config if present.
    cfg = json.loads((ROOT / 'RMagNet/runs/stage2_transmission_r128/run_config.json').read_text())
    data_value = cfg.get('args', {}).get('data_root') or cfg.get('args', {}).get('data')
    audit = {'m4_training_ids': len(old_ids), 'm4_eval_group_overlap': [],
             'stage2': 'historical manifest audit unavailable; inherited evaluation limitation'}
    if cfg.get('train_ids'):
        stage2_groups = {str(s).split('_')[0] for s in cfg['train_ids']}
        audit['stage2'] = {'source': 'Stage 2 run_config.json train_ids',
                          'matching_validation_groups': sorted(stage2_groups & {records[s]['group'] for s in splits['validation']}),
                          'matching_test_groups': sorted(stage2_groups & {records[s]['group'] for s in splits['test']}),
                          'interpretation': 'inherited potential exposure; old source manifest unavailable; retrospective validation/test only'}
    elif data_value:
        p = Path(data_value)
        training = p / 'splits/train.txt'
        if training.is_file():
            ids = training.read_text().split()
            overlap = {s.split('_')[0] for s in ids} & held_groups
            if overlap:
                raise RuntimeError(f'Stage 2 group overlap: {sorted(overlap)}')
            audit['stage2'] = {'path': str(training), 'group_overlap': []}
    return audit
