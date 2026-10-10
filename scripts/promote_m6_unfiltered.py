"""Reuse verified raw M6 references; remove directional screening without GPU work."""
import json
import shutil
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from src.rmagnet.m4_cache import atomic_json, sha256
from src.rmagnet.m6_common import (CACHE, PROJECT, DATA, VERSION, RULE, RULE_SHA,
    STRICT_RULE_SHA, DIRECTION_MODE, load_dataset, audit_sources,
    teacher_identity, check_lineage)

if DIRECTION_MODE != 'unfiltered':
    raise RuntimeError('This converter only implements the user-authorized unfiltered recipe')
source = PROJECT / 'data_cache/m6_polar_negative_v1'
target = CACHE
allowed = (PROJECT / 'data_cache').resolve()
for p in (source, target):
    if p.is_symlink() or p.resolve().parent != allowed:
        raise RuntimeError('Unexpected cache conversion path')
if (target / 'manifest.json').is_file():
    manifest = json.loads((target / 'manifest.json').read_text())
    if manifest.get('complete') and manifest.get('rule_sha256') == RULE_SHA:
        print(json.dumps({'status': 'already_promoted', 'cache': str(target)}), flush=True)
        raise SystemExit(0)
_, records, splits = load_dataset(DATA)
base = target if target.exists() else source
if not base.is_dir():
    raise FileNotFoundError(base)
hashes = audit_sources(DATA, records, splits['train'])
rows = []
for sid in splits['train']:
    row = json.loads((base / 'records' / f'{sid}.json').read_text())
    if row['id'] != sid or row['rule_sha256'] not in (STRICT_RULE_SHA, RULE_SHA):
        raise RuntimeError(f'Wrong raw reference identity: {sid}')
    if row['source_sha256'] != hashes[sid]:
        raise RuntimeError(f'Changed reference source: {sid}')
    if (base / row['cache']).stat().st_size != row['cache_bytes']:
        raise RuntimeError(f'Incomplete reference file: {sid}')
    diagnostic = row.get('strict_negative_diagnostic', row['negative'])
    active = any(diagnostic['layers'][str(l)]['norm_i_p90_mean'] > 1e-12 for l in (37,39,41))
    row.update(rule_sha256=RULE_SHA, source_feature_rule_sha256=STRICT_RULE_SHA,
               strict_negative_diagnostic=diagnostic,
               negative={'active': active, 'direction_screening': False,
                         'token_weight': 'runtime ones for finite nonzero raw directions'},
               weight_materialization='legacy stored masks ignored; runtime weights are defined by v3 rule')
    rows.append(row)
if base == source:
    source.rename(target)
for row in rows:
    atomic_json(target / 'records' / f'{row["id"]}.json', row)
# Delete only obsolete strict-preview/failed-gate files in the converted cache.
if (target / 'previews').is_dir():
    shutil.rmtree(target / 'previews')
for name in ('rejected_preflight.json',):
    (target / name).unlink(missing_ok=True)
manifest = {'complete': True, 'cache_version': VERSION, 'rule_sha256': RULE_SHA,
    'formula': RULE, 'teacher': {'model': 'Qwen/Qwen-Image-Edit-2509', **teacher_identity()},
    'source_dataset': {'manifest_sha256': sha256(DATA / 'manifest.json'),
        'train_ids': splits['train'], 'sample_count': len(rows)},
    'negative_active_samples': sum(r['negative']['active'] for r in rows),
    'storage': {'total_bytes': sum(r['cache_bytes'] for r in rows),
        'total_gib': sum(r['cache_bytes'] for r in rows) / 2**30},
    'samples': rows, 'lineage_audit': check_lineage(records, splits),
    'project_git_commit': subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip(),
    'raw_feature_identity': 'Original v1 bytes and file SHA retained; no teacher/model/data changes',
    'integrity_policy': 'Creation-time hashes retained; full raw-file checksum verification required by training preflight',
    'weight_policy': 'Legacy stored masks ignored; masks computed from finite nonzero directions at load time',
    'completed_at_utc': datetime.now(timezone.utc).isoformat()}
atomic_json(target / 'manifest.json', manifest)
print(json.dumps({'status': 'promoted', 'samples': len(rows),
    'active_samples_from_saved_norms': manifest['negative_active_samples'],
    'cache_gib': manifest['storage']['total_gib'], 'path': str(target)}), flush=True)
