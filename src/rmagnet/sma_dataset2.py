"""Prepare a grouped, aspect-preserving clean SMA dataset with explicit labels."""
import argparse
import copy
import hashlib
import json
import os
import random
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import m2_prepare_data as prep

VERSION = 'sma-variable-aspect-v4-expanded-clean'


def assign_groups(samples, seed, inherited):
    groups = defaultdict(list)
    for r in samples:
        groups[r['group']].append(r)
    # Old groups keep their split. New groups use the existing aspect-balanced
    # rule, then move the fewest complete groups to satisfy four-rank accounting.
    free = [r for r in samples if r['group'] not in inherited]
    assignment = dict(inherited)
    if free:
        assignment.update(prep.deterministic_split(free, seed))
    counts = Counter(assignment[r['group']] for r in samples)
    if counts['train'] % 4:
        options = []
        keys = [g for g in groups if g not in inherited]
        import itertools
        for number in range(1, 4):
            for selected in itertools.combinations(keys, number):
                for targets in itertools.product(('train', 'validation', 'test'), repeat=number):
                    trial = dict(assignment)
                    trial.update(zip(selected, targets))
                    c = Counter(trial[r['group']] for r in samples)
                    if c['train'] % 4 or min(c.values()) == 0 or len(c) != 3:
                        continue
                    moved = sum(len(groups[g]) for g in selected if assignment[g] != trial[g])
                    score = (moved, abs(c['train']/len(samples)-.8), abs(c['validation']-c['test']), selected, targets)
                    options.append((score, trial))
            if options:
                break
        if not options:
            raise RuntimeError('Cannot obtain complete four-rank epochs without splitting a capture group')
        assignment = min(options, key=lambda x:x[0])[1]
    return assignment


def prepare(a):
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    # The new folder can have corrected names even when the old archive did not.
    original_role = prep.role_from_suffix
    if a.labels == 'gt-suffix':
        def role(suffix):
            return 'input' if not suffix else {'_gt':'gt','_90':'reflection_90','_dolp':'dolp'}[suffix.lower()]
        prep.role_from_suffix = role
    else:
        prep.role_from_suffix = original_role
    fresh, stats = prep.scan_archive(a.archive, 512*384, 16)
    old = None
    inherited = {}
    samples = []
    if a.base:
        old = json.loads((a.base/'manifest.json').read_text())
        if old['version'] != 'm2-variable-aspect-v3-excluded-misaligned' or old.get('label_noise'):
            raise RuntimeError('Merge requires the original clean v3 dataset')
        samples = copy.deepcopy(old['samples'])
        for r in samples:
            inherited[r['group']] = r['split']
            r['dataset_source'] = 'original-clean-v3'
        excluded_ids = {r['id'] for r in old.get('excluded_samples', [])}
        if excluded_ids & {r['id'] for r in fresh}:
            raise RuntimeError('Excluded scene supplied again')
    prior_ids = {r['id'] for r in samples}
    if prior_ids & {r['id'] for r in fresh}:
        raise RuntimeError('Incoming IDs overlap the original dataset')
    # Byte-identical original images in another group would leak supervision.
    prior_hashes = {v['sha256']:r['group'] for r in samples for v in r['source'].values()}
    for r in fresh:
        r['dataset_source'] = 'data_set2'
        for v in r['source'].values():
            if v['sha256'] in prior_hashes and prior_hashes[v['sha256']] != r['group']:
                raise RuntimeError(f'Duplicate source across capture groups: {r["id"]}')
    samples += fresh
    assignment = assign_groups(samples, a.seed, inherited)
    for r in samples:
        r['split'] = assignment[r['group']]
    counts = Counter(r['split'] for r in samples)
    if min(counts.values()) == 0 or len(counts) != 3 or counts['train'] % 4:
        raise RuntimeError('Invalid split sizes')
    output.mkdir(parents=True)
    for folder in (*prep.OUTPUT_FOLDERS.values(), 'splits'):
        (output/folder).mkdir()
    if old:
        for r in samples[:len(old['samples'])]:
            for folder in prep.OUTPUT_FOLDERS.values():
                os.link(a.base/folder/f'{r["id"]}.png', output/folder/f'{r["id"]}.png')
    with zipfile.ZipFile(a.archive) as z:
        for r in fresh:
            r['processed'] = {}
            for role, folder in prep.OUTPUT_FOLDERS.items():
                im = prep.open_member(z, z.getinfo(r['source'][role]['archive_member']))
                im = prep.process_image(im, role, tuple(r['target_size']))
                path = output/folder/f'{r["id"]}.png'
                im.save(path, compress_level=6)
                r['processed'][role] = {'path':str(path.relative_to(output)), 'sha256':prep.sha256_path(path),
                    'size':list(im.size), 'mode':im.mode, 'bytes':path.stat().st_size}
    for split in ('train','validation','test'):
        (output/'splits'/f'{split}.txt').write_text('\n'.join(sorted(r['id'] for r in samples if r['split']==split))+'\n')
    # Duplicate processed targets must not cross split boundaries.
    target_splits = defaultdict(set)
    for r in samples:
        target_splits[r['processed']['gt']['sha256']].add(r['split'])
    if any(len(s)>1 for s in target_splits.values()):
        raise RuntimeError('Identical GT image crosses splits')
    previews = prep.build_previews(output, {r['id']:r for r in fresh})
    manifest = {'complete':True, 'version':VERSION, 'created_utc':datetime.now(timezone.utc).isoformat(),
        'label_correction':{'status':'applied','new_source_mapping':a.labels,'authority':'explicit user confirmation'},
        'source_archive':{'path':str(a.archive),'sha256':prep.sha256_path(a.archive),'bytes':a.archive.stat().st_size},
        'base_dataset':{'path':str(a.base),'manifest_sha256':prep.sha256_path(a.base/'manifest.json')} if old else None,
        'transform':{'target_pixels':512*384,'dimension_multiple':16,'policy':'preserve aspect, no crop/padding/upscale; Lanczos RGB, BOX DoLP'},
        'split':{'seed':a.seed,'group_key':'prefix before first underscore','sample_counts':dict(counts),
            'group_counts':dict(Counter(assignment[g] for g in {r['group'] for r in samples})),
            'original_splits_preserved':bool(old),'requested_fractions':[.8,.1,.1],
            'train_multiple':4,'adjustment':'whole groups only; four-rank complete epochs without duplication'},
        'scan_stats_new':stats,'preview_samples':previews,'excluded_samples':old.get('excluded_samples',[]) if old else [],'samples':samples}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'prepared','counts':dict(counts),'new':len(fresh),'total':len(samples),'root':str(output)},indent=2))


def reuse_cache(a):
    from .sma_cache import CACHE_VERSION, valid_record
    manifest = json.loads((a.output/'manifest.json').read_text())
    records = {r['id']:r for r in manifest['samples']}
    train = (a.output/'splits/train.txt').read_text().split()
    a.cache.mkdir(parents=True,exist_ok=True)
    for folder in ('samples','records'):(a.cache/folder).mkdir(exist_ok=True)
    reused = []
    old_cache = json.loads((a.reuse_cache/'manifest.json').read_text())
    teacher = old_cache['teacher']['adapter_sha256']
    for sid in train:
        if not valid_record(a.reuse_cache,sid,teacher):continue
        rec = json.loads((a.reuse_cache/'records'/f'{sid}.json').read_text())
        roles = {'input':'input','gt':'gt','p90':'reflection_90','dolp':'dolp'}
        if any(rec['source_sha256'][k]!=records[sid]['processed'][v]['sha256'] for k,v in roles.items()):continue
        for folder, suffix in (('samples','.safetensors'),('records','.json')):
            os.link(a.reuse_cache/folder/f'{sid}{suffix}',a.cache/folder/f'{sid}{suffix}')
        reused.append(sid)
    (a.cache/'reuse_record.json').write_text(json.dumps({'reused_ids':reused,'count':len(reused),'source':str(a.reuse_cache),'teacher_sha256':teacher},indent=2))
    print(json.dumps({'cache_reused':len(reused),'to_extract':len(train)-len(reused)}))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','reuse-cache'])
    p.add_argument('--archive',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base',type=Path);p.add_argument('--labels',choices=['gt-suffix','bare-gt'])
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--cache',type=Path);p.add_argument('--reuse-cache',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':
        if not a.labels or not a.archive:p.error('Explicit --labels and --archive required')
        prepare(a)
    else:reuse_cache(a)


if __name__=='__main__':main()
