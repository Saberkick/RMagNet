"""Delete explicit obsolete personal caches after checking live dependencies."""
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone

ROOT = Path('/share/linmingheng-local/xuke').resolve()
PROJECT = ROOT / 'RMagNet-M6'
archive = PROJECT / 'results_archive/M6/cleanup_20261010'
archive.mkdir(parents=True, exist_ok=True)
cache_root = ROOT / 'RMagNet/data_cache'
names = ['m4_multilayer_v1', 'm4_best_multilayer_v1', 'm2a_q20',
         'sma_m4final_v1', 'sma_gtnoise5_v1', 'sma_dataset2_v1']
protected = cache_root / 'ws_c1_m5_dataset3_s1_e20_layers'
targets = [cache_root / n for n in names]
targets += [PROJECT / 'runs/m6_cache_logs', PROJECT / 'runs/m6_pipeline_e5']
commands, opened = [], []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        command = (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if str(ROOT) not in command:
            continue
        commands.append({'pid': int(proc.name), 'command': command})
        for fd in (proc/'fd').iterdir():
            try:
                opened.append((int(proc.name), fd.resolve(strict=False)))
            except OSError:
                pass
    except (OSError, PermissionError):
        continue
# Archive M4 training identity BEFORE deleting the tensor cache.
lineage = cache_root / 'm4_multilayer_v1/manifest.json'
if lineage.is_file():
    shutil.copy2(lineage, PROJECT / 'docx/M6branch/M6_HISTORICAL_M4_CACHE_MANIFEST.json')
report = {'time_utc': datetime.now(timezone.utc).isoformat(), 'deleted': [],
          'preserved_active_cache': str(protected), 'live_personal_commands': commands}
for p in targets:
    if not p.exists():
        continue
    resolved = p.resolve()
    if p.is_symlink() or not resolved.is_relative_to(ROOT) or resolved == protected.resolve():
        raise RuntimeError(f'Unsafe cleanup target: {p}')
    if resolved.parent not in (cache_root.resolve(), (PROJECT/'runs').resolve()):
        raise RuntimeError(f'Unexpected cleanup parent: {p}')
    if any(str(p) in entry['command'] or str(resolved) in entry['command'] for entry in commands):
        raise RuntimeError(f'Live command refers to {p}')
    if any(f.is_relative_to(resolved) for _, f in opened):
        raise RuntimeError(f'Live file descriptor refers to {p}')
    nbytes = sum(f.stat().st_size for f in p.rglob('*') if f.is_file() and not f.is_symlink())
    destination = archive / p.name
    for f in p.rglob('*'):
        if f.is_file() and f.suffix in ('.json', '.jsonl', '.csv', '.log', '.txt'):
            out = destination / f.relative_to(p)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
    shutil.rmtree(p)
    report['deleted'].append({'path': str(p), 'logical_bytes': nbytes})
report['logical_gib_deleted'] = sum(r['logical_bytes'] for r in report['deleted'])/2**30
report['free_gib_after'] = shutil.disk_usage(ROOT).free/2**30
(archive/'cleanup_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
