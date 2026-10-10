"""Check fresh three-view references against a legacy cache before reuse."""
import json
import os
import torch
import torch.nn.functional as F
from src.rmagnet.m6_common import DATA, ROOT, MID, load_backend, source_paths, source_hashes
from src.rmagnet.m6_cache import capture, old_reference
from src.rmagnet.m6_losses import normalize_features
from src.rmagnet.stage1_train import image_tensor

torch.cuda.set_device(0)
device = torch.device('cuda:0')
print(json.dumps({'event': 'binding', 'visible': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'uuid': str(torch.cuda.get_device_properties(0).uuid), 'pid': os.getpid()}), flush=True)
backend = load_backend(device)
backend.set_trainable_branch(None)
backend.transformer.eval()
backend.vae.eval()
for sid in ('101_2292_1137', '103_831_2184'):
    paths = source_paths(DATA, sid)
    fresh = {role: capture(backend, image_tensor(path)[None], MID)
             for role, path in paths.items()}
    old = old_reference(ROOT / 'RMagNet/data_cache/m4_multilayer_v1', sid, source_hashes(DATA, sid))
    result = {'id': sid, 'layers': {}}
    for l in MID:
        zi, _ = normalize_features(fresh['input'][l][0])
        zg, _ = normalize_features(fresh['gt'][l][0])
        zp, _ = normalize_features(fresh['p90'][l][0])
        cosine = (F.normalize(zp - zi, dim=-1) * F.normalize(zi - zg, dim=-1)).sum(-1)
        row = {'fresh_alignment_mean': float(cosine.mean()),
               'fresh_fraction_above_point2': float((cosine > .2).float().mean())}
        if old is not None:
            legacy = old[f'q{l}_gt'].float()
            now = fresh['gt'][l][0].float()
            row['legacy_gt_relative_l2'] = float((legacy-now).norm()/now.norm())
            row['legacy_gt_cosine'] = float(F.cosine_similarity(legacy, now, dim=-1).mean())
        result['layers'][str(l)] = row
    print(json.dumps(result), flush=True)
