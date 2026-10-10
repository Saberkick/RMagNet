"""GT-anchored positive reflection-direction penalty and strict gradient caps."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from .m6_common import MID, RULE

def normalize_features(q):
    centered = q.float() - q.float().mean(dim=-2, keepdim=True)
    norm = centered.norm(dim=-1)
    return F.normalize(centered, dim=-1, eps=1e-6), norm > 1e-6

def build_directions(q_i, q_gt, q_p90, gate, valid):
    directions, confidence, layer_valid = {}, {}, []
    diagnostics = {}
    for l in MID:
        zi, vi = normalize_features(q_i[l])
        zy, vy = normalize_features(q_gt[l])
        zp, vp = normalize_features(q_p90[l])
        delta_p, delta_gt = zp - zi, zi - zy
        np_, ng = delta_p.norm(dim=-1), delta_gt.norm(dim=-1)
        u = F.normalize(delta_p, dim=-1, eps=1e-6)
        cosine = (u * F.normalize(delta_gt, dim=-1, eps=1e-6)).sum(-1)
        if RULE.get('direction_mode') == 'unfiltered':
            good = torch.isfinite(u).all(-1) & (delta_p.norm(dim=-1) > 1e-12)
            directions[l] = u
            layer_valid.append(good)
            confidence[l] = good.float()
            diagnostics[str(l)] = {'alignment_mean': float(cosine.mean()),
                'norm_i_p90_mean': float(np_.mean()), 'norm_i_gt_mean': float(ng.mean()),
                'valid_fraction': float(good.float().mean()),
                'negative_alignment_fraction': float((cosine < 0).float().mean())}
            continue
        calibrated = RULE.get('direction_mode') == 'gt-calibrated'
        reliability = cosine.abs() if calibrated else cosine
        if calibrated:
            u = u * torch.where(cosine >= 0, 1.0, -1.0)[:, None]
        good = vi & vy & vp & (np_ >= RULE['min_norm']) & (ng >= RULE['min_norm']) & valid & (reliability > RULE['align_low'])
        layer_valid.append(good)
        confidence[l] = ((reliability - RULE['align_low']) / RULE['align_width']).clamp(0, 1)
        directions[l] = u
        diagnostics[str(l)] = {'alignment_mean': float(cosine.mean()),
                               'norm_i_p90_mean': float(np_.mean()),
                               'norm_i_gt_mean': float(ng.mean()),
                               'valid_fraction': float(good.float().mean()),
                               'negative_alignment_fraction': float((cosine < 0).float().mean())}
    if RULE.get('direction_mode') == 'unfiltered':
        mass = float(torch.stack(list(confidence.values())).mean())
        return directions, confidence, {'layers': diagnostics,
            'confidence_mass_before_skip': mass, 'active': mass > 0}
    quorum = torch.stack(layer_valid).sum(0) >= RULE['min_layers']
    masks = {l: gate.float() * confidence[l] * quorum * layer_valid[j]
             for j, l in enumerate(MID)}
    mass = float(torch.stack(list(masks.values())).mean())
    if mass < RULE['min_mass']:
        masks = {l: torch.zeros_like(v) for l, v in masks.items()}
    return directions, masks, {'layers': diagnostics, 'confidence_mass_before_skip': mass,
                               'active': mass >= RULE['min_mass']}

def negative_loss(features, batch, device):
    values, active_fractions = [], []
    for l in MID:
        pred, _ = normalize_features(features[l])
        target, _ = normalize_features(batch[f'q{l}_gt'].to(device))
        u = F.normalize(batch[f'q{l}_negative_direction'].to(device).float(), dim=-1, eps=1e-6)
        mask = batch[f'q{l}_negative_weight'].to(device).float()
        if float(mask.sum()) <= 1e-8:
            continue
        residual = ((pred - target) * u).sum(-1)
        error = residual.square() if RULE.get('direction_mode') == 'unfiltered' else residual.clamp_min(0).square()
        values.append((mask * error).sum() / mask.sum())
        active_fractions.append((mask * (residual > 0).float()).sum() / mask.sum())
    if not values:
        z = features[MID[0]].sum().float() * 0.0
        return z, {'negative_active': 0.0, 'negative_positive_residual_fraction': 0.0}
    return torch.stack(values).mean(), {
        'negative_active': 1.0,
        'negative_positive_residual_fraction': float(torch.stack(active_fractions).mean().detach())}

def norm(t):
    return float(t.float().norm().detach())

def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float((a.float() * b.float()).sum().detach()) / (na * nb)

class M6GradientController:
    def __init__(self, warmup=36):
        self.targets = {'spatial': 0.08, 'texture': 0.08,
                        'semantic': 0.06, 'negative': 0.02}
        self.warmup = warmup
        self.ema = {}

    def combine(self, base, auxiliaries, step):
        nb = norm(base)
        if not math.isfinite(nb) or nb <= 1e-12:
            raise RuntimeError('Invalid base output gradient')
        ramp = min(1.0, step / self.warmup)
        scaled, logs = {}, {}
        for name, gradient in auxiliaries.items():
            ng = norm(gradient)
            if not math.isfinite(ng):
                raise RuntimeError(f'Nonfinite {name} output gradient')
            target = self.targets[name] * ramp
            if ng <= 1e-12 or target == 0:
                scale = 0.0
            else:
                raw = min(10.0, target * nb / ng)
                prev = self.ema.get(name, raw)
                self.ema[name] = 0.9 * prev + 0.1 * raw
                scale = min(self.ema[name], target * nb / ng)
            scaled[name] = gradient * scale
            logs[f'{name}_raw_norm'] = ng
            logs[f'{name}_scale'] = scale
            logs[f'{name}_ratio_before_group_cap'] = norm(scaled[name]) / nb
        semantic = scaled['semantic'] + scaled['negative']
        s_cap = min(1.0, 0.08 * ramp * nb / max(norm(semantic), 1e-12))
        auxiliary = scaled['spatial'] + scaled['texture'] + semantic * s_cap
        a_cap = min(1.0, 0.25 * nb / max(norm(auxiliary), 1e-12))
        logs.update({'actual_aux_base_ratio': norm(auxiliary * a_cap) / nb,
                     'semantic_combined_ratio': norm(semantic * s_cap * a_cap) / nb,
                     'negative_actual_ratio': norm(scaled['negative'] * s_cap * a_cap) / nb,
                     'negative_base_cosine': cosine(auxiliaries['negative'], base),
                     'negative_positive_cosine': cosine(auxiliaries['negative'], auxiliaries['semantic']),
                     'semantic_cap_scale': s_cap, 'aux_cap_scale': a_cap,
                     'base_output_gradient_norm': nb})
        return base + auxiliary * a_cap, logs
