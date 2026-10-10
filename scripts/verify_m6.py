"""Numerical direction/cap checks without loading Qwen or creating checkpoints."""
import json
import torch
from src.rmagnet.m6_losses import build_directions, negative_loss, M6GradientController
from src.rmagnet.m6_common import MID, DIRECTION_MODE

torch.manual_seed(2026)
n, c = 64, 3072
gt = torch.randn(n, c)
reflection = torch.randn(n, c)
i, p = gt + 0.35 * reflection, gt + 0.7 * reflection
u, mask, stats = build_directions({l: i for l in MID}, {l: gt for l in MID},
                                {l: p for l in MID}, torch.ones(n), torch.ones(n, dtype=torch.bool))
assert stats['active']
batch = {}
for l in MID:
    batch[f'q{l}_gt'] = gt[None]
    batch[f'q{l}_negative_direction'] = u[l][None]
    batch[f'q{l}_negative_weight'] = mask[l][None]
zero, _ = negative_loss({l: gt[None].clone().requires_grad_() for l in MID}, batch, torch.device('cpu'))
assert float(zero.detach()) < 1e-10
pred = i[None].clone().requires_grad_()
positive, log = negative_loss({l: pred for l in MID}, batch, torch.device('cpu'))
gradient = torch.autograd.grad(positive, pred)[0]
assert float(positive.detach()) > 0 and torch.isfinite(gradient).all() and float(gradient.norm()) > 0
# A descent update must reduce the penalty under this controlled construction.
updated, _ = negative_loss({l: pred.detach() - gradient * 100 for l in MID}, batch, torch.device('cpu'))
assert float(updated.detach()) < float(positive.detach())
inactive_batch = {k: torch.zeros_like(v) if k.endswith('weight') else v for k, v in batch.items()}
inactive, _ = negative_loss({l: pred for l in MID}, inactive_batch, torch.device('cpu'))
inactive_grad = torch.autograd.grad(inactive, pred)[0]
assert float(inactive.detach()) == 0 and float(inactive_grad.abs().max()) == 0
_, bad_masks, bad_stats = build_directions({l: gt for l in MID}, {l: gt for l in MID},
                                         {l: gt for l in MID}, torch.ones(n), torch.ones(n, dtype=torch.bool))
assert not bad_stats['active'] and all(float(v.sum()) == 0 for v in bad_masks.values())
# Reversed polarization motion must remain rejected in strict mode, while the
# explicit GT-calibrated mode orients the candidate axis toward I-GT.
reverse_u, reverse_masks, reverse_stats = build_directions(
    {l: i for l in MID}, {l: gt for l in MID}, {l: gt for l in MID},
    torch.ones(n), torch.ones(n, dtype=torch.bool))
assert reverse_stats['active'] == (DIRECTION_MODE in ('gt-calibrated', 'unfiltered'))
if DIRECTION_MODE in ('gt-calibrated', 'unfiltered'):
    reverse_batch = {f'q{l}_{role}': value[None]
                     for l in MID for role, value in (
                         ('gt', gt), ('negative_direction', reverse_u[l]),
                         ('negative_weight', reverse_masks[l]))}
    reverse_loss, _ = negative_loss({l: pred for l in MID}, reverse_batch, torch.device('cpu'))
    assert float(reverse_loss.detach()) > 0
controller = M6GradientController()
base = torch.randn(1, 3, 12, 12)
aux = {k: torch.randn_like(base) * 10**j for j, k in enumerate(controller.targets)}
combined, ratios = controller.combine(base, aux, 100)
assert ratios['actual_aux_base_ratio'] <= 0.25 + 1e-6
assert ratios['semantic_combined_ratio'] <= 0.08 + 1e-6
assert ratios['negative_actual_ratio'] <= 0.02 + 1e-6
aux['negative'].zero_()
_, inactive_ratios = controller.combine(base, aux, 101)
assert inactive_ratios['negative_actual_ratio'] == 0
print(json.dumps({'status': 'passed', 'direction_mode': DIRECTION_MODE, 'tests': ['GT zero', 'positive residual nonzero',
    'descent direction', 'empty support connected zero', 'identical views skip',
    '25pct total cap', '8pct semantic cap', '2pct negative cap', 'inactive gradient exactly zero',
    'reversed polarization motion uses only the explicitly selected polarity mode'],
    'synthetic_positive_loss': float(positive.detach()), 'confidence_mass': stats['confidence_mass_before_skip']}, indent=2))
