"""C1: spatial semantic modulation of image-FFN LoRA with benefit gates."""
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .sma import Projection, CleanMemory
from .sma_joint import canonical
from .m4_cache import atomic_safetensors
from safetensors import safe_open
import safetensors.torch

C_VERSION = 'sma-conditioned-c1-v1'
TARGETS = ('39_in', '39_out', '41_in', '41_out')


class Condition(nn.Module):
    def __init__(self, dim=256, rank=128):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, rank))
        nn.init.normal_(self.project[-1].weight, std=.02)
        nn.init.zeros_(self.project[-1].bias)
        self.gamma = nn.Parameter(torch.zeros(()))
        self.gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def logits(self, memory):
        return self.gate(memory.detach().float())

    def forward(self, low_rank, memory, mode):
        if low_rank.shape[:2] != memory.shape[:2] or low_rank.shape[-1] != 128:
            raise RuntimeError('C image-token / low-rank geometry mismatch')
        if mode == 'off':
            return low_rank
        with torch.autocast('cuda', enabled=False):
            g = torch.ones_like(memory[..., :1]) if mode == 'on' else self.logits(memory).sigmoid()
            modulation = 1 + .25 * self.gamma.tanh() * g * self.project(memory.float()).tanh()
            return low_rank * modulation.to(low_rank.dtype)


class ConditionedSMA(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = Projection()
        self.memory = CleanMemory()
        self.readers = nn.ModuleDict({k: Condition() for k in TARGETS})

    def freeze_memory(self):
        self.projection.requires_grad_(False)
        self.memory.requires_grad_(False).eval()

    def load_memory(self, state):
        expected = {k for k in self.state_dict() if not k.startswith('readers.')}
        selected = {k: v for k, v in state.items() if k in expected}
        if set(selected) != expected:
            raise RuntimeError('C pretrained memory keys missing')
        result = self.load_state_dict(selected, strict=False)
        if result.unexpected_keys or any(not k.startswith('readers.') for k in result.missing_keys):
            raise RuntimeError('C memory loading mismatch')

    def gate_parameters(self):
        return [p for m in self.readers.values() for p in m.gate.parameters()]


@dataclass
class State:
    enabled: bool = False
    grid: tuple | None = None
    memory: torch.Tensor | None = None
    mode: str = 'learned'


class Runtime:
    def __init__(self, sma):
        self.sma = sma
        self.state = State()
        self.default_mode = 'learned'
        self.last_memory = None
        self.hits = {k: 0 for k in TARGETS}

    @contextmanager
    def scope(self, enabled, grid=None, memory=None, mode=None):
        previous = self.state
        self.state = State(enabled, grid, memory, mode or self.default_mode)
        try:
            yield
        finally:
            self.state = previous

    @contextmanager
    def mode(self, mode):
        previous = self.default_mode
        self.default_mode = mode
        try:
            yield
        finally:
            self.default_mode = previous

    def checkpoint(self, function, *args, **kwargs):
        state = State(self.state.enabled, self.state.grid, self.state.memory, self.state.mode)
        if not any(isinstance(v, torch.Tensor) and v.requires_grad for v in args):
            return function(*args, **kwargs)
        outer, first_call = self.state, True
        def bound(*values):
            nonlocal first_call
            with self.scope(state.enabled, state.grid, state.memory, state.mode):
                result = function(*values, **kwargs)
                if first_call:
                    outer.memory = self.state.memory
                    first_call = False
                return result
        return checkpoint(bound, *args, use_reentrant=False)


class MemoryBlock(nn.Module):
    def __init__(self, original, runtime):
        super().__init__()
        self.original, self.runtime = original, runtime

    def forward(self, *args, **kwargs):
        text, image = self.original(*args, **kwargs)
        state = self.runtime.state
        if state.enabled:
            if state.grid is None or image.shape[1] != state.grid[0]*state.grid[1]:
                raise RuntimeError('C memory token grid mismatch')
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                state.memory = self.runtime.sma.memory(self.runtime.sma.projection(image), state.grid).detach()
                self.runtime.last_memory = state.memory
        return text, image


def install(backend, sma):
    runtime = Runtime(sma)
    backend.sma, backend.sma_runtime = sma, runtime
    blocks = backend.transformer.transformer_blocks
    blocks[36] = MemoryBlock(blocks[36], runtime)
    for number in (39, 41):
        block = blocks[number-1]
        for suffix, module in [('in', block.img_mlp.net[0].proj), ('out', block.img_mlp.net[2])]:
            key = f'{number}_{suffix}'
            if module.r['default'] != 128 or getattr(module.lora_dropout['default'], 'p', 0) != 0:
                raise RuntimeError('C expects pinned rank128, dropout0 LoRA')
            def hook(_module, _args, output, key=key):
                state = runtime.state
                if not state.enabled:
                    return output
                if state.memory is None:
                    raise RuntimeError('C condition used before Q37 memory')
                runtime.hits[key] += 1
                return sma.readers[key](output, state.memory, state.mode)
            module.lora_A['default'].register_forward_hook(hook)
    backend.transformer.enable_gradient_checkpointing(gradient_checkpointing_func=runtime.checkpoint)
    def normalized(self, image, branch):
        if branch != 'transmission':
            raise ValueError('C supports transmission only')
        from .qwen_layer_probe import deterministic_encode
        with torch.no_grad():
            latent = deterministic_encode(self, image)
        return forward(self, latent)
    backend.forward_normalized = MethodType(normalized, backend)
    return runtime


def forward(backend, latent):
    with backend.sma_runtime.scope(True, (latent.shape[-2]//2, latent.shape[-1]//2)):
        edited = backend.upstream.flow_step(latent, backend.transformer, backend.vae, backend.embeddings)
        return backend.upstream.decode(edited, backend.vae)


def save_joint(path, backend, m4_sha):
    state = {'sma.'+k: v for k, v in backend.sma.state_dict().items()}
    state.update({'lora.'+canonical(k): v for k, v in backend.transformer.named_parameters() if '.lora_' in k and '.default.' in k})
    atomic_safetensors(path, state, {'architecture': C_VERSION, 'base_m4_sha256': m4_sha, 'experiment': 'C1', 'contains': 'student LoRA and full conditioned semantic memory'})


def load_joint(path, backend, sma, device, m4_sha):
    with safe_open(path, framework='pt') as f:
        md = f.metadata()
    if md.get('architecture') != C_VERSION or md.get('base_m4_sha256') != m4_sha:
        raise RuntimeError('C checkpoint identity mismatch')
    state = safetensors.torch.load_file(path, device=str(device))
    expected = {k for k, _ in backend.transformer.named_parameters() if '.lora_' in k and '.default.' in k}
    lora = {k[5:]: v for k, v in state.items() if k.startswith('lora.')}
    if set(lora) != expected:
        raise RuntimeError('C LoRA layout mismatch; load before install')
    _, unexpected = backend.transformer.load_state_dict(lora, strict=False)
    if unexpected:
        raise RuntimeError(unexpected)
    sma.load_state_dict({k[4:]: v for k, v in state.items() if k.startswith('sma.')}, strict=True)
    if len(state) != len(lora)+len(sma.state_dict()):
        raise RuntimeError('Unexpected C checkpoint tensors')
    return md


def local_error(image, target):
    diff = (image.float()-target.float()).abs().mean(1, keepdim=True)
    dx = ((image[..., 1:]-image[..., :-1])-(target[..., 1:]-target[..., :-1])).abs().mean(1, keepdim=True)
    dy = ((image[..., 1:, :]-image[..., :-1, :])-(target[..., 1:, :]-target[..., :-1, :])).abs().mean(1, keepdim=True)
    error = diff + .1*(F.pad(dx, (0, 1, 0, 0))+F.pad(dy, (0, 0, 0, 1)))
    # No artificial zero-padding decrease of border errors.
    return F.avg_pool2d(error, 9, stride=1, padding=4, count_include_pad=False)


def calibrate(backend, latent, target, step, epoch, interval=8, force=False):
    """Extra fixed-weight counterfactuals; BCE gradients touch gate heads only."""
    import torch.distributed as dist
    runtime, sma = backend.sma_runtime, backend.sma
    with torch.no_grad():
        normal_memory = runtime.last_memory
        with runtime.mode('off'):
            t0 = forward(backend, latent).detach()
        m0 = runtime.last_memory
        with runtime.mode('on'):
            t1 = forward(backend, latent).detach()
        m1 = runtime.last_memory
        if not torch.equal(m0, m1):
            raise RuntimeError('Counterfactual Q37 memory changed before intervention')
        grid = (latent.shape[-2]//2, latent.shape[-1]//2)
        delta = F.adaptive_avg_pool2d(local_error(t0, target)-local_error(t1, target), grid).flatten(2).transpose(1, 2)
        raw_tau = local_error(t0, target).mean().clamp(.002, .1)
        if dist.is_initialized():
            dist.all_reduce(raw_tau); raw_tau /= dist.get_world_size()
        old = getattr(runtime, 'tau', None)
        runtime.tau = float(raw_tau) if old is None else .9*old+.1*float(raw_tau)
        tau = runtime.tau
        y = (delta/tau).sigmoid().detach()
        weight = (delta.abs()/tau).clamp(max=1).detach()
        memory = m0.detach()
        benefit_fraction = (delta > 0).float().mean()
        output_delta = (t1-t0).abs().mean()
    params = sma.gate_parameters()
    with torch.autocast('cuda', enabled=False):
        losses = []
        gs = []
        for module in sma.readers.values():
            logits = module.logits(memory)
            losses.append((F.binary_cross_entropy_with_logits(logits, y, reduction='none')*weight).sum()/(weight.sum()+1e-8))
            gs.append(logits.sigmoid().detach().mean())
        loss = torch.stack(losses).mean()
    aux = list(torch.autograd.grad(loss, params))
    # Measure the actual DDP-averaged gate gradients before computing the cap.
    main_global = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]
    aux_global = [g.detach().clone() for g in aux]
    if dist.is_initialized():
        for g in main_global+aux_global:
            dist.all_reduce(g); g /= dist.get_world_size()
    norm = lambda gs: torch.stack([v.float().square().sum() for v in gs]).sum().sqrt()
    main_norm, aux_norm = norm(main_global), norm(aux_global)
    coefficient = min(1., .1*float(main_norm)/(float(aux_norm)+1e-12))
    if not all(torch.isfinite(g).all() for g in aux) or not torch.isfinite(loss):
        raise RuntimeError('Nonfinite C reliability gradients')
    for p, g in zip(params, aux):
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        p.grad.add_(g, alpha=coefficient)
    return {'calibration': True, 'reliability_loss': float(loss.detach()), 'reliability_tau': tau,
            'reliability_coefficient': coefficient, 'reliability_gate_gradient_ratio': coefficient*float(aux_norm)/(float(main_norm)+1e-12),
            'gate_mean': float(torch.stack(gs).mean()), 'condition_beneficial_token_fraction': float(benefit_fraction),
            'condition_output_l1': float(output_delta), 'counterfactual_memory_identical': True}


def diagnostics(sma, runtime):
    with torch.no_grad():
        if runtime.last_memory is None:
            raise RuntimeError('C condition memory inactive')
        return {'condition_gamma': {k: float(.25*m.gamma.tanh()) for k, m in sma.readers.items()},
                'gate_mean': float(torch.stack([m.logits(runtime.last_memory).sigmoid().mean() for m in sma.readers.values()]).mean()),
                'condition_hooks_active': all(v > 0 for v in runtime.hits.values())}
