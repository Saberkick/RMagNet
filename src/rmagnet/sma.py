"""Spatial clean-content memory and checkpoint-safe reads inside frozen M4."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SMA_VERSION = "sma-rms-v1"


def positions(grid, device, dtype):
    h, w = grid
    y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=device),
                          torch.linspace(-1, 1, w, device=device), indexing="ij")
    return torch.stack((x, y), -1).reshape(1, h*w, 2).to(dtype)


class Projection(nn.Module):
    def __init__(self, hidden=3072, dim=256):
        super().__init__()
        self.register_buffer("basis", torch.zeros(hidden, dim))
        self.register_buffer("mean", torch.zeros(hidden))
        self.register_buffer("scale", torch.ones(dim))

    def forward(self, h):
        return (F.layer_norm(h.float(), (h.shape[-1],)) - self.mean) @ self.basis


class MemoryBlock(nn.Module):
    def __init__(self, dim=256, slots=16):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, slots, dim)*0.02)
        self.pos = nn.Linear(2, dim)
        self.norm = nn.LayerNorm(dim)
        self.pool = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.read = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim))

    def forward(self, x, grid):
        h, w = grid
        p = self.pos(positions(grid, x.device, x.dtype))
        n = self.norm(x)
        q = self.queries.expand(x.shape[0], -1, -1)
        u = self.pool(q, n+p, n, need_weights=False)[0]
        local = self.local(n.transpose(1, 2).reshape(x.shape[0], -1, h, w)).flatten(2).transpose(1, 2)
        x = x + local + self.read(n+p, u, u, need_weights=False)[0]
        return x + self.ffn(x)


class CleanMemory(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.blocks = nn.ModuleList([MemoryBlock(dim), MemoryBlock(dim)])
        self.delta = nn.Linear(dim, dim)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, z, grid):
        x = z
        for block in self.blocks:
            x = block(x, grid)
        return z + self.delta(x)


class MemoryRead(nn.Module):
    def __init__(self, hidden=3072, dim=256):
        super().__init__()
        self.down = nn.Linear(hidden, dim)
        self.local = nn.Sequential(nn.Linear(dim*2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pool_queries = nn.Parameter(torch.randn(1, 16, dim)*0.02)
        self.pos = nn.Linear(2, dim)
        self.pool = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.read = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.gate = nn.Linear(dim*2, 1)
        self.up = nn.Linear(dim, hidden)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, h, memory, grid):
        dtype = h.dtype
        # FP32 trainable adapters; frozen hidden states stay BF16.
        q = self.down(F.layer_norm(h.float(), (h.shape[-1],)))
        m = memory.float()
        p = self.pos(positions(grid, h.device, torch.float32))
        u = self.pool(self.pool_queries.expand(h.shape[0], -1, -1), m+p, m, need_weights=False)[0]
        local = self.local(torch.cat((q, m), -1))
        context = self.read(q+p, u, u, need_weights=False)[0]
        gate = torch.sigmoid(self.gate(torch.cat((q, m), -1)))
        # Qwen's residual stream has RMS around 1e6 in the pinned checkpoint.
        # Unit-scale additive updates disappear in BF16 and yield ~1e-9 grads.
        # Inject a bounded relative residual; detach scale to avoid changing
        # the original stream's Jacobian through the normalization statistic.
        rms = h.float().square().mean(-1, keepdim=True).sqrt().detach().clamp_min(1.0)
        delta = gate * rms * torch.tanh(self.up(local+context))
        return h + delta.to(dtype)


class SMA(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = Projection()
        self.memory = CleanMemory()
        self.readers = nn.ModuleDict({str(n): MemoryRead() for n in (39, 41)})

    def freeze_memory(self):
        for p in self.projection.parameters(): p.requires_grad_(False)
        for p in self.memory.parameters(): p.requires_grad_(False)
        self.memory.eval()


@dataclass
class State:
    enabled: bool = False
    grid: tuple[int, int] | None = None
    memory: torch.Tensor | None = None


class Runtime:
    def __init__(self, sma):
        self.sma = sma
        self.state = State()

    @contextmanager
    def scope(self, enabled, grid=None, memory=None):
        previous = self.state
        self.state = State(enabled, grid, memory)
        try: yield
        finally: self.state = previous

    def checkpoint(self, function, *args, **kwargs):
        # Freeze the branch and detached memory in this call's closure. Teacher
        # VJPs and later image forwards must not change checkpoint recomputation.
        state = State(self.state.enabled, self.state.grid, self.state.memory)
        owner = function if isinstance(function, nn.Module) else getattr(function, "__self__", None)
        inputs_require_grad = any(isinstance(x, torch.Tensor) and x.requires_grad for x in args)
        is_reader = isinstance(owner, MemoryWrappedBlock) and owner.number in (39, 41)
        if not inputs_require_grad and not (state.enabled and is_reader):
            return function(*args, **kwargs)
        outer = self.state
        first_call = True
        def bound(*values):
            nonlocal first_call
            with self.scope(state.enabled, state.grid, state.memory):
                result = function(*values, **kwargs)
                # Joint LoRA makes Q37 checkpointed too. Its detached memory
                # must reach Q39/Q41 during the original forward; recomputation
                # keeps its own captured scope and must not overwrite another pass.
                if first_call:
                    outer.memory = self.state.memory
                    first_call = False
                return result
        return checkpoint(bound, *args, use_reentrant=False)


class MemoryWrappedBlock(nn.Module):
    def __init__(self, original, runtime, number):
        super().__init__()
        self.original = original
        # Runtime is not an nn.Module. SMA owns and saves new weights separately.
        self.runtime = runtime
        self.number = number

    def forward(self, *args, **kwargs):
        text, image = self.original(*args, **kwargs)
        state = self.runtime.state
        if state.enabled:
            if state.grid is None or image.shape[1] != state.grid[0]*state.grid[1]:
                raise RuntimeError("SMA image token grid mismatch")
            if self.number == 37:
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    z = self.runtime.sma.projection(image)
                    state.memory = self.runtime.sma.memory(z, state.grid).detach()
            else:
                if state.memory is None:
                    raise RuntimeError("Memory was not generated before read")
                with torch.autocast("cuda", enabled=False):
                    image = self.runtime.sma.readers[str(self.number)](image, state.memory, state.grid)
        return text, image


def install(backend, sma):
    runtime = Runtime(sma)
    backend.sma = sma
    backend.sma_runtime = runtime
    for n in (37, 39, 41):
        original = backend.transformer.transformer_blocks[n-1]
        backend.transformer.transformer_blocks[n-1] = MemoryWrappedBlock(original, runtime, n)
    backend.transformer.enable_gradient_checkpointing(gradient_checkpointing_func=runtime.checkpoint)

    def normalized(self, image, branch):
        if branch != "transmission": raise ValueError("SMA only supports transmission")
        from .qwen_layer_probe import deterministic_encode
        with torch.no_grad(): latent = deterministic_encode(self, image)
        return forward(self, latent)
    backend.forward_normalized = MethodType(normalized, backend)
    return runtime


def forward(backend, latent):
    grid = (latent.shape[-2]//2, latent.shape[-1]//2)
    with backend.sma_runtime.scope(True, grid):
        edited = backend.upstream.flow_step(latent, backend.transformer, backend.vae, backend.embeddings)
        return backend.upstream.decode(edited, backend.vae)
