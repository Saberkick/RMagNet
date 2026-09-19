"""CPU-only M1a contract, initialization and stage-freezing checks."""

import torch

from .backend import ToyFusionBackend, ToySharedBackend
from .conditioning import FusionConditionMixer
from .m1a import M1aRMagNet


def _set_only(model: torch.nn.Module, prefixes: tuple[str, ...]) -> list[str]:
    selected = []
    for name, parameter in model.named_parameters():
        enabled = name.startswith(prefixes)
        parameter.requires_grad_(enabled)
        if enabled:
            selected.append(name)
    return selected


def main() -> None:
    torch.manual_seed(2026)
    model = M1aRMagNet(
        ToySharedBackend(12, 4),
        ToyFusionBackend(12),
        fusion_conditioner=FusionConditionMixer(12),
    )
    image = torch.rand(1, 3, 32, 32)

    maps = model.interface_head(image)
    t = model.backend(image, "transmission")
    r = model.backend(image, "reflection")
    condition = model.fusion_conditioner(image, t, r, maps)
    if not torch.equal(condition, t):
        raise RuntimeError("Fusion conditioner is not T-anchored at initialization")

    output = model(image)
    if output.transmission.shape != image.shape:
        raise RuntimeError("M1a output shape mismatch")

    stages = {
        "reflection": ("backend.adapters.reflection.", "backend.heads.reflection."),
        "transmission": ("backend.adapters.transmission.", "backend.heads.transmission."),
        "fusion": ("fusion_conditioner.", "fusion_backend."),
    }
    counts = {}
    for stage, prefixes in stages.items():
        names = _set_only(model, prefixes)
        if not names:
            raise RuntimeError(f"No parameters selected for {stage}")
        fresh = model(image)
        if stage == "reflection":
            loss = fresh.reflection_candidate.mean()
        elif stage == "transmission":
            loss = fresh.transmission_candidate.mean()
        else:
            loss = fresh.transmission.mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        grads = [
            p.grad
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError(f"Bad gradients for {stage}")
        counts[stage] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"m1a_smoke_ok shape={tuple(output.transmission.shape)} trainable={counts}")


if __name__ == "__main__":
    main()
