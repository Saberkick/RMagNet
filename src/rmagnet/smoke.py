"""CPU-only contract and gradient smoke check; no real-model inference."""

import torch

from .backend import AdapterRoutedBackend, ToySharedBackend
from .fusion import FusionHead
from .interface import InterfaceHead
from .losses import synthetic_losses
from .system import RMagNet


def main() -> None:
    torch.manual_seed(2026)
    model = RMagNet(InterfaceHead(16), ToySharedBackend(12, 4), FusionHead(16))
    image = torch.rand(1, 3, 32, 32)
    interface_mask = torch.zeros(1, 1, 32, 32)
    interface_mask[:, :, 4:28, 4:28] = 1
    edit_mask = torch.zeros_like(interface_mask)
    edit_mask[:, :, 8:24, 8:24] = 1
    clean = (image - 0.2 * edit_mask).clamp(0, 1)
    reflection = (image - clean).clamp(0, 1)

    output = model(image)
    assert output.transmission.shape == image.shape
    assert output.reflection_candidate.shape == image.shape
    assert output.maps.interface.shape == edit_mask.shape
    assert output.uncertainty.shape == edit_mask.shape

    losses = synthetic_losses(
        output, image, clean, reflection, interface_mask, edit_mask
    )
    losses["total"].backward()
    for prefix in (
        "interface_head.",
        "backend.adapters.transmission.",
        "backend.adapters.reflection.",
        "fusion_head.",
    ):
        gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError(f"Missing or nonfinite gradients in {prefix}")
        if not any(g.abs().sum() > 0 for g in gradients):
            raise RuntimeError(f"Zero gradients in {prefix}")

    class FakeAdapterTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.active = ""
            self.visited: list[str] = []

        def set_adapter(self, name: str) -> None:
            self.active = name
            self.visited.append(name)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value * (0.9 if self.active == "T" else 0.1)

    fake = FakeAdapterTransformer()
    routed = AdapterRoutedBackend(
        fake,
        lambda transformer, value, branch: transformer(value),
        {"transmission": "T", "reflection": "R"},
    )
    candidate_t = routed(image, "transmission")
    candidate_r = routed(image, "reflection")
    if fake.visited != ["T", "R"] or torch.allclose(candidate_t, candidate_r):
        raise RuntimeError("Adapter routing did not produce separate branch passes")

    print(
        "smoke_ok "
        f"output={tuple(output.transmission.shape)} "
        f"loss={losses['total'].item():.6f} "
        "interface/T/R/fusion_gradients=ok adapter_routing=ok"
    )


if __name__ == "__main__":
    main()
