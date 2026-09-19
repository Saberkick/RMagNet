"""M1a three-pass system contract: LoRA_T, LoRA_R, then LoRA_Fuse."""

from __future__ import annotations

import torch
from torch import nn

from .backend import BranchBackend, FusionBackend
from .conditioning import (
    FusionConditionMixer,
    IdentityBranchConditioner,
    NullInterfaceHead,
)
from .contracts import ModelOutput


class M1aRMagNet(nn.Module):
    """Compose two separation passes and one fusion-specific DiT pass.

    A production FusionBackend must route the same frozen transformer to a
    named ``fusion`` adapter. M1a defines and tests the contract; M1b will wire
    the real Qwen flow path and measure its memory.
    """

    def __init__(
        self,
        backend: BranchBackend,
        fusion_backend: FusionBackend,
        *,
        interface_head: nn.Module | None = None,
        transmission_conditioner: nn.Module | None = None,
        reflection_conditioner: nn.Module | None = None,
        fusion_conditioner: FusionConditionMixer | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.fusion_backend = fusion_backend
        self.interface_head = interface_head or NullInterfaceHead()
        self.transmission_conditioner = (
            transmission_conditioner or IdentityBranchConditioner()
        )
        self.reflection_conditioner = (
            reflection_conditioner or IdentityBranchConditioner()
        )
        self.fusion_conditioner = fusion_conditioner or FusionConditionMixer()

    def forward(self, image: torch.Tensor) -> ModelOutput:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected RGB tensor with shape [B, 3, H, W]")
        maps = self.interface_head(image)
        t_input = self.transmission_conditioner(image, maps)
        r_input = self.reflection_conditioner(image, maps)
        transmission = self.backend(t_input, "transmission")
        reflection = self.backend(r_input, "reflection")
        fusion_condition = self.fusion_conditioner(
            image, transmission, reflection, maps
        )
        final, uncertainty = self.fusion_backend(fusion_condition)
        return ModelOutput(
            transmission,
            reflection,
            final,
            uncertainty,
            maps,
        )
