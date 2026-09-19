"""Condition adapters for the optional interface path and DiT fusion pass."""

from __future__ import annotations

import torch
from torch import nn

from .contracts import InterfaceMaps


def null_interface_maps(image: torch.Tensor) -> InterfaceMaps:
    """Return a neutral condition without inventing interface evidence."""
    shape = (image.shape[0], 1, image.shape[2], image.shape[3])
    zeros = image.new_zeros(shape)
    ones = image.new_ones(shape)
    return InterfaceMaps(zeros, zeros, ones)


class NullInterfaceHead(nn.Module):
    """Disable the optional Interface Head while preserving tensor contracts."""

    def forward(self, image: torch.Tensor) -> InterfaceMaps:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected RGB tensor with shape [B, 3, H, W]")
        return null_interface_maps(image)


class BranchConditionMixer(nn.Module):
    """Convert RGB plus interface maps back to RGB before a T/R DiT pass.

    The final projection is zero initialized, so construction is exactly the
    identity and cannot change the established WindowSeat T baseline.
    """

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(6, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, 3, 3, padding=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, image: torch.Tensor, maps: InterfaceMaps) -> torch.Tensor:
        evidence = torch.cat(
            [image, maps.interface, maps.edit, maps.keep_confidence], dim=1
        )
        return (image + self.body(evidence)).clamp(0, 1)


class IdentityBranchConditioner(nn.Module):
    """M1a default: leave T/R inputs unchanged."""

    def forward(self, image: torch.Tensor, maps: InterfaceMaps) -> torch.Tensor:
        return image


class FusionConditionMixer(nn.Module):
    """Compress I/T/R/interface evidence to an RGB condition for LoRA_Fuse.

    Qwen's VAE/DiT latent width is fixed, so raw channel concatenation cannot be
    passed to the existing transformer. This mixer keeps T as an exact residual
    anchor at initialization and learns only the condition supplied to the
    third, fusion-specific DiT pass.
    """

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(12, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, 3, 3, padding=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(
        self,
        image: torch.Tensor,
        transmission: torch.Tensor,
        reflection: torch.Tensor,
        maps: InterfaceMaps,
    ) -> torch.Tensor:
        if image.shape != transmission.shape or image.shape != reflection.shape:
            raise ValueError("I, T and R must have identical RGB shapes")
        evidence = torch.cat(
            [
                image,
                transmission,
                reflection,
                maps.interface,
                maps.edit,
                maps.keep_confidence,
            ],
            dim=1,
        )
        return (transmission + self.body(evidence)).clamp(0, 1)
