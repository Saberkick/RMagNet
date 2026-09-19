from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InterfaceMaps:
    interface: torch.Tensor  # B, 1, H, W
    edit: torch.Tensor  # B, 1, H, W
    keep_confidence: torch.Tensor  # B, 1, H, W


@dataclass(frozen=True)
class ModelOutput:
    transmission_candidate: torch.Tensor  # B, 3, H, W
    reflection_candidate: torch.Tensor  # B, 3, H, W
    transmission: torch.Tensor  # B, 3, H, W
    uncertainty: torch.Tensor  # B, 1, H, W
    maps: InterfaceMaps

