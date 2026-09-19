import torch
from torch import nn

from .contracts import InterfaceMaps


class FusionHead(nn.Module):
    """Fuse two candidate images and interface evidence in pixel space."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
        )
        self.output = nn.Conv2d(width, 4, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        image: torch.Tensor,
        transmission: torch.Tensor,
        reflection: torch.Tensor,
        maps: InterfaceMaps,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if transmission.shape != image.shape or reflection.shape != image.shape:
            raise ValueError("Both branch candidates must match input RGB shape")
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
        residual, uncertainty_logit = self.output(self.features(evidence)).split(
            [3, 1], dim=1
        )
        result = image + maps.edit * (transmission - image + residual)
        return result.clamp(0, 1), torch.sigmoid(uncertainty_logit)

