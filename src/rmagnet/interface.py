import torch
from torch import nn

from .contracts import InterfaceMaps


class InterfaceHead(nn.Module):
    """Small trainable contract prototype; replace/condition on frozen features later."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
        )
        self.output = nn.Conv2d(width, 3, 1)

    def forward(self, image: torch.Tensor) -> InterfaceMaps:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected RGB tensor with shape [B, 3, H, W]")
        logits = self.output(self.features(image))
        interface, edit, keep = torch.sigmoid(logits).split(1, dim=1)
        return InterfaceMaps(interface, edit, keep)

