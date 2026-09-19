import torch
from torch import nn

from .backend import BranchBackend
from .contracts import ModelOutput
from .fusion import FusionHead
from .interface import InterfaceHead


class RMagNet(nn.Module):
    def __init__(
        self,
        interface_head: InterfaceHead,
        backend: BranchBackend,
        fusion_head: FusionHead,
    ) -> None:
        super().__init__()
        self.interface_head = interface_head
        self.backend = backend
        self.fusion_head = fusion_head

    def forward(self, image: torch.Tensor) -> ModelOutput:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected RGB tensor with shape [B, 3, H, W]")
        maps = self.interface_head(image)
        transmission_candidate = self.backend(image, "transmission")
        reflection_candidate = self.backend(image, "reflection")
        transmission, uncertainty = self.fusion_head(
            image, transmission_candidate, reflection_candidate, maps
        )
        return ModelOutput(
            transmission_candidate,
            reflection_candidate,
            transmission,
            uncertainty,
            maps,
        )

