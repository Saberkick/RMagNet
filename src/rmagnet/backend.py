from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

import torch
from torch import nn


BRANCHES = ("transmission", "reflection")


class BranchBackend(nn.Module, ABC):
    """One shared backbone, with a separate computation for each branch."""

    @abstractmethod
    def forward(self, image: torch.Tensor, branch: str) -> torch.Tensor:
        """Return a [B, 3, H, W] candidate in [0, 1]."""


class FusionBackend(nn.Module, ABC):
    """Run the third pass using a fusion-specific adapter."""

    @abstractmethod
    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return final RGB and one-channel uncertainty."""


class AdapterRoutedBackend(BranchBackend):
    """Route adapters on one Transformer object; Qwen-specific forward is injected.

    This is an integration seam, not a WindowSeat implementation. In particular,
    `run_branch` must implement the exact VAE/latent/flow path and output mapping.
    Adapter switching assumes serialized branch forwards in one process.
    """

    def __init__(
        self,
        transformer: nn.Module,
        run_branch: Callable[[nn.Module, torch.Tensor, str], torch.Tensor],
        adapter_names: Mapping[str, str],
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.run_branch = run_branch
        self.adapter_names = dict(adapter_names)
        if set(self.adapter_names) != set(BRANCHES):
            raise ValueError("Provide one adapter name for each T/R branch")
        if not hasattr(transformer, "set_adapter"):
            raise TypeError("Transformer must support set_adapter")

    def forward(self, image: torch.Tensor, branch: str) -> torch.Tensor:
        if branch not in BRANCHES:
            raise ValueError(f"Unknown branch: {branch}")
        self.transformer.set_adapter(self.adapter_names[branch])
        candidate = self.run_branch(self.transformer, image, branch)
        if candidate.shape != image.shape:
            raise ValueError("Branch output must match input RGB shape")
        return candidate


class ToySharedBackend(BranchBackend):
    """Tiny shared trunk with separate low-rank branch residuals for CPU checks.

    It does not approximate Qwen quality, parameter count, or memory use.
    """

    def __init__(self, width: int = 16, rank: int = 4) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
        )
        self.adapters = nn.ModuleDict(
            {
                branch: nn.Sequential(
                    nn.Conv2d(width, rank, 1, bias=False),
                    nn.SiLU(),
                    nn.Conv2d(rank, width, 1, bias=False),
                )
                for branch in BRANCHES
            }
        )
        self.heads = nn.ModuleDict(
            {branch: nn.Conv2d(width, 3, 1) for branch in BRANCHES}
        )

    def forward(self, image: torch.Tensor, branch: str) -> torch.Tensor:
        if branch not in BRANCHES:
            raise ValueError(f"Unknown branch: {branch}")
        hidden = self.shared(image)
        hidden = hidden + self.adapters[branch](hidden)
        logits = self.heads[branch](hidden)
        if branch == "transmission":
            return (image + 0.1 * torch.tanh(logits)).clamp(0, 1)
        return torch.sigmoid(logits)



class ToyFusionBackend(FusionBackend):
    """CPU contract check for LoRA_Fuse; not a DiT approximation."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, 4, 3, padding=1),
        )

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual, uncertainty = self.net(condition).split([3, 1], dim=1)
        final = (condition + 0.1 * torch.tanh(residual)).clamp(0, 1)
        return final, torch.sigmoid(uncertainty)
