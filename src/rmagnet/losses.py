import torch
import torch.nn.functional as F

from .contracts import ModelOutput


def _masked_l1(
    prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    expanded = weight.expand_as(prediction)
    return ((prediction - target).abs() * expanded).sum() / expanded.sum().clamp_min(1)


def synthetic_losses(
    output: ModelOutput,
    image: torch.Tensor,
    clean: torch.Tensor,
    reflection: torch.Tensor,
    interface_mask: torch.Tensor,
    edit_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Illustrative synthetic supervision for scaffold checks, not tuned weights."""
    parts = {
        "final": F.l1_loss(output.transmission, clean),
        "transmission": F.l1_loss(output.transmission_candidate, clean),
        "reflection": F.l1_loss(output.reflection_candidate, reflection),
        "interface": F.binary_cross_entropy(output.maps.interface, interface_mask),
        "edit": F.binary_cross_entropy(output.maps.edit, edit_mask),
        "keep": _masked_l1(output.transmission, image, 1 - edit_mask),
    }
    parts["total"] = (
        parts["final"]
        + 0.5 * parts["transmission"]
        + 0.25 * parts["reflection"]
        + 0.1 * parts["interface"]
        + 0.1 * parts["edit"]
        + 0.2 * parts["keep"]
    )
    return parts

