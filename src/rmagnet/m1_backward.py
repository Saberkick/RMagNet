"""Offline single-branch gradient/VRAM probe; does not optimize model weights."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .qwen_backend import QwenSharedBackend, ROOT


INPUT = ROOT / "datasets/own3/processed_1024/blended/11.png"
GT = ROOT / "datasets/own3/processed_1024/transmission_layer/11.png"


def load_crop(path: Path, side: int, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        w, h = rgb.size
        if side > min(w, h):
            raise ValueError(f"Crop {side} exceeds image {w}x{h}")
        x0, y0 = (w - side) // 2, (h - side) // 2
        crop = rgb.crop((x0, y0, x0 + side, y0 + side))
        array = np.array(crop, dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return (tensor.to(device) / 255 * 2 - 1).contiguous()


def probe(branch: str, side: int, rank: int, checkpointing: bool) -> dict:
    torch.manual_seed(2026)
    device = torch.device("cuda")
    backend = QwenSharedBackend.from_local(device, reflection_rank=rank)
    if checkpointing:
        backend.transformer.enable_gradient_checkpointing()
    backend.set_trainable_branch(branch)
    summary = backend.trainable_summary()
    if summary["trainable"] == 0:
        raise RuntimeError(f"No trainable LoRA parameters for {branch}")
    names = [
        name
        for name, parameter in backend.transformer.named_parameters()
        if parameter.requires_grad
    ]
    expected = ".default." if branch == "transmission" else ".reflection."
    if any(expected not in name or ".lora_" not in name for name in names):
        raise RuntimeError(f"Unexpected trainable parameters: {names[:5]}")
    image = load_crop(INPUT, side, device)
    target = load_crop(GT, side, device) if branch == "transmission" else torch.zeros_like(image)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    free_before, _ = torch.cuda.mem_get_info()
    started = time.monotonic()
    prediction = backend.forward_normalized(image, branch)
    loss = F.l1_loss(prediction.float(), target.float())
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    gradients = [
        parameter.grad
        for name, parameter in backend.transformer.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
        raise RuntimeError("Missing or nonfinite branch gradients")
    gradient_l1 = sum(g.abs().sum().item() for g in gradients)
    if gradient_l1 <= 0:
        raise RuntimeError("All branch gradients are zero")
    free_after, total = torch.cuda.mem_get_info()
    return {
        "branch": branch,
        "crop": side,
        "r_rank": rank,
        "gradient_checkpointing": checkpointing,
        "trainable_parameters": summary["trainable"],
        "trainable_tensor_count": len(names),
        "gradient_tensor_count": len(gradients),
        "gradient_l1": gradient_l1,
        "loss": loss.item(),
        "seconds_forward_backward": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "free_before_gib": free_before / 2**30,
        "free_after_gib": free_after / 2**30,
        "total_gib": total / 2**30,
        "scope": "one forward/backward, no optimizer state or optimizer step",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=["transmission", "reflection"], required=True)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--r-rank", type=int, default=8)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = probe(args.branch, args.crop, args.r_rank, args.checkpointing)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
