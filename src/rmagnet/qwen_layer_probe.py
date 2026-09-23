"""Export deterministic Qwen image-token difference maps for layer selection."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .m1b_train import PROJECT, sha256
from .qwen_backend import QwenSharedBackend
from .stage1_train import DEFAULT_DATA, discover_ids, image_tensor


DEFAULT_OUTPUT = PROJECT / "runs/qwen_layer_probe_seed2026"
LAYERS = (19, 29, 39)  # zero based; user-facing blocks 20, 30, 40 of 60


def deterministic_encode(backend: QwenSharedBackend, image: torch.Tensor) -> torch.Tensor:
    """Use the posterior mode so I/GT differences contain no sampling-noise difference."""
    vae = backend.vae
    image = image.to(device=vae.device, dtype=vae.dtype)
    distribution = vae.encode(image.unsqueeze(2)).latent_dist
    latent = distribution.mode()
    mean = torch.tensor(vae.config.latents_mean, device=latent.device, dtype=latent.dtype)
    std_inv = 1 / torch.tensor(vae.config.latents_std, device=latent.device, dtype=latent.dtype)
    return (latent - mean.view(1, vae.config.z_dim, 1, 1, 1)) * std_inv.view(1, vae.config.z_dim, 1, 1, 1)


class FeatureCapture:
    def __init__(self, transformer: torch.nn.Module, layers: tuple[int, ...]):
        self.values: dict[int, torch.Tensor] = {}
        self.handles = []
        for layer in layers:
            def hook(_module, _inputs, output, layer=layer):
                if not isinstance(output, tuple) or len(output) != 2:
                    raise RuntimeError(f"Unexpected block {layer} output")
                self.values[layer] = output[1].detach().float().cpu()
            self.handles.append(transformer.transformer_blocks[layer].register_forward_hook(hook))

    def clear(self) -> None:
        self.values.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def features(backend: QwenSharedBackend, capture: FeatureCapture,
             image: torch.Tensor, layers: tuple[int, ...]) -> dict[int, torch.Tensor]:
    capture.clear()
    latent = deterministic_encode(backend, image)
    if not hasattr(backend.transformer, "disable_lora") or not hasattr(backend.transformer, "enable_lora"):
        raise RuntimeError("Diffusers transformer does not expose LoRA on/off controls")
    backend.transformer.disable_lora()
    try:
        backend.upstream.flow_step(latent, backend.transformer, backend.vae, backend.embeddings)
    finally:
        backend.transformer.enable_lora()
    if set(capture.values) != set(layers):
        raise RuntimeError(f"Missing captured layers: {set(layers) - set(capture.values)}")
    return dict(capture.values)


def cosine_map(first: torch.Tensor, second: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    if first.shape != second.shape or first.ndim != 3 or first.shape[0] != 1:
        raise ValueError(f"Unexpected feature shapes: {first.shape}, {second.shape}")
    expected = shape[0] * shape[1]
    if first.shape[1] != expected:
        raise ValueError(f"Expected {expected} image tokens, got {first.shape[1]}")
    distance = 1 - (F.normalize(first, dim=-1) * F.normalize(second, dim=-1)).sum(-1)
    return distance.reshape(*shape).numpy().astype(np.float32)


def normalized(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((values - low) / max(high - low, 1e-8), 0, 1)


def colorize(values01: np.ndarray, size: tuple[int, int]) -> Image.Image:
    gray = Image.fromarray(np.round(values01 * 255).astype(np.uint8), mode="L")
    gray = gray.resize(size, Image.Resampling.BILINEAR)
    # A fixed black/purple/yellow scale; all three layers of a sample share limits.
    return ImageOps.colorize(gray, black="#000004", mid="#b5367a", white="#fcfdbf")


def label(image: Image.Image, title: str, height: int = 24) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + height), "white")
    canvas.paste(image, (0, height))
    ImageDraw.Draw(canvas).text((6, 5), title, fill="black", font=ImageFont.load_default())
    return canvas


def make_row(images: list[Image.Image]) -> Image.Image:
    width = sum(image.width for image in images)
    canvas = Image.new("RGB", (width, max(image.height for image in images)), "white")
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--val-ids", default="11,12,17")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    val_ids = args.val_ids.split(",")
    train_ids = [sample_id for sample_id in discover_ids(args.data_root) if sample_id not in val_ids]
    if len(train_ids) != 50 or args.count != 5:
        raise RuntimeError(f"Expected fixed 50-image pool and 5 samples, got {len(train_ids)}/{args.count}")
    sample_ids = random.Random(args.seed).sample(train_ids, args.count)
    args.output.mkdir(parents=True)
    raw_dir = args.output / "raw_maps"
    image_dir = args.output / "images"
    raw_dir.mkdir()
    image_dir.mkdir()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    backend = QwenSharedBackend.from_local(device)
    backend.set_trainable_branch(None)
    backend.transformer.eval()
    backend.vae.eval()
    if len(backend.transformer.transformer_blocks) != 60:
        raise RuntimeError("Expected a 60-block Qwen transformer")
    capture = FeatureCapture(backend.transformer, LAYERS)
    records = []
    summary_rows = []
    try:
        for sample_id in sample_ids:
            input_path = args.data_root / "blended" / f"{sample_id}.png"
            gt_path = args.data_root / "transmission_layer" / f"{sample_id}.png"
            dolp_path = args.data_root / "dolp" / f"{sample_id}.png"
            inp = image_tensor(input_path)[None]
            gt = image_tensor(gt_path)[None]
            f_input = features(backend, capture, inp, LAYERS)
            f_gt = features(backend, capture, gt, LAYERS)
            maps = {layer: cosine_map(f_input[layer], f_gt[layer], (24, 32)) for layer in LAYERS}
            stacked = np.stack(list(maps.values()))
            low, high = (float(x) for x in np.quantile(stacked, [0.02, 0.98]))
            input_image = Image.open(input_path).convert("RGB")
            gt_image = Image.open(gt_path).convert("RGB")
            dolp_image = Image.open(dolp_path).convert("L")
            if input_image.size != (512, 384) or gt_image.size != input_image.size or dolp_image.size != input_image.size:
                raise ValueError(f"Unaligned visualization inputs for {sample_id}")
            rgb_diff = np.abs(np.asarray(input_image, dtype=np.int16) - np.asarray(gt_image, dtype=np.int16)).mean(2)
            diff_image = colorize(np.clip(rgb_diff / max(float(np.quantile(rgb_diff, .98)), 1), 0, 1), input_image.size)
            details = [label(input_image, f"{sample_id} input I"), label(gt_image, "GT"),
                       label(ImageOps.colorize(dolp_image, black="black", white="white"), "DoLP"),
                       label(diff_image, "pixel |I-GT|")]
            overlays = []
            stats = {}
            for layer, values in maps.items():
                display = normalized(values, low, high)
                heat = colorize(display, input_image.size)
                overlay = Image.blend(input_image, heat, .55)
                user_layer = layer + 1
                np.save(raw_dir / f"{sample_id}_layer{user_layer:02d}.npy", values, allow_pickle=False)
                heat.save(image_dir / f"{sample_id}_layer{user_layer:02d}_heat.png")
                overlay.save(image_dir / f"{sample_id}_layer{user_layer:02d}_overlay.png")
                details.append(label(heat, f"Qwen block {user_layer}/60 heat"))
                overlays.append(label(overlay, f"{sample_id} block {user_layer}/60 overlay"))
                stats[str(user_layer)] = {"min": float(values.min()), "mean": float(values.mean()),
                                          "p50": float(np.median(values)), "p95": float(np.quantile(values, .95)),
                                          "max": float(values.max())}
            details.append(label(Image.new("RGB", input_image.size, "white"), f"shared scale p02={low:.5f} p98={high:.5f}"))
            make_row(details[:4]).save(args.output / f"{sample_id}_references.png")
            make_row(details[4:]).save(args.output / f"{sample_id}_layer_heats.png")
            summary_rows.append(make_row([label(input_image, f"{sample_id} input"), label(gt_image, "GT"), *overlays]))
            records.append({"id": sample_id, "input_sha256": sha256(input_path), "gt_sha256": sha256(gt_path),
                            "dolp_sha256": sha256(dolp_path), "shared_display_p02": low,
                            "shared_display_p98": high, "layers": stats})
            print(json.dumps(records[-1]), flush=True)
    finally:
        capture.close()
    summary = Image.new("RGB", (max(row.width for row in summary_rows), sum(row.height for row in summary_rows)), "white")
    y = 0
    for row in summary_rows:
        summary.paste(row, (0, y))
        y += row.height
    summary.save(args.output / "SUMMARY_5x3_OVERLAYS.png")
    manifest = {"seed": args.seed, "sample_pool": train_ids, "sample_ids_in_display_order": sample_ids,
                "validation_ids_excluded": val_ids, "qwen_blocks_one_based": [layer + 1 for layer in LAYERS],
                "feature": "image-stream output after block; cosine distance between I and GT",
                "vae_encoding": "posterior mode (deterministic; no sampling noise)",
                "adapter": "all LoRA adapters disabled; frozen base Qwen",
                "flow_timestep": 499, "token_grid": [24, 32],
                "display_normalization": "per sample, one shared p02/p98 across all three layers; raw float32 maps retained",
                "records": records}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "sample_ids": sample_ids}, indent=2), flush=True)


if __name__ == "__main__":
    main()
