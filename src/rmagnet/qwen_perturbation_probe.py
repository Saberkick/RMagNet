"""Probe all frozen Qwen blocks with controlled image perturbations."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .m2a_prepare import dataset_state, sample_paths
from .qwen_all_layer_probe import (
    BLOCK_COUNT,
    FeatureCapture,
    all_features,
    card,
    colorize,
    grid_image,
    robust_unit,
)
from .qwen_backend import QwenSharedBackend
from .stage1_train import image_tensor


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path("/share/linmingheng-local/xuke/datasets/rmagnet_m2_aspect")
DEFAULT_OUTPUT = PROJECT / "runs/qwen_perturbation_probe"
DEFAULT_SAMPLE = "96_2492_938"
DEFAULT_DONOR = "69_2200_1512"
HIDDEN_SIZE = 3072


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).mul(2.0).sub(1.0)


def cosine_map(first: torch.Tensor, second: torch.Tensor, grid: tuple[int, int]) -> np.ndarray:
    expected = grid[0] * grid[1]
    if first.shape != second.shape or first.shape[1:] != (expected, HIDDEN_SIZE):
        raise ValueError(f"Unexpected feature shapes: {first.shape}, {second.shape}")
    distance = 1.0 - (
        F.normalize(first.float(), dim=-1) * F.normalize(second.float(), dim=-1)
    ).sum(-1)
    return distance.reshape(*grid).numpy().astype(np.float32)


def fractional_box(size: tuple[int, int], box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    width, height = size
    return (
        int(round(box[0] * width)),
        int(round(box[1] * height)),
        int(round(box[2] * width)),
        int(round(box[3] * height)),
    )


def local_blur(image: Image.Image) -> tuple[Image.Image, Image.Image, dict]:
    box = fractional_box(image.size, (0.48, 0.17, 0.80, 0.46))
    binary = Image.new("L", image.size, 0)
    ImageDraw.Draw(binary).rectangle(box, fill=255)
    blend_mask = binary.filter(ImageFilter.GaussianBlur(radius=7))
    blurred = image.filter(ImageFilter.GaussianBlur(radius=7))
    return Image.composite(blurred, image, blend_mask), binary, {
        "operation": "Gaussian blur radius 7 in a glass/textured patch",
        "box_xyxy": list(box),
    }


def stroke_delete(image: Image.Image) -> tuple[Image.Image, Image.Image, dict]:
    width, height = image.size
    source = np.asarray(image, dtype=np.uint8)
    sample_box = fractional_box(image.size, (0.36, 0.57, 0.62, 0.72))
    patch = source[sample_box[1]:sample_box[3], sample_box[0]:sample_box[2]]
    fill = tuple(int(value) for value in np.median(patch.reshape(-1, 3), axis=0))

    result = image.copy()
    mask = Image.new("L", image.size, 0)
    draw_result = ImageDraw.Draw(result)
    draw_mask = ImageDraw.Draw(mask)
    relative_rectangles = (
        (0.350, 0.802, 0.405, 0.812),
        (0.430, 0.800, 0.485, 0.810),
        (0.512, 0.803, 0.568, 0.813),
        (0.355, 0.852, 0.420, 0.862),
        (0.452, 0.850, 0.520, 0.860),
        (0.548, 0.851, 0.610, 0.861),
        (0.375, 0.905, 0.455, 0.913),
    )
    rectangles = []
    for values in relative_rectangles:
        rect = fractional_box(image.size, values)
        rectangles.append(list(rect))
        draw_result.rectangle(rect, fill=fill)
        draw_mask.rectangle(rect, fill=255)
    return result, mask, {
        "operation": "erase narrow horizontal segments across sign text",
        "fill_rgb": list(fill),
        "rectangles_xyxy": rectangles,
    }


def exposure_white_balance(image: Image.Image) -> tuple[Image.Image, Image.Image, dict]:
    bright = ImageEnhance.Brightness(image).enhance(1.06)
    array = np.asarray(bright, dtype=np.float32)
    gains = np.asarray([1.05, 1.00, 0.95], dtype=np.float32)
    array = np.clip(array * gains[None, None, :], 0, 255).astype(np.uint8)
    mask = Image.new("L", image.size, 255)
    return Image.fromarray(array, mode="RGB"), mask, {
        "operation": "global brightness 1.06 then RGB gains [1.05,1.00,0.95]",
        "brightness": 1.06,
        "rgb_gains": gains.tolist(),
    }


def object_replace(image: Image.Image, donor: Image.Image) -> tuple[Image.Image, Image.Image, dict]:
    target_box = fractional_box(image.size, (0.50, 0.16, 0.79, 0.46))
    donor_box = fractional_box(donor.size, (0.38, 0.58, 0.62, 0.96))
    crop = donor.crop(donor_box).resize(
        (target_box[2] - target_box[0], target_box[3] - target_box[1]),
        Image.Resampling.LANCZOS,
    )
    local_mask = Image.new("L", crop.size, 255).filter(ImageFilter.GaussianBlur(radius=5))
    result = image.copy()
    result.paste(crop, target_box[:2], local_mask)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle(target_box, fill=255)
    return result, mask, {
        "operation": "replace an upper glass region with a resized donor scene crop",
        "target_box_xyxy": list(target_box),
        "donor_box_xyxy": list(donor_box),
    }


def resize_mask(mask: Image.Image, grid: tuple[int, int]) -> np.ndarray:
    token = mask.resize((grid[1], grid[0]), Image.Resampling.BOX)
    return np.asarray(token, dtype=np.float32) / 255.0


def response_metrics(values: np.ndarray, mask: np.ndarray, global_change: bool) -> dict:
    eps = 1e-12
    flat = np.maximum(values.reshape(-1).astype(np.float64), 0.0)
    total = float(flat.sum())
    probability = flat / max(total, eps)
    entropy = float(
        -(probability * np.log(probability + eps)).sum()
        / max(np.log(len(probability)), eps)
    )
    threshold = float(np.quantile(values, 0.80))
    top = values >= threshold
    metrics = {
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "normalized_spatial_entropy": entropy,
        "area_above_half_max": float((values >= 0.5 * values.max()).mean()),
    }
    if global_change:
        metrics.update(
            {
                "inside_mean": float(values.mean()),
                "outside_mean": None,
                "inside_outside_ratio": None,
                "response_energy_inside_mask": 1.0,
                "top20_inside_fraction": 1.0,
            }
        )
        return metrics

    inside_weight = mask
    outside_weight = 1.0 - mask
    inside_mean = float((values * inside_weight).sum() / max(inside_weight.sum(), eps))
    outside_mean = float((values * outside_weight).sum() / max(outside_weight.sum(), eps))
    metrics.update(
        {
            "inside_mean": inside_mean,
            "outside_mean": outside_mean,
            "inside_outside_ratio": inside_mean / max(outside_mean, eps),
            "response_energy_inside_mask": float(
                (values * inside_weight).sum() / max(values.sum(), eps)
            ),
            "top20_inside_fraction": float(
                (top.astype(np.float32) * inside_weight).sum() / max(top.sum(), 1)
            ),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE)
    parser.add_argument("--donor-id", default=DEFAULT_DONOR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "raw_maps").mkdir()

    data_root = args.data_root.resolve()
    dataset_manifest, records = dataset_state(data_root)
    records_by_id = {record["id"]: record for record in records}
    if args.sample_id not in records_by_id or args.donor_id not in records_by_id:
        raise ValueError("Sample and donor must both belong to the corrected M2 train split")
    record = records_by_id[args.sample_id]
    paths = sample_paths(data_root, args.sample_id)
    donor_paths = sample_paths(data_root, args.donor_id)
    with Image.open(paths["input"]) as source:
        base = source.convert("RGB")
    with Image.open(donor_paths["input"]) as source:
        donor = source.convert("RGB")
    width, height = base.size
    token_grid = (height // 16, width // 16)

    perturbations = {
        "local_blur": local_blur(base),
        "stroke_delete": stroke_delete(base),
        "exposure_white_balance": exposure_white_balance(base),
        "object_replace": object_replace(base, donor),
    }

    reference_cards = [card(base, f"base {args.sample_id}")]
    for name, (image, mask, _metadata) in perturbations.items():
        difference = np.abs(
            np.asarray(image, dtype=np.float32) - np.asarray(base, dtype=np.float32)
        ).mean(axis=2)
        reference_cards.extend(
            [
                card(image, name),
                card(colorize(robust_unit(difference), image.size), f"{name} pixel delta"),
                card(mask.convert("RGB"), f"{name} mask"),
            ]
        )
    grid_image(reference_cards, 4).save(output / "PERTURBATION_REFERENCES.png")

    device = torch.device(args.device)
    torch.manual_seed(2026)
    torch.cuda.manual_seed(2026)
    backend = QwenSharedBackend.from_local(device)
    backend.set_trainable_branch(None)
    backend.transformer.eval()
    backend.vae.eval()
    if len(backend.transformer.transformer_blocks) != BLOCK_COUNT:
        raise RuntimeError("Expected a 60-block Qwen transformer")

    capture = FeatureCapture(backend.transformer)
    rows = []
    raw_payload = {}
    manifest_perturbations = {}
    try:
        base_features = all_features(backend, capture, image_to_tensor(base)[None])
        for name, (image, mask_image, metadata) in perturbations.items():
            altered_features = all_features(
                backend, capture, image_to_tensor(image)[None]
            )
            maps = np.stack(
                [
                    cosine_map(base_features[layer], altered_features[layer], token_grid)
                    for layer in range(BLOCK_COUNT)
                ]
            )
            if not np.isfinite(maps).all():
                raise ValueError(f"Non-finite response in {name}")
            raw_payload[name] = maps
            low, high = (float(x) for x in np.quantile(maps, [0.02, 0.98]))
            token_mask = resize_mask(mask_image, token_grid)
            global_change = name == "exposure_white_balance"

            shared_cards = []
            local_cards = []
            for layer, values in enumerate(maps, start=1):
                shared = np.clip((values - low) / max(high - low, 1e-8), 0, 1)
                local = robust_unit(values)
                heat_shared = colorize(shared, base.size)
                heat_local = colorize(local, base.size)
                overlay = Image.blend(base, heat_local, 0.55)
                shared_cards.append(card(heat_shared, f"block {layer:02d} shared"))
                local_cards.append(card(overlay, f"block {layer:02d} local overlay"))
                metrics = response_metrics(values, token_mask, global_change)
                rows.append({"perturbation": name, "block": layer, **metrics})

            grid_image(shared_cards, 10).save(
                output / "images" / f"{name}_all60_shared_heat.png"
            )
            grid_image(local_cards, 10).save(
                output / "images" / f"{name}_all60_local_overlay.png"
            )
            manifest_perturbations[name] = {
                **metadata,
                "shared_display_p02": low,
                "shared_display_p98": high,
                "mask_token_fraction": float(token_mask.mean()),
            }
            print(
                json.dumps(
                    {
                        "perturbation": name,
                        "complete": True,
                        "raw_mean_range": [float(maps.mean((1, 2)).min()), float(maps.mean((1, 2)).max())],
                    }
                ),
                flush=True,
            )
    finally:
        capture.close()

    np.savez_compressed(
        output / "raw_maps" / f"{args.sample_id}_perturbations_all60.npz",
        **raw_payload,
        blocks=np.arange(1, BLOCK_COUNT + 1, dtype=np.int16),
        token_grid=np.asarray(token_grid, dtype=np.int16),
    )
    fieldnames = list(rows[0])
    with (output / "layer_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_manifest["version"],
        "sample_id": args.sample_id,
        "donor_id": args.donor_id,
        "pixel_size_wh": [width, height],
        "token_grid_hw": list(token_grid),
        "qwen_blocks_one_based": list(range(1, BLOCK_COUNT + 1)),
        "model": "frozen base Qwen-Image-Edit-2509; all LoRA disabled",
        "vae": "posterior mode; deterministic",
        "flow_timestep": 499,
        "distance": "1-cosine(Q_block(base), Q_block(perturbed))",
        "perturbations": manifest_perturbations,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
