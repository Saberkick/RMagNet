"""Export all 60 frozen-Qwen I/GT image-token difference maps for M2 samples."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .m1b_train import sha256
from .m2a_prepare import dataset_state, sample_paths
from .qwen_backend import QwenSharedBackend
from .qwen_layer_probe import deterministic_encode
from .stage1_train import image_tensor


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path("/share/linmingheng-local/xuke/datasets/rmagnet_m2_aspect")
DEFAULT_OUTPUT = PROJECT / "runs/qwen_all_layer_probe_m2"
BLOCK_COUNT = 60
FLOW_TIMESTEP = 499
HIDDEN_SIZE = 3072
BUCKETS = (
    "extreme_portrait",
    "portrait",
    "near_square",
    "landscape",
    "extreme_landscape",
)


class FeatureCapture:
    def __init__(self, transformer: torch.nn.Module):
        self.values: dict[int, torch.Tensor] = {}
        self.handles = []
        for layer, block in enumerate(transformer.transformer_blocks):
            def hook(_module, _inputs, output, layer=layer):
                if not isinstance(output, tuple) or len(output) != 2:
                    raise RuntimeError(f"Unexpected block {layer + 1} output")
                self.values[layer] = output[1].detach().to(
                    device="cpu", dtype=torch.bfloat16
                )
            self.handles.append(block.register_forward_hook(hook))

    def clear(self) -> None:
        self.values.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def all_features(
    backend: QwenSharedBackend,
    capture: FeatureCapture,
    image: torch.Tensor,
) -> dict[int, torch.Tensor]:
    capture.clear()
    latent = deterministic_encode(backend, image)
    backend.transformer.disable_lora()
    try:
        backend.upstream.flow_step(
            latent, backend.transformer, backend.vae, backend.embeddings
        )
    finally:
        backend.transformer.enable_lora()
    if set(capture.values) != set(range(BLOCK_COUNT)):
        missing = sorted(set(range(BLOCK_COUNT)) - set(capture.values))
        raise RuntimeError(f"Missing Qwen blocks: {missing}")
    return dict(capture.values)


def cosine_map(first: torch.Tensor, second: torch.Tensor, grid: tuple[int, int]) -> np.ndarray:
    if first.shape != second.shape or first.ndim != 3 or first.shape[0] != 1:
        raise ValueError(f"Unexpected feature shapes: {first.shape}, {second.shape}")
    expected = grid[0] * grid[1]
    if first.shape[1:] != (expected, HIDDEN_SIZE):
        raise ValueError(
            f"Expected image tokens {(expected, HIDDEN_SIZE)}, got {tuple(first.shape[1:])}"
        )
    distance = 1.0 - (
        F.normalize(first.float(), dim=-1) * F.normalize(second.float(), dim=-1)
    ).sum(-1)
    return distance.reshape(*grid).numpy().astype(np.float32)


def robust_unit(values: np.ndarray, low_q: float = 0.02, high_q: float = 0.98) -> np.ndarray:
    low, high = np.quantile(values, [low_q, high_q])
    return np.clip((values - low) / max(float(high - low), 1e-8), 0.0, 1.0)


def colorize(values01: np.ndarray, size: tuple[int, int]) -> Image.Image:
    gray = Image.fromarray(np.round(values01 * 255).astype(np.uint8), mode="L")
    gray = gray.resize(size, Image.Resampling.BILINEAR)
    return ImageOps.colorize(
        gray, black="#000004", mid="#b5367a", white="#fcfdbf"
    )


def card(image: Image.Image, title: str, box: tuple[int, int] = (224, 224)) -> Image.Image:
    image = image.convert("RGB")
    fitted = ImageOps.contain(image, box, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (box[0] + 12, box[1] + 34), "white")
    x = (canvas.width - fitted.width) // 2
    y = 28 + (box[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    ImageDraw.Draw(canvas).text(
        (6, 7), title, fill="black", font=ImageFont.load_default()
    )
    return canvas


def grid_image(cards: list[Image.Image], columns: int) -> Image.Image:
    rows = (len(cards) + columns - 1) // columns
    cell_w = max(item.width for item in cards)
    cell_h = max(item.height for item in cards)
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), "#d8d8d8")
    for index, item in enumerate(cards):
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        canvas.paste(item, (x, y))
    return canvas


def choose_samples(records: list[dict], seed: int) -> list[dict]:
    generator = random.Random(seed)
    by_bucket: dict[str, list[dict]] = {}
    for record in records:
        by_bucket.setdefault(record["aspect_bucket"], []).append(record)
    selected = []
    for bucket in BUCKETS:
        members = sorted(by_bucket.get(bucket, []), key=lambda item: item["id"])
        if not members:
            raise RuntimeError(f"No training sample in aspect bucket {bucket}")
        selected.append(generator.choice(members))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = output / "raw_maps"
    image_dir = output / "images"
    raw_dir.mkdir()
    image_dir.mkdir()

    data_root = args.data_root.resolve()
    dataset_manifest, records = dataset_state(data_root)
    selected = choose_samples(records, args.seed)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    backend = QwenSharedBackend.from_local(device)
    backend.set_trainable_branch(None)
    backend.transformer.eval()
    backend.vae.eval()
    if len(backend.transformer.transformer_blocks) != BLOCK_COUNT:
        raise RuntimeError("Expected a 60-block Qwen transformer")
    if any(parameter.requires_grad for parameter in backend.transformer.parameters()):
        raise RuntimeError("Probe requires a fully frozen transformer")

    capture = FeatureCapture(backend.transformer)
    manifest_records = []
    stats_rows = []
    try:
        for position, record in enumerate(selected, start=1):
            sample_id = record["id"]
            width, height = record["target_size"]
            token_grid = (height // 16, width // 16)
            paths = sample_paths(data_root, sample_id)

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            f_input = all_features(
                backend, capture, image_tensor(paths["input"])[None]
            )
            f_gt = all_features(
                backend, capture, image_tensor(paths["gt"])[None]
            )
            torch.cuda.synchronize(device)

            maps = [
                cosine_map(f_input[layer], f_gt[layer], token_grid)
                for layer in range(BLOCK_COUNT)
            ]
            stacked = np.stack(maps)
            if not np.isfinite(stacked).all():
                raise ValueError(f"Non-finite feature difference for {sample_id}")
            shared_low, shared_high = (
                float(value) for value in np.quantile(stacked, [0.02, 0.98])
            )

            with Image.open(paths["input"]) as source:
                input_image = source.convert("RGB")
            with Image.open(paths["gt"]) as source:
                gt_image = source.convert("RGB")
            with Image.open(paths["dolp"]) as source:
                dolp_image = source.convert("L")
            if input_image.size != (width, height) or gt_image.size != input_image.size:
                raise ValueError(f"Unaligned M2 pair for {sample_id}")

            pixel_delta = np.abs(
                np.asarray(input_image, dtype=np.float32)
                - np.asarray(gt_image, dtype=np.float32)
            ).mean(axis=2)
            pixel_delta = robust_unit(pixel_delta)
            references = [
                card(input_image, f"{sample_id} input I"),
                card(gt_image, "GT"),
                card(colorize(pixel_delta, input_image.size), "pixel |I-GT| robust"),
                card(
                    ImageOps.colorize(dolp_image, black="black", white="white"),
                    "DoLP reference",
                ),
            ]
            grid_image(references, 4).save(
                output / f"{sample_id}_references.png", compress_level=6
            )

            shared_cards = []
            overlay_cards = []
            for layer, values in enumerate(maps, start=1):
                shared = np.clip(
                    (values - shared_low) / max(shared_high - shared_low, 1e-8),
                    0.0,
                    1.0,
                )
                per_layer = robust_unit(values)
                shared_heat = colorize(shared, input_image.size)
                local_heat = colorize(per_layer, input_image.size)
                overlay = Image.blend(input_image, local_heat, 0.55)
                shared_cards.append(card(shared_heat, f"block {layer:02d} shared"))
                overlay_cards.append(card(overlay, f"block {layer:02d} local overlay"))

                stats_rows.append(
                    {
                        "id": sample_id,
                        "aspect_bucket": record["aspect_bucket"],
                        "block": layer,
                        "min": float(values.min()),
                        "mean": float(values.mean()),
                        "p50": float(np.median(values)),
                        "p95": float(np.quantile(values, 0.95)),
                        "max": float(values.max()),
                    }
                )

            grid_image(shared_cards, 10).save(
                image_dir / f"{sample_id}_all60_shared_heat.png", compress_level=6
            )
            grid_image(overlay_cards, 10).save(
                image_dir / f"{sample_id}_all60_local_overlay.png", compress_level=6
            )
            np.savez_compressed(
                raw_dir / f"{sample_id}_all60.npz",
                difference=stacked,
                blocks=np.arange(1, BLOCK_COUNT + 1, dtype=np.int16),
                token_grid=np.asarray(token_grid, dtype=np.int16),
            )

            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            manifest_records.append(
                {
                    "id": sample_id,
                    "aspect_bucket": record["aspect_bucket"],
                    "pixel_size_wh": [width, height],
                    "token_grid_hw": list(token_grid),
                    "input": str(paths["input"]),
                    "gt": str(paths["gt"]),
                    "input_sha256": sha256(paths["input"]),
                    "gt_sha256": sha256(paths["gt"]),
                    "shared_display_p02": shared_low,
                    "shared_display_p98": shared_high,
                    "peak_allocated_gib": peak,
                }
            )
            print(
                json.dumps(
                    {
                        "sample": f"{position}/{len(selected)}",
                        "id": sample_id,
                        "bucket": record["aspect_bucket"],
                        "size": [width, height],
                        "peak_allocated_gib": round(peak, 3),
                    }
                ),
                flush=True,
            )
            del f_input, f_gt, maps, stacked
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        capture.close()

    with (output / "layer_stats.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats_rows[0]))
        writer.writeheader()
        writer.writerows(stats_rows)

    manifest = {
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "manual screening of all frozen Qwen blocks using I/GT token cosine difference",
        "dataset_version": dataset_manifest["version"],
        "data_root": str(data_root),
        "split": "train",
        "selection": "one deterministic sample from each of five aspect buckets",
        "seed": args.seed,
        "sample_ids": [record["id"] for record in selected],
        "qwen_blocks_one_based": list(range(1, BLOCK_COUNT + 1)),
        "feature": "image-stream output after each transformer block",
        "distance": "1-cosine(Q_block(I), Q_block(GT))",
        "adapter": "all LoRA adapters disabled; frozen base Qwen",
        "vae_encoding": "posterior mode; deterministic",
        "flow_timestep": FLOW_TIMESTEP,
        "visualization": {
            "shared_heat": "one p02/p98 scale shared by all 60 blocks within each sample; compare magnitude",
            "local_overlay": "each block independently p02/p98 normalized; compare spatial pattern only",
            "raw": "float32 unnormalized difference arrays",
        },
        "records": manifest_records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
