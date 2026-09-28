"""Compare used Qwen layers with all LoRA disabled versus M4-best LoRA enabled."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.rmagnet.m1b_train import load_initial
from src.rmagnet.m2a_prepare import dataset_state, sample_paths
from src.rmagnet.m4_cache import ALL_BLOCKS, MAX_BLOCK, StopAfterSelectedBlocks, sha256
from src.rmagnet.qwen_backend import ADAPTER_NAMES, QwenSharedBackend
from src.rmagnet.qwen_layer_probe import deterministic_encode
from src.rmagnet.stage1_train import image_tensor


ROOT = Path("/share/linmingheng-local/xuke")
PROJECT = ROOT / "RMagNet"
DEFAULT_DATA = ROOT / "datasets/rmagnet_m2_aspect"
DEFAULT_ADAPTER = PROJECT / "runs/m4_e30_p4/best_transmission_lora.safetensors"
DEFAULT_OUTPUT = PROJECT / "runs/m4_best_layer_change"
BUCKETS = ("extreme_portrait", "portrait", "near_square", "landscape", "extreme_landscape")
HIDDEN_SIZE = 3072


@torch.inference_mode()
def features(backend: QwenSharedBackend, image: torch.Tensor, adapted: bool) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for block_number in ALL_BLOCKS:
        def hook(_module, _inputs, output, block_number=block_number):
            captured[block_number] = output[1].detach().cpu().to(torch.bfloat16)
            if block_number == MAX_BLOCK:
                raise StopAfterSelectedBlocks
        handles.append(backend.transformer.transformer_blocks[block_number - 1].register_forward_hook(hook))
    if adapted:
        backend.transformer.enable_lora()
        backend.transformer.set_adapter(ADAPTER_NAMES["transmission"])
    else:
        backend.transformer.disable_lora()
    try:
        latent = deterministic_encode(backend, image)
        try:
            backend.upstream.flow_step(latent, backend.transformer, backend.vae, backend.embeddings)
        except StopAfterSelectedBlocks:
            pass
    finally:
        backend.transformer.enable_lora()
        backend.transformer.set_adapter(ADAPTER_NAMES["transmission"])
        for handle in handles:
            handle.remove()
    if set(captured) != set(ALL_BLOCKS):
        raise RuntimeError(f"Missing selected blocks: {set(ALL_BLOCKS) - set(captured)}")
    return captured


def cosine_map(first: torch.Tensor, second: torch.Tensor, grid: tuple[int, int]) -> np.ndarray:
    expected = grid[0] * grid[1]
    if first.shape != second.shape or tuple(first.shape) != (1, expected, HIDDEN_SIZE):
        raise ValueError(f"Unexpected feature shape: {first.shape}, {second.shape}, grid={grid}")
    value = 1.0 - (F.normalize(first.float(), dim=-1) * F.normalize(second.float(), dim=-1)).sum(-1)
    return value.reshape(*grid).numpy().astype(np.float32)


def robust_pair(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joined = np.concatenate([first.reshape(-1), second.reshape(-1)])
    low, high = np.quantile(joined, [0.02, 0.98])
    scale = max(float(high - low), 1e-8)
    a = np.clip((first - low) / scale, 0, 1)
    b = np.clip((second - low) / scale, 0, 1)
    delta = np.abs(b - a)
    return a, b, delta


def colorize(values: np.ndarray, size: tuple[int, int]) -> Image.Image:
    gray = Image.fromarray(np.round(values * 255).astype(np.uint8), mode="L").resize(size, Image.Resampling.BILINEAR)
    return ImageOps.colorize(gray, black="#000004", mid="#b5367a", white="#fcfdbf")


def card(image: Image.Image, title: str, box: tuple[int, int] = (240, 190)) -> Image.Image:
    fitted = ImageOps.contain(image.convert("RGB"), box, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (box[0] + 12, box[1] + 34), "white")
    canvas.paste(fitted, ((canvas.width - fitted.width) // 2, 28 + (box[1] - fitted.height) // 2))
    ImageDraw.Draw(canvas).text((6, 7), title, fill="black", font=ImageFont.load_default())
    return canvas


def grid(cards: list[Image.Image], columns: int) -> Image.Image:
    rows = (len(cards) + columns - 1) // columns
    width, height = max(x.width for x in cards), max(x.height for x in cards)
    canvas = Image.new("RGB", (columns * width, rows * height), "#d8d8d8")
    for index, item in enumerate(cards):
        canvas.paste(item, ((index % columns) * width, (index // columns) * height))
    return canvas


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    a, b = first.reshape(-1).astype(np.float64), second.reshape(-1).astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def select(records: list[dict], train_ids: set[str], seed: int) -> list[dict]:
    rng = random.Random(seed)
    chosen = []
    for bucket in BUCKETS:
        candidates = sorted((x for x in records if x["id"] in train_ids and x["aspect_bucket"] == bucket), key=lambda x: x["id"])
        if not candidates:
            raise RuntimeError(f"No train sample for bucket {bucket}")
        chosen.append(rng.choice(candidates))
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    _manifest, records = dataset_state(args.data_root)
    train_ids = set((args.data_root / "splits/train.txt").read_text(encoding="utf-8").split())
    chosen = select(records, train_ids, args.seed)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    backend = QwenSharedBackend.from_local(device)
    backend.set_trainable_branch("transmission")
    load_initial(backend, args.adapter, device)
    backend.set_trainable_branch(None)
    backend.transformer.eval()
    backend.vae.eval()

    rows: list[dict] = []
    try:
        for number, record in enumerate(chosen, start=1):
            sample_id = record["id"]
            width, height = record["target_size"]
            token_grid = (height // 16, width // 16)
            paths = sample_paths(args.data_root, sample_id)
            with Image.open(paths["input"]) as stream:
                input_image = stream.convert("RGB")
            with Image.open(paths["gt"]) as stream:
                gt_image = stream.convert("RGB")
            gc.collect()
            torch.cuda.empty_cache()
            base_i = features(backend, image_tensor(paths["input"])[None], adapted=False)
            base_gt = features(backend, image_tensor(paths["gt"])[None], adapted=False)
            best_i = features(backend, image_tensor(paths["input"])[None], adapted=True)
            best_gt = features(backend, image_tensor(paths["gt"])[None], adapted=True)

            cards = [card(input_image, f"{sample_id} input"), card(gt_image, "GT"), card(Image.new("RGB", input_image.size, "white"), "columns: base / M4-best / abs delta")]
            for block in ALL_BLOCKS:
                before = cosine_map(base_i[block], base_gt[block], token_grid)
                after = cosine_map(best_i[block], best_gt[block], token_grid)
                before_view, after_view, delta_view = robust_pair(before, after)
                cards.extend([
                    card(colorize(before_view, input_image.size), f"Q{block} base I-GT"),
                    card(colorize(after_view, input_image.size), f"Q{block} M4-best I-GT"),
                    card(colorize(delta_view, input_image.size), f"Q{block} normalized |delta|"),
                ])
                rows.append({
                    "id": sample_id,
                    "aspect_bucket": record["aspect_bucket"],
                    "block": block,
                    "base_mean": float(before.mean()),
                    "m4_best_mean": float(after.mean()),
                    "mean_change_percent": 100.0 * (float(after.mean()) / max(float(before.mean()), 1e-12) - 1.0),
                    "raw_mae": float(np.abs(after - before).mean()),
                    "spatial_correlation": correlation(before, after),
                })
            grid(cards, 3).save(args.output / f"{sample_id}_used_layers_base_vs_m4best.png", compress_level=6)
            print(json.dumps({"sample": f"{number}/{len(chosen)}", "id": sample_id, "bucket": record["aspect_bucket"]}), flush=True)
            del base_i, base_gt, best_i, best_gt
    finally:
        del backend
        gc.collect()
        torch.cuda.empty_cache()

    with (args.output / "layer_change.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_block = {}
    for block in ALL_BLOCKS:
        values = [row for row in rows if row["block"] == block]
        by_block[str(block)] = {
            "mean_change_percent": float(np.mean([x["mean_change_percent"] for x in values])),
            "raw_mae": float(np.mean([x["raw_mae"] for x in values])),
            "spatial_correlation": float(np.mean([x["spatial_correlation"] for x in values])),
        }
    report = {
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "adapter": str(args.adapter),
        "adapter_sha256": sha256(args.adapter),
        "comparison": "Q_l(I) versus Q_l(GT), all LoRA disabled compared with M4-best Transmission LoRA enabled",
        "sample_ids": [x["id"] for x in chosen],
        "blocks": list(ALL_BLOCKS),
        "aggregate_by_block": by_block,
    }
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# M4-best 对所用 Qwen 层的影响", "", f"- 权重 SHA-256：`{report['adapter_sha256']}`", f"- 固定训练样本：`{', '.join(report['sample_ids'])}`", "- 比较量：同一层 `1-cos(Q_l(I), Q_l(GT))`，对比关闭全部 LoRA 与启用 M4-best LoRA。", "- `spatial_correlation` 越接近 1，热点位置越稳定；`mean_change_percent` 表示原始差异强度变化。", "", "| Block | 差异均值变化 | 原始图 MAE | 空间相关性 |", "|---:|---:|---:|---:|"]
    for block in ALL_BLOCKS:
        value = by_block[str(block)]
        lines.append(f"| {block} | {value['mean_change_percent']:+.2f}% | {value['raw_mae']:.6f} | {value['spatial_correlation']:.4f} |")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output), "aggregate_by_block": by_block}, indent=2), flush=True)


if __name__ == "__main__":
    main()
