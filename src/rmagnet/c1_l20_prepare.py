"""Build the deterministic C1-L20 Qwen/DoLP training cache.

This command performs no training. It runs the frozen base Qwen image-edit
backbone with every LoRA adapter disabled, captures image-stream features after
one-based block 20 for I and GT, and writes only the GT feature plus the derived
spatial weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .qwen_backend import MANIFEST as DOWNLOAD_MANIFEST
from .qwen_backend import QwenSharedBackend, check_snapshots
from .qwen_layer_probe import FeatureCapture, cosine_map, features
from .stage1_train import DEFAULT_DATA, discover_ids, image_tensor


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT / "data_cache/c1_l20"
BLOCK_INDEX = 19
BLOCK_NUMBER = BLOCK_INDEX + 1
FLOW_TIMESTEP = 499
TOKEN_GRID = (24, 32)  # height, width at 512x384 input resolution
EXPECTED_SIZE = (512, 384)  # width, height
FORMULA_VERSION = "c1-l20-qwen-majority-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_unit(values: np.ndarray, low_q: float, high_q: float) -> tuple[np.ndarray, float, float]:
    low, high = (float(x) for x in np.quantile(values, [low_q, high_q]))
    if not np.isfinite([low, high]).all():
        raise ValueError("Non-finite Qwen difference quantiles")
    if high <= low + 1e-12:
        return np.zeros_like(values, dtype=np.float32), low, high
    result = np.clip((values - low) / (high - low), 0.0, 1.0)
    return result.astype(np.float32), low, high


def resize_float(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(values).float()[None, None]
    resized = F.interpolate(tensor, size=(size[1], size[0]), mode="bilinear", align_corners=False)
    return resized[0, 0].numpy()


def gray(values01: np.ndarray, size: tuple[int, int] | None = None) -> Image.Image:
    array = np.round(np.clip(values01, 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(array, mode="L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return image


def heat(values01: np.ndarray, size: tuple[int, int] | None = None) -> Image.Image:
    return ImageOps.colorize(gray(values01, size), black="#000004", mid="#b5367a", white="#fcfdbf")


def labelled(image: Image.Image, text: str) -> Image.Image:
    bar = 28
    canvas = Image.new("RGB", (image.width, image.height + bar), "white")
    canvas.paste(image.convert("RGB"), (0, bar))
    ImageDraw.Draw(canvas).text((6, 7), text, fill="black", font=ImageFont.load_default())
    return canvas


def preview_panel(items: list[tuple[str, Image.Image]]) -> Image.Image:
    labelled_items = [labelled(image, title) for title, image in items]
    width = sum(image.width for image in labelled_items)
    result = Image.new("RGB", (width, max(image.height for image in labelled_items)), "white")
    x = 0
    for image in labelled_items:
        result.paste(image, (x, 0))
        x += image.width
    return result


def validate_dataset(root: Path, val_ids: set[str]) -> tuple[list[str], list[str]]:
    all_ids = discover_ids(root)
    role_sets: dict[str, set[str]] = {}
    for role in ("blended", "transmission_layer", "dolp"):
        folder = root / role
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        role_sets[role] = {path.stem for path in folder.glob("*.png")}
    expected = set(all_ids)
    for role, actual in role_sets.items():
        if actual != expected:
            raise RuntimeError(
                f"{role} IDs differ: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
            )
    missing_val = val_ids - expected
    if missing_val:
        raise RuntimeError(f"Validation IDs are absent: {sorted(missing_val)}")
    train_ids = [sample_id for sample_id in all_ids if sample_id not in val_ids]
    if not train_ids:
        raise RuntimeError("No training samples remain after excluding validation IDs")
    for sample_id in all_ids:
        sizes = []
        for role in ("blended", "transmission_layer", "dolp"):
            path = root / role / f"{sample_id}.png"
            with Image.open(path) as image:
                sizes.append(image.size)
                if role == "dolp" and image.mode != "L":
                    raise ValueError(f"DoLP must be 8-bit grayscale L: {path} has mode {image.mode}")
        if sizes != [EXPECTED_SIZE, EXPECTED_SIZE, EXPECTED_SIZE]:
            raise ValueError(f"Unexpected or unaligned sizes for {sample_id}: {sizes}")
    return all_ids, train_ids


def prompt_metadata(backend: QwenSharedBackend, lora_snapshot: Path) -> dict[str, object]:
    embedding_path = lora_snapshot / "text_embeddings/state_dict.safetensors"
    return {
        "kind": "fixed_precomputed_windowseat_text_embeddings",
        "source_prompt_text": None,
        "source_prompt_note": (
            "WindowSeat publishes fixed embeddings but not their recoverable source string; "
            "the embedding hash below is the exact prompt identity."
        ),
        "embedding_file": str(embedding_path),
        "embedding_sha256": sha256(embedding_path),
        "tensors": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in backend.embeddings.items()
        },
    }


def sample_paths(root: Path, sample_id: str) -> dict[str, Path]:
    return {
        "input": root / "blended" / f"{sample_id}.png",
        "gt": root / "transmission_layer" / f"{sample_id}.png",
        "dolp": root / "dolp" / f"{sample_id}.png",
    }


def existing_complete(output: Path) -> bool:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        return False
    for record in manifest.get("samples", []):
        for key in ("gt_feature", "weight"):
            if not (output / record["cache"][key]).is_file():
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--val-ids", default="11,12,17")
    parser.add_argument("--q-low-quantile", type=float, default=0.02)
    parser.add_argument("--q-high-quantile", type=float, default=0.98)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.q_low_quantile < args.q_high_quantile <= 1:
        raise ValueError("Qwen normalization quantiles must satisfy 0 <= low < high <= 1")
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    val_ids = {value for value in args.val_ids.split(",") if value}
    all_ids, train_ids = validate_dataset(args.data_root, val_ids)

    if existing_complete(args.output) and not args.overwrite:
        print(json.dumps({"status": "already_complete", "output": str(args.output)}, indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Refusing to mix with non-empty incomplete output: {args.output}; "
                "inspect it, then rerun with --overwrite"
            )
        import shutil
        shutil.rmtree(args.output)

    gt_dir = args.output / "gt_features"
    weight_dir = args.output / "weights"
    preview_dir = args.output / "previews"
    for folder in (gt_dir, weight_dir, preview_dir):
        folder.mkdir(parents=True, exist_ok=False)

    device = torch.device(args.device)
    backend = QwenSharedBackend.from_local(device)
    backend.set_trainable_branch(None)
    backend.transformer.eval()
    backend.vae.eval()
    if len(backend.transformer.transformer_blocks) != 60:
        raise RuntimeError("Expected the pinned Qwen backbone to contain 60 transformer blocks")
    if any(parameter.requires_grad for parameter in backend.transformer.parameters()):
        raise RuntimeError("The cache builder requires a fully frozen transformer")

    download_manifest = json.loads(DOWNLOAD_MANIFEST.read_text(encoding="utf-8"))
    model_entries = {entry["repo"]: entry for entry in download_manifest["models"]}
    qwen_entry = model_entries["Qwen/Qwen-Image-Edit-2509"]
    windowseat_entry = model_entries["huawei-bayerlab/windowseat-reflection-removal-v1-0"]
    _, lora_snapshot = check_snapshots()

    capture = FeatureCapture(backend.transformer, (BLOCK_INDEX,))
    records: list[dict[str, object]] = []
    try:
        for position, sample_id in enumerate(train_ids, start=1):
            paths = sample_paths(args.data_root, sample_id)
            input_tensor = image_tensor(paths["input"])[None]
            gt_tensor = image_tensor(paths["gt"])[None]

            q_input = features(backend, capture, input_tensor, (BLOCK_INDEX,))[BLOCK_INDEX]
            q_gt = features(backend, capture, gt_tensor, (BLOCK_INDEX,))[BLOCK_INDEX]
            if q_input.shape != q_gt.shape or q_gt.ndim != 3 or q_gt.shape[0] != 1:
                raise ValueError(f"Unexpected Q20 feature shapes for {sample_id}: {q_input.shape}, {q_gt.shape}")
            if q_gt.shape[1] != TOKEN_GRID[0] * TOKEN_GRID[1]:
                raise ValueError(f"Q20 token count changed for {sample_id}: {q_gt.shape}")

            dq_raw = cosine_map(q_input, q_gt, TOKEN_GRID)
            dq, q_low, q_high = robust_unit(dq_raw, args.q_low_quantile, args.q_high_quantile)
            with Image.open(paths["dolp"]) as image:
                dolp_pixel = np.asarray(image, dtype=np.float32) / 255.0
            dolp_token = resize_float(dolp_pixel, (TOKEN_GRID[1], TOKEN_GRID[0]))
            dolp_token = np.clip(dolp_token, 0.0, 1.0)

            score = dq * (0.7 + 0.3 * dolp_token)
            if not (np.all(score >= -1e-6) and np.all(score <= dq + 1e-6)):
                raise RuntimeError(f"DoLP multiplier invariant failed for {sample_id}")
            if not np.all(score >= 0.7 * dq - 1e-6):
                raise RuntimeError(f"DoLP lower multiplier invariant failed for {sample_id}")
            weight_raw_token = np.clip(1.0 + 2.0 * score, 1.0, 3.0)
            weight_token = weight_raw_token / float(weight_raw_token.mean())
            weight_pixel = resize_float(weight_token, EXPECTED_SIZE)
            weight_pixel = weight_pixel / float(weight_pixel.mean())

            arrays = (dq_raw, dq, dolp_token, score, weight_raw_token, weight_token, weight_pixel)
            if not all(np.isfinite(array).all() for array in arrays):
                raise ValueError(f"Non-finite cache values for {sample_id}")
            if not np.isclose(weight_token.mean(), 1.0, atol=2e-6):
                raise ValueError(f"Token weight mean is not one for {sample_id}")
            if not np.isclose(weight_pixel.mean(), 1.0, atol=2e-6):
                raise ValueError(f"Pixel weight mean is not one for {sample_id}")

            feature_rel = Path("gt_features") / f"{sample_id}.safetensors"
            weight_rel = Path("weights") / f"{sample_id}.npz"
            safetensors.torch.save_file(
                {"q20_gt": q_gt[0].to(torch.float16).contiguous()},
                args.output / feature_rel,
                metadata={"sample_id": sample_id, "block_one_based": str(BLOCK_NUMBER)},
            )
            np.savez_compressed(
                args.output / weight_rel,
                weight_pixel=weight_pixel.astype(np.float16),
                weight_token=weight_token.astype(np.float16),
            )

            dq_image = heat(dq, EXPECTED_SIZE)
            dolp_image = heat(dolp_pixel)
            score_image = heat(resize_float(score, EXPECTED_SIZE))
            # The final mean-one W is visualized with a fixed [0, 2] display interval.
            weight_image = heat(np.clip(weight_pixel / 2.0, 0, 1))
            dq_image.save(preview_dir / f"{sample_id}_dq.png")
            dolp_image.save(preview_dir / f"{sample_id}_dolp.png")
            score_image.save(preview_dir / f"{sample_id}_score.png")
            weight_image.save(preview_dir / f"{sample_id}_weight.png")
            preview_panel([
                (f"{sample_id} D_Q block 20", dq_image),
                ("DoLP / 255", dolp_image),
                ("S = D_Q*(0.7+0.3*DoLP)", score_image),
                (f"W mean=1 [{weight_pixel.min():.3f},{weight_pixel.max():.3f}]", weight_image),
            ]).save(preview_dir / f"{sample_id}_panel.png")

            record = {
                "id": sample_id,
                "source": {
                    role: {"path": str(path), "sha256": sha256(path)}
                    for role, path in paths.items()
                },
                "image_size_wh": list(EXPECTED_SIZE),
                "q20_feature_shape": list(q_gt[0].shape),
                "q_difference": {
                    "raw_min": float(dq_raw.min()),
                    "raw_max": float(dq_raw.max()),
                    "normalization_low": q_low,
                    "normalization_high": q_high,
                    "normalized_min": float(dq.min()),
                    "normalized_max": float(dq.max()),
                },
                "weight_stats": {
                    "score_min": float(score.min()),
                    "score_max": float(score.max()),
                    "raw_min": float(weight_raw_token.min()),
                    "raw_max": float(weight_raw_token.max()),
                    "token_mean_after_normalization": float(weight_token.mean()),
                    "pixel_min": float(weight_pixel.min()),
                    "pixel_max": float(weight_pixel.max()),
                    "pixel_mean": float(weight_pixel.mean()),
                },
                "cache": {
                    "gt_feature": str(feature_rel),
                    "gt_feature_sha256": sha256(args.output / feature_rel),
                    "weight": str(weight_rel),
                    "weight_sha256": sha256(args.output / weight_rel),
                    "preview_panel": str(Path("previews") / f"{sample_id}_panel.png"),
                },
            }
            records.append(record)
            print(json.dumps({"sample": sample_id, "progress": f"{position}/{len(train_ids)}", "weight_mean": record["weight_stats"]["pixel_mean"]}), flush=True)
            del q_input, q_gt, input_tensor, gt_tensor
            torch.cuda.empty_cache()
    finally:
        capture.close()

    manifest = {
        "schema_version": 1,
        "complete": True,
        "formula_version": FORMULA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_git_commit": subprocess.check_output(
            ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "data_root": str(args.data_root),
        "all_ids": all_ids,
        "train_ids": train_ids,
        "validation_ids_excluded": sorted(val_ids, key=int),
        "models": {
            "qwen_checkpoint": {"repo": qwen_entry["repo"], "revision": qwen_entry["revision"]},
            "windowseat_assets": {"repo": windowseat_entry["repo"], "revision": windowseat_entry["revision"]},
        },
        "qwen_feature": {
            "block_one_based": BLOCK_NUMBER,
            "block_zero_based_index": BLOCK_INDEX,
            "flow_timestep": FLOW_TIMESTEP,
            "token_grid_hw": list(TOKEN_GRID),
            "stream": "image-stream output after transformer block",
            "distance": "1 - cosine(Q20(I), Q20(GT))",
            "vae_encoding": "posterior mode; deterministic",
            "adapters": "all LoRA adapters disabled",
            "stored_gt_dtype": "float16",
            "input_feature_retention": "Q20(I) discarded after each sample",
        },
        "prompt": prompt_metadata(backend, lora_snapshot),
        "weighting": {
            "dq_normalization": (
                f"per-image robust min-max using raw cosine-distance q{args.q_low_quantile:.4f} "
                f"and q{args.q_high_quantile:.4f}, clipped to [0,1]"
            ),
            "dolp_normalization": "8-bit grayscale value / 255; no per-image contrast stretching",
            "score_formula": "S = D_Q * (0.7 + 0.3 * D_DoLP)",
            "raw_weight_formula": "W_raw = clip(1 + 2*S, 1, 3)",
            "final_weight_formula": "W = W_raw / mean(W_raw), then bilinear upsample and renormalize mean to 1",
            "important_range_note": (
                "[1,3] applies to W_raw. Exact mean-one normalization necessarily permits final W below 1."
            ),
            "stored_weight_dtype": "float16",
            "stored_arrays": ["weight_token[24,32]", "weight_pixel[384,512]"],
        },
        "samples": records,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "samples": len(records), "output": str(args.output), "manifest": str(manifest_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
