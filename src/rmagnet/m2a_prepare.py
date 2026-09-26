"""Build dynamic-shape Q20 supervision cache for the M2 training split.

The frozen Qwen backbone is stopped immediately after block 20.  Only one image
is resident at a time, and captured features are moved to CPU as BF16 before the
next forward.  Progress is committed after each sample for safe resume.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
from PIL import Image

from .c1_l20_prepare import (
    heat,
    preview_panel,
    resize_float,
    robust_unit,
    sha256,
)
from .qwen_backend import MANIFEST as DOWNLOAD_MANIFEST
from .qwen_backend import QwenSharedBackend, check_snapshots
from .qwen_layer_probe import cosine_map, deterministic_encode
from .stage1_train import image_tensor


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = Path("/share/linmingheng-local/xuke/datasets/rmagnet_m2_aspect")
DEFAULT_OUTPUT = PROJECT / "data_cache/m2a_q20"
BLOCK_INDEX = 19
BLOCK_NUMBER = 20
FLOW_TIMESTEP = 499
HIDDEN_SIZE = 3072
FORMULA_VERSION = "m2a-q20-variable-aspect-v1"


class StopAfterQ20(Exception):
    """Internal early exit after the required Qwen block has been captured."""


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def dataset_state(data_root: Path) -> tuple[dict, list[dict]]:
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("version") != "m2-variable-aspect-v2-corrected-labels":
        raise RuntimeError("M2 processed dataset manifest is incomplete or incompatible")
    train_ids = (data_root / "splits/train.txt").read_text(encoding="utf-8").split()
    samples_by_id = {record["id"]: record for record in manifest["samples"]}
    if len(train_ids) != 144 or len(set(train_ids)) != len(train_ids):
        raise RuntimeError(f"Expected 144 unique M2 train IDs, got {len(train_ids)}")
    if not set(train_ids) <= set(samples_by_id):
        raise RuntimeError("Train split contains IDs absent from the M2 manifest")
    train_records = [samples_by_id[sample_id] for sample_id in train_ids]
    for record in train_records:
        width, height = record["target_size"]
        if width % 16 or height % 16:
            raise ValueError(f"M2 size is not divisible by 16: {record['id']} {width}x{height}")
        paths = sample_paths(data_root, record["id"])
        sizes = []
        for role, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                sizes.append(image.size)
                expected_mode = "L" if role == "dolp" else "RGB"
                if image.mode != expected_mode:
                    raise ValueError(f"Wrong mode for {record['id']}/{role}: {image.mode}")
        if sizes != [(width, height)] * 3:
            raise ValueError(f"Unaligned M2 cache sources for {record['id']}: {sizes}")
    return manifest, train_records


def representative_ids(records: list[dict]) -> set[str]:
    by_bucket: dict[str, list[dict]] = {}
    for record in records:
        by_bucket.setdefault(record["aspect_bucket"], []).append(record)
    selected = set()
    for bucket, members in by_bucket.items():
        ordered = sorted(members, key=lambda item: item["source_aspect_ratio"])
        selected.add(ordered[len(ordered) // 2]["id"])
    if len(selected) != 7:
        raise RuntimeError(f"Expected seven M2 aspect buckets, selected {len(selected)}")
    return selected


def sample_paths(root: Path, sample_id: str) -> dict[str, Path]:
    return {
        "input": root / "blended" / f"{sample_id}.png",
        "gt": root / "transmission_layer" / f"{sample_id}.png",
        "dolp": root / "dolp" / f"{sample_id}.png",
    }


@torch.inference_mode()
def q20_feature(backend: QwenSharedBackend, image: torch.Tensor) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def capture_and_stop(_module, _inputs, output):
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("Unexpected Qwen block-20 output")
        captured["feature"] = output[1].detach().to(device="cpu", dtype=torch.bfloat16)
        raise StopAfterQ20

    handle = backend.transformer.transformer_blocks[BLOCK_INDEX].register_forward_hook(
        capture_and_stop
    )
    backend.transformer.disable_lora()
    try:
        latent = deterministic_encode(backend, image)
        try:
            backend.upstream.flow_step(latent, backend.transformer, backend.vae, backend.embeddings)
        except StopAfterQ20:
            pass
        else:
            raise RuntimeError("Qwen forward reached the end without the Q20 stop hook")
    finally:
        backend.transformer.enable_lora()
        handle.remove()
    if "feature" not in captured:
        raise RuntimeError("Q20 feature was not captured")
    return captured["feature"]


def make_preview(
    output: Path,
    sample_id: str,
    pixel_size: tuple[int, int],
    dq: np.ndarray,
    dolp_pixel: np.ndarray,
    score: np.ndarray,
    weight_pixel: np.ndarray,
) -> str:
    dq_image = heat(dq, pixel_size)
    dolp_image = heat(dolp_pixel)
    score_image = heat(resize_float(score, pixel_size))
    weight_image = heat(np.clip(weight_pixel / 2.0, 0, 1))
    relative = Path("previews") / f"{sample_id}_panel.png"
    preview_panel(
        [
            (f"{sample_id} D_Q block 20", dq_image),
            ("DoLP / 255", dolp_image),
            ("S = D_Q*(0.7+0.3*DoLP)", score_image),
            (f"W mean=1 [{weight_pixel.min():.3f},{weight_pixel.max():.3f}]", weight_image),
        ]
    ).save(output / relative, compress_level=6)
    return str(relative)


def current_git_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
    ).strip()


def progress_identity(
    data_root: Path,
    data_manifest_sha: str,
    selected_ids: list[str],
    args: argparse.Namespace,
) -> dict:
    return {
        "formula_version": FORMULA_VERSION,
        "data_root": str(data_root),
        "data_manifest_sha256": data_manifest_sha,
        "selected_ids": selected_ids,
        "selection": "seven-bucket-gate" if args.gate_only else "full-train-split",
        "q_low_quantile": args.q_low_quantile,
        "q_high_quantile": args.q_high_quantile,
        "block_zero_based_index": BLOCK_INDEX,
        "flow_timestep": FLOW_TIMESTEP,
    }


def load_or_initialize_progress(output: Path, identity: dict) -> dict:
    path = output / "progress.json"
    if path.is_file():
        progress = json.loads(path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise RuntimeError("Existing M2a progress belongs to another cache configuration")
        return progress
    progress = {
        "complete": False,
        "identity": identity,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": [],
    }
    atomic_json(path, progress)
    return progress


def verify_completed_record(output: Path, record: dict) -> None:
    feature_path = output / record["cache"]["gt_feature"]
    weight_path = output / record["cache"]["weight"]
    if not feature_path.is_file() or sha256(feature_path) != record["cache"]["gt_feature_sha256"]:
        raise RuntimeError(f"Broken resumable feature cache: {record['id']}")
    if not weight_path.is_file() or sha256(weight_path) != record["cache"]["weight_sha256"]:
        raise RuntimeError(f"Broken resumable weight cache: {record['id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--q-low-quantile", type=float, default=0.02)
    parser.add_argument("--q-high-quantile", type=float, default=0.98)
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.q_low_quantile < args.q_high_quantile <= 1:
        raise ValueError("Qwen normalization quantiles must satisfy 0 <= low < high <= 1")

    data_root = args.data_root.resolve()
    output = args.output.resolve()
    data_manifest, train_records = dataset_state(data_root)
    preview_ids = representative_ids(train_records)
    selected_records = (
        [record for record in train_records if record["id"] in preview_ids]
        if args.gate_only
        else train_records
    )
    selected_ids = [record["id"] for record in selected_records]
    identity = progress_identity(
        data_root, sha256(data_root / "manifest.json"), selected_ids, args
    )

    complete_path = output / "manifest.json"
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("complete") and complete.get("identity") == identity:
            print(json.dumps({"status": "already_complete", "output": str(output)}, indent=2))
            return
        raise RuntimeError(f"Existing manifest is incompatible: {complete_path}")

    output.mkdir(parents=True, exist_ok=True)
    for folder in ("gt_features", "weights", "previews"):
        (output / folder).mkdir(exist_ok=True)
    progress = load_or_initialize_progress(output, identity)
    completed = {record["id"] for record in progress["records"]}
    if len(completed) != len(progress["records"]):
        raise RuntimeError("Duplicate IDs in M2a progress")
    for record in progress["records"]:
        verify_completed_record(output, record)

    remaining = [record for record in selected_records if record["id"] not in completed]
    print(
        json.dumps(
            {
                "status": "resume" if completed else "start",
                "selected": len(selected_records),
                "completed": len(completed),
                "remaining": len(remaining),
                "device": args.device,
            },
            indent=2,
        ),
        flush=True,
    )

    device = torch.device(args.device)
    torch.manual_seed(2026)
    torch.cuda.manual_seed(2026)
    backend = QwenSharedBackend.from_local(device)
    backend.set_trainable_branch(None)
    backend.transformer.eval()
    backend.vae.eval()
    if len(backend.transformer.transformer_blocks) != 60:
        raise RuntimeError("Expected the pinned Qwen backbone to contain 60 blocks")
    if any(parameter.requires_grad for parameter in backend.transformer.parameters()):
        raise RuntimeError("M2a requires a fully frozen transformer")

    download_manifest = json.loads(DOWNLOAD_MANIFEST.read_text(encoding="utf-8"))
    model_entries = {entry["repo"]: entry for entry in download_manifest["models"]}
    qwen_entry = model_entries["Qwen/Qwen-Image-Edit-2509"]
    windowseat_entry = model_entries["huawei-bayerlab/windowseat-reflection-removal-v1-0"]
    _, lora_snapshot = check_snapshots()

    try:
        for position, source_record in enumerate(remaining, start=len(completed) + 1):
            sample_id = source_record["id"]
            width, height = source_record["target_size"]
            token_grid = (height // 16, width // 16)
            paths = sample_paths(data_root, sample_id)

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            input_tensor = image_tensor(paths["input"])[None]
            q_input = q20_feature(backend, input_tensor)
            del input_tensor
            torch.cuda.empty_cache()
            gt_tensor = image_tensor(paths["gt"])[None]
            q_gt = q20_feature(backend, gt_tensor)
            del gt_tensor
            torch.cuda.synchronize(device)

            expected_tokens = token_grid[0] * token_grid[1]
            expected_shape = (1, expected_tokens, HIDDEN_SIZE)
            if tuple(q_input.shape) != expected_shape or tuple(q_gt.shape) != expected_shape:
                raise ValueError(
                    f"Unexpected Q20 shape for {sample_id}: I={tuple(q_input.shape)}, "
                    f"GT={tuple(q_gt.shape)}, expected={expected_shape}"
                )

            dq_raw = cosine_map(q_input.float(), q_gt.float(), token_grid)
            dq, q_low, q_high = robust_unit(
                dq_raw, args.q_low_quantile, args.q_high_quantile
            )
            with Image.open(paths["dolp"]) as image:
                dolp_pixel = np.asarray(image, dtype=np.float32) / 255.0
            dolp_token = np.clip(
                resize_float(dolp_pixel, (token_grid[1], token_grid[0])), 0.0, 1.0
            )
            score = dq * (0.7 + 0.3 * dolp_token)
            if not np.all(score >= 0.7 * dq - 1e-6) or not np.all(score <= dq + 1e-6):
                raise RuntimeError(f"DoLP multiplier invariant failed for {sample_id}")
            raw_weight = np.clip(1.0 + 2.0 * score, 1.0, 3.0)
            weight_token = raw_weight / float(raw_weight.mean())
            weight_pixel = resize_float(weight_token, (width, height))
            weight_pixel = weight_pixel / float(weight_pixel.mean())
            arrays = (dq_raw, dq, dolp_token, score, raw_weight, weight_token, weight_pixel)
            if not all(np.isfinite(array).all() for array in arrays):
                raise ValueError(f"Non-finite M2a cache array for {sample_id}")
            if not np.isclose(weight_token.mean(), 1.0, atol=2e-6):
                raise ValueError(f"Token weight mean differs from one: {sample_id}")
            if not np.isclose(weight_pixel.mean(), 1.0, atol=2e-6):
                raise ValueError(f"Pixel weight mean differs from one: {sample_id}")

            feature_rel = Path("gt_features") / f"{sample_id}.safetensors"
            weight_rel = Path("weights") / f"{sample_id}.npz"
            feature_tmp = output / feature_rel.with_suffix(".safetensors.tmp")
            weight_tmp = output / weight_rel.with_suffix(".npz.tmp")
            safetensors.torch.save_file(
                {"q20_gt": q_gt[0].contiguous()},
                feature_tmp,
                metadata={
                    "sample_id": sample_id,
                    "block_one_based": str(BLOCK_NUMBER),
                    "dtype": "bfloat16",
                    "token_grid_hw": f"{token_grid[0]},{token_grid[1]}",
                },
            )
            with weight_tmp.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    weight_pixel=weight_pixel.astype(np.float16),
                    weight_token=weight_token.astype(np.float16),
                )
            os.replace(feature_tmp, output / feature_rel)
            os.replace(weight_tmp, output / weight_rel)

            preview_rel = None
            if sample_id in preview_ids:
                preview_rel = make_preview(
                    output, sample_id, (width, height), dq, dolp_pixel, score, weight_pixel
                )

            peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
            peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
            record = {
                "id": sample_id,
                "group": source_record["group"],
                "aspect_bucket": source_record["aspect_bucket"],
                "source": {
                    role: {"path": str(path), "sha256": sha256(path)}
                    for role, path in paths.items()
                },
                "image_size_wh": [width, height],
                "token_grid_hw": list(token_grid),
                "q20_feature_shape": [expected_tokens, HIDDEN_SIZE],
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
                    "raw_min": float(raw_weight.min()),
                    "raw_max": float(raw_weight.max()),
                    "token_mean": float(weight_token.mean()),
                    "pixel_min": float(weight_pixel.min()),
                    "pixel_max": float(weight_pixel.max()),
                    "pixel_mean": float(weight_pixel.mean()),
                },
                "memory_gib": {
                    "peak_allocated": peak_allocated,
                    "peak_reserved": peak_reserved,
                },
                "cache": {
                    "gt_feature": str(feature_rel),
                    "gt_feature_sha256": sha256(output / feature_rel),
                    "weight": str(weight_rel),
                    "weight_sha256": sha256(output / weight_rel),
                    "preview_panel": preview_rel,
                },
            }
            progress["records"].append(record)
            progress["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(output / "progress.json", progress)
            print(
                json.dumps(
                    {
                        "sample": sample_id,
                        "progress": f"{position}/{len(selected_records)}",
                        "size_wh": [width, height],
                        "token_grid_hw": list(token_grid),
                        "peak_allocated_gib": round(peak_allocated, 3),
                        "peak_reserved_gib": round(peak_reserved, 3),
                    }
                ),
                flush=True,
            )
            del q_input, q_gt
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        del backend
        gc.collect()
        torch.cuda.empty_cache()

    records = progress["records"]
    if [record["id"] for record in records] != selected_ids:
        raise RuntimeError("M2a completion order or selected IDs do not match")
    manifest = {
        "schema_version": 1,
        "complete": True,
        "identity": identity,
        "created_at_utc": progress["created_at_utc"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_git_commit": current_git_commit(),
        "source_dataset": {
            "manifest": str(data_root / "manifest.json"),
            "manifest_sha256": identity["data_manifest_sha256"],
            "split": "train" if not args.gate_only else "seven bucket representatives",
            "full_train_count": len(train_records),
            "cached_count": len(records),
        },
        "models": {
            "qwen_checkpoint": {
                "repo": qwen_entry["repo"],
                "revision": qwen_entry["revision"],
            },
            "windowseat_assets": {
                "repo": windowseat_entry["repo"],
                "revision": windowseat_entry["revision"],
            },
        },
        "qwen_feature": {
            "block_one_based": BLOCK_NUMBER,
            "block_zero_based_index": BLOCK_INDEX,
            "flow_timestep": FLOW_TIMESTEP,
            "token_grid": "per sample [H/16,W/16]",
            "hidden_size": HIDDEN_SIZE,
            "stream": "image-stream output after transformer block",
            "distance": "1 - cosine(Q20(I), Q20(GT))",
            "vae_encoding": "posterior mode; deterministic",
            "adapters": "all LoRA adapters disabled",
            "forward_scope": "early stop immediately after block 20",
            "stored_gt_dtype": "bfloat16",
            "input_feature_retention": "Q20(I) discarded after each sample",
        },
        "prompt": {
            "kind": "fixed_precomputed_windowseat_text_embeddings",
            "embedding_file": str(lora_snapshot / "text_embeddings/state_dict.safetensors"),
            "embedding_sha256": sha256(
                lora_snapshot / "text_embeddings/state_dict.safetensors"
            ),
        },
        "weighting": {
            "dq_normalization": (
                f"per-image q{args.q_low_quantile:.4f}/q{args.q_high_quantile:.4f} "
                "robust min-max clipped to [0,1]"
            ),
            "score_formula": "S = D_Q * (0.7 + 0.3 * D_DoLP)",
            "raw_weight_formula": "W_raw = clip(1 + 2*S, 1, 3)",
            "final_weight_formula": "mean-one token W; bilinear pixel W; mean-one again",
            "stored_weight_dtype": "float16",
        },
        "aspect_bucket_counts": dict(Counter(record["aspect_bucket"] for record in records)),
        "memory": {
            "batch_size": 1,
            "single_gpu": True,
            "max_peak_allocated_gib": max(
                record["memory_gib"]["peak_allocated"] for record in records
            ),
            "max_peak_reserved_gib": max(
                record["memory_gib"]["peak_reserved"] for record in records
            ),
        },
        "samples": records,
    }
    atomic_json(complete_path, manifest)
    progress["complete"] = True
    progress["completed_at_utc"] = manifest["completed_at_utc"]
    atomic_json(output / "progress.json", progress)
    print(
        json.dumps(
            {
                "status": "complete",
                "samples": len(records),
                "output": str(output),
                "manifest": str(complete_path),
                "max_peak_allocated_gib": manifest["memory"]["max_peak_allocated_gib"],
                "max_peak_reserved_gib": manifest["memory"]["max_peak_reserved_gib"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
