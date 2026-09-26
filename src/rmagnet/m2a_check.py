"""Validate the completed dynamic-shape M2a Q20 cache without loading Qwen."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import safetensors.torch
import torch

from .c1_l20_prepare import sha256


DEFAULT_CACHE = Path("/share/linmingheng-local/xuke/RMagNet/data_cache/m2a_q20")
FORMULA_VERSION = "m2a-q20-variable-aspect-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    root = args.cache_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest.get("identity", {})
    if not manifest.get("complete") or identity.get("formula_version") != FORMULA_VERSION:
        raise RuntimeError("M2a cache manifest is incomplete or incompatible")
    expected_count = 7 if identity.get("selection") == "seven-bucket-gate" else 144
    records = manifest["samples"]
    if len(records) != expected_count or len({record["id"] for record in records}) != expected_count:
        raise RuntimeError(f"Expected {expected_count} unique M2a samples")

    feature_bytes = 0
    weight_bytes = 0
    grids = Counter()
    for record in records:
        sample_id = record["id"]
        width, height = record["image_size_wh"]
        token_h, token_w = record["token_grid_hw"]
        if (token_h, token_w) != (height // 16, width // 16):
            raise RuntimeError(f"Wrong token grid for {sample_id}")
        feature_path = root / record["cache"]["gt_feature"]
        weight_path = root / record["cache"]["weight"]
        if sha256(feature_path) != record["cache"]["gt_feature_sha256"]:
            raise RuntimeError(f"Feature hash mismatch for {sample_id}")
        if sha256(weight_path) != record["cache"]["weight_sha256"]:
            raise RuntimeError(f"Weight hash mismatch for {sample_id}")
        feature = safetensors.torch.load_file(feature_path)["q20_gt"]
        expected_shape = (token_h * token_w, 3072)
        if feature.dtype != torch.bfloat16 or tuple(feature.shape) != expected_shape:
            raise RuntimeError(
                f"Invalid Q20 feature for {sample_id}: {feature.dtype} {tuple(feature.shape)}"
            )
        if not torch.isfinite(feature).all():
            raise RuntimeError(f"Non-finite Q20 feature for {sample_id}")
        with np.load(weight_path) as stored:
            pixel = stored["weight_pixel"]
            token = stored["weight_token"]
        if pixel.shape != (height, width) or token.shape != (token_h, token_w):
            raise RuntimeError(f"Invalid weight shape for {sample_id}")
        if not np.isfinite(pixel).all() or not np.isfinite(token).all():
            raise RuntimeError(f"Non-finite weight for {sample_id}")
        if abs(float(pixel.mean()) - 1.0) > 2e-3 or abs(float(token.mean()) - 1.0) > 2e-3:
            raise RuntimeError(f"Mean-one weight invariant failed for {sample_id}")
        feature_bytes += feature_path.stat().st_size
        weight_bytes += weight_path.stat().st_size
        grids[f"{token_h}x{token_w}"] += 1

    print(
        json.dumps(
            {
                "status": "ok",
                "samples": len(records),
                "selection": identity["selection"],
                "feature_dtype": "bfloat16",
                "feature_bytes": feature_bytes,
                "weight_bytes": weight_bytes,
                "token_grid_count": len(grids),
                "aspect_bucket_counts": manifest["aspect_bucket_counts"],
                "max_peak_allocated_gib": manifest["memory"]["max_peak_allocated_gib"],
                "max_peak_reserved_gib": manifest["memory"]["max_peak_reserved_gib"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
