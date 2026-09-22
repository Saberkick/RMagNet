"""Align the supplied 8-bit DoLP PNGs with the existing 512x384 pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .stage1_train import DEFAULT_DATA, discover_ids


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--threshold", type=int, default=64)
    parser.add_argument("--val-ids", default="11,12,17")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 255:
        raise ValueError("Threshold must be in [0,255]")
    ids = discover_ids(args.data_root)
    source = args.data_root / "dolp_original"
    expected = {f"{sample_id}_DoLP.png" for sample_id in ids}
    actual = {path.name for path in source.glob("*_DoLP.png")}
    if expected != actual:
        raise RuntimeError(f"DoLP IDs differ: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    dolp_dir = args.data_root / "dolp"
    mask_dir = args.data_root / "dolp_mask"
    dolp_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    records = []
    for sample_id in ids:
        path = source / f"{sample_id}_DoLP.png"
        with Image.open(path) as image:
            if image.mode != "L" or image.width < 512 or image.height < 384 or image.width * 3 != image.height * 4:
                raise ValueError(f"Unexpected DoLP encoding: {path}, {image.mode}, {image.size}")
            original_size = list(image.size)
            resized = image.resize((512, 384), Image.Resampling.LANCZOS)
            dolp = np.asarray(resized, dtype=np.uint8)
        mask = (dolp >= args.threshold).astype(np.uint8) * 255
        with Image.open(args.data_root / "blended" / f"{sample_id}.png") as inp:
            with Image.open(args.data_root / "transmission_layer" / f"{sample_id}.png") as gt:
                if inp.size != (512, 384) or gt.size != inp.size:
                    raise ValueError(f"RGB size mismatch for {sample_id}")
        dolp_path = dolp_dir / f"{sample_id}.png"
        mask_path = mask_dir / f"{sample_id}.png"
        resized.save(dolp_path)
        Image.fromarray(mask, mode="L").save(mask_path)
        records.append({
            "id": sample_id,
            "original_size": original_size,
            "source_sha256": sha256(path),
            "dolp_sha256": sha256(dolp_path),
            "mask_sha256": sha256(mask_path),
            "coverage": float((mask > 0).mean()),
            "median": float(np.median(dolp)),
            "p90": float(np.percentile(dolp, 90)),
        })
    val_ids = {item for item in args.val_ids.split(",") if item}
    train_coverage = [r["coverage"] for r in records if r["id"] not in val_ids]
    manifest = {
        "threshold_uint8": args.threshold,
        "resize": "Pillow Lanczos, full 4:3 source frame to 512x384",
        "mask_rule": "DoLP >= threshold; no per-image top-k",
        "ids": ids,
        "val_ids": sorted(val_ids, key=int),
        "train_coverage_min": min(train_coverage),
        "train_coverage_median": float(np.median(train_coverage)),
        "train_coverage_max": max(train_coverage),
        "samples": records,
    }
    (args.data_root / "m1b_dolp_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
