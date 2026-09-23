"""Fast C1-L20 cache validation that runs before loading the Qwen backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import safetensors.torch
import torch

DEFAULT_OUTPUT = Path("/share/linmingheng-local/xuke/RMagNet/data_cache/c1_l20")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest_path = args.cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("formula_version") != "c1-l20-qwen-majority-v1":
        raise RuntimeError("Wrong or incomplete C1-L20 cache manifest")
    if manifest["qwen_feature"].get("stored_gt_dtype") != "bfloat16":
        raise RuntimeError("Q20(GT) cache is not the repaired BF16 version")
    if len(manifest["train_ids"]) != 50 or len(manifest["samples"]) != 50:
        raise RuntimeError("Expected 50 C1-L20 training samples")
    for sample in manifest["samples"]:
        sample_id = sample["id"]
        feature_path = args.cache_root / sample["cache"]["gt_feature"]
        weight_path = args.cache_root / sample["cache"]["weight"]
        if sha256(feature_path) != sample["cache"]["gt_feature_sha256"]:
            raise RuntimeError(f"GT feature hash mismatch for {sample_id}")
        feature = safetensors.torch.load_file(feature_path)["q20_gt"]
        if feature.dtype != torch.bfloat16 or feature.shape != (768, 3072) or not torch.isfinite(feature).all():
            raise RuntimeError(f"Invalid Q20(GT) cache for {sample_id}")
        with np.load(weight_path) as stored:
            pixel = stored["weight_pixel"]
            token = stored["weight_token"]
        if pixel.shape != (384, 512) or token.shape != (24, 32):
            raise RuntimeError(f"Invalid weight shape for {sample_id}")
        if not np.isfinite(pixel).all() or not np.isfinite(token).all():
            raise RuntimeError(f"Non-finite weight for {sample_id}")
    print(json.dumps({"status": "ok", "samples": 50, "feature_dtype": "bfloat16"}, indent=2))


if __name__ == "__main__":
    main()
