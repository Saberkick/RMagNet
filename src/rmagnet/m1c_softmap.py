"""Training-only continuous DoLP + DINO(I, GT) patch weights for M1c."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .m1b_train import DINO_REVISION, DinoPatchLoss, sha256
from .stage1_train import DEFAULT_DATA, discover_ids, image_tensor


def make_weight(dolp: torch.Tensor, input_features: torch.Tensor,
                gt_features: torch.Tensor) -> tuple[np.ndarray, dict[str, float]]:
    """Return a positive 24x32 map with per-image mean exactly one."""
    if dolp.shape != (1, 1, 384, 512) or input_features.shape != (1, 768, 384):
        raise ValueError("Unexpected DoLP or DINO patch shape")
    if gt_features.shape != input_features.shape:
        raise ValueError("DINO feature shapes differ")
    p = F.interpolate(dolp.float(), size=(24, 32), mode="area").flatten().clamp(0, 1)
    distance = (1 - (input_features * gt_features).sum(-1)).flatten().clamp_min(0)
    q10, q90 = torch.quantile(distance, torch.tensor([0.1, 0.9], device=distance.device))
    spread = q90 - q10
    d = ((distance - q10) / spread.clamp_min(1e-4)).clamp(0, 1)
    if spread < 1e-4:
        d.zero_()
    raw = 0.25 + 0.75 * (p + d) / 2
    weight = (raw / raw.mean()).reshape(24, 32)
    if not torch.isfinite(weight).all() or weight.min() <= 0:
        raise RuntimeError("Invalid soft weight")
    stats = {"dolp_mean": float(p.mean()), "dino_distance_mean": float(distance.mean()),
             "dino_distance_q10": float(q10), "dino_distance_q90": float(q90),
             "weight_min": float(weight.min()), "weight_max": float(weight.max()),
             "weight_std": float(weight.std())}
    return weight.cpu().numpy().astype(np.float32), stats


def shifted_weight(weight: torch.Tensor, sample_id: str, seed: int) -> torch.Tensor:
    """Fixed toroidal displacement: identical values and local structure, wrong location."""
    if weight.shape[-2:] != (24, 32):
        raise ValueError(f"Expected 24x32 map, got {weight.shape}")
    digest = hashlib.sha256(f"m1c:{seed}:{sample_id}".encode()).digest()
    dy = 6 + int.from_bytes(digest[:4], "big") % 13
    dx = 8 + int.from_bytes(digest[4:8], "big") % 17
    return torch.roll(weight, shifts=(dy, dx), dims=(-2, -1))


def prepare(data_root: Path, output: Path, val_ids: list[str], layer: int) -> Path:
    ids = discover_ids(data_root)
    train_ids = [sample_id for sample_id in ids if sample_id not in val_ids]
    if len(train_ids) != 50 or sorted(val_ids, key=int) != ["11", "12", "17"]:
        raise RuntimeError(f"Expected 50 train and 11/12/17 validation, got {len(train_ids)}")
    m1b_manifest = json.loads((data_root / "m1b_dolp_manifest.json").read_text())
    if ids != m1b_manifest["ids"]:
        raise RuntimeError("M1b data manifest differs")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Existing maps need explicit inspection: {manifest_path}")
    dino = DinoPatchLoss(torch.device("cuda:0"), layer)
    records = []
    with torch.no_grad():
        for sample_id in train_ids:
            inp_path = data_root / "blended" / f"{sample_id}.png"
            gt_path = data_root / "transmission_layer" / f"{sample_id}.png"
            dolp_path = data_root / "dolp" / f"{sample_id}.png"
            inp = image_tensor(inp_path)
            gt = image_tensor(gt_path)
            with Image.open(dolp_path) as loaded:
                dolp = torch.from_numpy(np.asarray(loaded.convert("L"), dtype=np.uint8).copy()).float()[None, None] / 255
            if inp.shape != (3, 384, 512) or gt.shape != inp.shape or dolp.shape != (1, 1, 384, 512):
                raise ValueError(f"Input/GT/DoLP alignment differs for {sample_id}")
            features_i = dino.features(((inp[None] + 1) * .5).clamp(0, 1))
            features_gt = dino.features(((gt[None] + 1) * .5).clamp(0, 1))
            weight, stats = make_weight(dolp.to(dino.device), features_i, features_gt)
            weight_path = output / f"{sample_id}.npy"
            np.save(weight_path, weight, allow_pickle=False)
            records.append({"id": sample_id, "input_sha256": sha256(inp_path),
                            "gt_sha256": sha256(gt_path), "dolp_sha256": sha256(dolp_path),
                            "weight_sha256": sha256(weight_path), **stats})
            print(json.dumps(records[-1]), flush=True)
    manifest = {"method": "0.25 + 0.75*(DoLP_area + robust_DINO_cosine_difference)/2; normalize each map to mean 1",
                "dino_model": "facebook/dinov2-small", "dino_revision": DINO_REVISION,
                "dino_layer": layer, "map_shape": [24, 32], "ids": train_ids,
                "val_ids_excluded": val_ids, "samples": records}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-ids", default="11,12,17")
    parser.add_argument("--layer", type=int, default=6)
    args = parser.parse_args()
    print(prepare(args.data_root, args.output, args.val_ids.split(","), args.layer))


if __name__ == "__main__":
    main()
