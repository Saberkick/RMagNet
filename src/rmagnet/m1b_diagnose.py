"""Zero-update check: does DoLP select DINO differences and detect blur?"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .m1b_train import DinoPatchLoss, M1BDataset, PROJECT
from .stage1_train import DEFAULT_DATA, discover_ids


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "runs/m1b/diagnostics")
    parser.add_argument("--val-ids", default="11,12,17")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    dino = DinoPatchLoss(device, layer=6)
    ids = discover_ids(args.data_root)
    val_ids = {item for item in args.val_ids.split(",") if item}
    loader = DataLoader(M1BDataset(args.data_root, ids, False), batch_size=1, shuffle=False)
    rows = []
    for batch in loader:
        sample_id = batch["id"][0]
        image = ((batch["image"].to(device) + 1) * 0.5).clamp(0, 1)
        target = ((batch["target"].to(device) + 1) * 0.5).clamp(0, 1)
        blurred = F.avg_pool2d(F.pad(target, (4, 4, 4, 4), mode="reflect"), 9, stride=1)
        gt_features = dino.features(target)
        input_dist = 1 - (dino.features(image) * gt_features).sum(-1)
        blur_dist = 1 - (dino.features(blurred) * gt_features).sum(-1)
        mask = F.interpolate(batch["mask"].to(device), size=(24, 32), mode="area").flatten(1)
        other = 1 - mask
        def weighted(values: torch.Tensor, weights: torch.Tensor) -> float:
            return float((values * weights).sum() / weights.sum().clamp_min(1e-6))
        rows.append({"id": sample_id, "split": "val" if sample_id in val_ids else "train",
                     "coverage": float(batch["mask"].mean()),
                     "input_gt_inside": weighted(input_dist, mask),
                     "input_gt_outside": weighted(input_dist, other),
                     "blur_gt_inside": weighted(blur_dist, mask),
                     "blur_gt_outside": weighted(blur_dist, other)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "patch_distances.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = {split: {key: sum(row[key] for row in rows if row["split"] == split) / sum(row["split"] == split for row in rows)
                     for key in ("coverage", "input_gt_inside", "input_gt_outside", "blur_gt_inside", "blur_gt_outside")}
             for split in ("train", "val")}
    report = {"dino": "facebook/dinov2-small, layer 6, whole image 448x336 before mask",
              "mask": "DoLP >= 64/255 at 512x384, area downsample to 24x32",
              "blur": "9x9 average filter with reflect padding, diagnostic only",
              "means": means, "rows": rows}
    (args.output_dir / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(means, indent=2), flush=True)


if __name__ == "__main__":
    main()
