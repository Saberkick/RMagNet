"""Set one shared DINO loss weight using a fixed training image, without updating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .m1b_train import DINO_REVISION, DinoPatchLoss, INITIAL, M1BDataset, PROJECT, calibrate_weight, load_initial, sha256
from .qwen_backend import QwenSharedBackend
from .stage1_train import DEFAULT_DATA, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--initial", type=Path, default=INITIAL)
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/m1b/sem_calibration.json")
    parser.add_argument("--sample-id", default="13")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Calibration requires two visible GPUs")
    seed_everything(args.seed)
    batch = next(iter(DataLoader(M1BDataset(args.data_root, [args.sample_id], False), batch_size=1)))
    model_device = torch.device("cuda:0")
    dino_device = torch.device("cuda:1")
    backend = QwenSharedBackend.from_local(model_device)
    backend.transformer.enable_gradient_checkpointing()
    backend.set_trainable_branch("transmission")
    load_initial(backend, args.initial, model_device)
    backend.transformer.train()
    dino = DinoPatchLoss(dino_device, layer=6)
    seed_everything(args.seed)
    result = calibrate_weight(backend, dino, batch, model_device, args.fraction)
    result.update({"sample_id": args.sample_id, "seed": args.seed, "initial_sha256": sha256(args.initial),
                   "dino_model": "facebook/dinov2-small", "dino_revision": DINO_REVISION,
                   "dino_layer": 6,
                   "mask_threshold_uint8": json.loads((args.data_root / "m1b_dolp_manifest.json").read_text())["threshold_uint8"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
