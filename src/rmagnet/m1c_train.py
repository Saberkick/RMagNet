"""M1c: matched base / true soft map / displaced soft map Stage-2 continuation."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .m1b_train import DINO_REVISION, DinoPatchLoss, INITIAL, M1BDataset, PROJECT, load_initial, sha256, validate
from .m1c_softmap import shifted_weight
from .qwen_backend import QwenSharedBackend
from .stage1_train import DEFAULT_DATA, append_jsonl, discover_ids, image_tensor, seed_everything
from .stage2_train import adapter_state, transmission_loss


DEFAULT_MAPS = PROJECT / "runs/m1c/softmaps"
DEFAULT_CALIBRATION = PROJECT / "runs/m1c/calibration.json"


class M1CDataset(Dataset):
    def __init__(self, root: Path, ids: list[str], maps: Path, augment: bool):
        self.root, self.ids, self.maps, self.augment = root, ids, maps, augment

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = self.ids[index]
        image = image_tensor(self.root / "blended" / f"{sample_id}.png")
        target = image_tensor(self.root / "transmission_layer" / f"{sample_id}.png")
        weight = torch.from_numpy(np.load(self.maps / f"{sample_id}.npy", allow_pickle=False).copy())
        if image.shape != (3, 384, 512) or target.shape != image.shape or weight.shape != (24, 32):
            raise ValueError(f"Unaligned M1c sample {sample_id}")
        if not torch.isfinite(weight).all() or weight.min() <= 0 or abs(float(weight.mean()) - 1) > 1e-5:
            raise ValueError(f"Invalid M1c weight for {sample_id}")
        if self.augment and torch.rand(()) < .5:
            image, target, weight = image.flip(-1), target.flip(-1), weight.flip(-1)
        return {"id": sample_id, "image": image, "target": target, "weight": weight}


def semantic_loss(dino: DinoPatchLoss, prediction: torch.Tensor,
                  target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    pred01 = ((prediction.float() + 1) * .5).clamp(0, 1)
    gt01 = ((target.float() + 1) * .5).clamp(0, 1)
    pred_features = dino.features(pred01)
    with torch.no_grad():
        gt_features = dino.features(gt01)
    weights = weight.to(dino.device).flatten(1)
    if pred_features.shape[1] != weights.shape[1]:
        raise ValueError("DINO patch count differs from soft map")
    distance = 1 - (pred_features * gt_features).sum(-1)
    return ((distance * weights).sum(1) / weights.sum(1)).mean().to(prediction.device)


def check_maps(data_root: Path, maps: Path, train_ids: list[str], val_ids: list[str], layer: int) -> str:
    path = maps / "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["ids"] != train_ids or manifest["val_ids_excluded"] != val_ids or manifest["dino_layer"] != layer:
        raise RuntimeError("M1c soft-map manifest differs from train/validation split")
    if manifest["dino_revision"] != DINO_REVISION:
        raise RuntimeError("DINO revision differs")
    if [record["id"] for record in manifest["samples"]] != train_ids:
        raise RuntimeError("Soft-map records do not cover exactly the training IDs")
    for record in manifest["samples"]:
        sample_id = record["id"]
        for key, source in (("input", data_root / "blended" / f"{sample_id}.png"),
                            ("gt", data_root / "transmission_layer" / f"{sample_id}.png"),
                            ("dolp", data_root / "dolp" / f"{sample_id}.png"),
                            ("weight", maps / f"{sample_id}.npy")):
            if sha256(source) != record[f"{key}_sha256"]:
                raise RuntimeError(f"Modified {key} for {sample_id}")
    return sha256(path)


def calibration(backend: QwenSharedBackend, dino: DinoPatchLoss,
                sample: dict[str, object], device: torch.device, fraction: float) -> dict[str, float]:
    prediction = backend.forward_normalized(sample["image"][None].to(device), "transmission")
    target = sample["target"][None].to(device)
    weight = sample["weight"][None]
    base, _ = transmission_loss(prediction, target, .2, .1)
    sem = semantic_loss(dino, prediction, target, weight)
    base_grad = torch.autograd.grad(base, prediction, retain_graph=True)[0]
    sem_grad = torch.autograd.grad(sem, prediction)[0]
    base_norm = float(base_grad.float().norm())
    sem_norm = float(sem_grad.float().norm())
    if not math.isfinite(base_norm) or not math.isfinite(sem_norm) or sem_norm <= 0:
        raise RuntimeError("Invalid calibration gradient norms")
    result = {"sem_weight": fraction * base_norm / sem_norm,
              "base_grad_norm": base_norm, "sem_grad_norm": sem_norm,
              "target_gradient_fraction": fraction, "sample_id": sample["id"],
              "seed": 2026, "dino_layer": dino.layer,
              "initial_sha256": sha256(INITIAL)}
    if not 0 < result["sem_weight"] <= 10:
        raise RuntimeError(f"Unreasonable semantic coefficient: {result['sem_weight']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "train"), default="train")
    parser.add_argument("--arm", choices=("base", "soft", "shift"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--maps", type=Path, default=DEFAULT_MAPS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--initial", type=Path, default=INITIAL)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dino-layer", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-ids", default="11,12,17")
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.epochs != 2 or not 1 <= args.max_steps <= 100 or args.learning_rate <= 0:
        raise ValueError("M1c requires 2 epochs, at most 100 steps and positive LR")
    if args.mode == "train" and (args.arm is None or args.run_dir is None):
        parser.error("Training needs --arm and --run-dir")
    if args.mode == "train" and args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Run already exists: {args.run_dir}")
    ids = discover_ids(args.data_root)
    val_ids = args.val_ids.split(",")
    train_ids = [sample_id for sample_id in ids if sample_id not in val_ids]
    if len(train_ids) != 50 or val_ids != ["11", "12", "17"]:
        raise RuntimeError("M1c expects the unchanged 50/3 split")
    maps_sha = check_maps(args.data_root, args.maps, train_ids, val_ids, args.dino_layer)
    model_device = torch.device("cuda:0")
    if torch.cuda.device_count() < (2 if args.mode == "calibrate" or args.arm != "base" else 1):
        raise RuntimeError("M1c needs two visible GPUs for calibration and semantic arms")
    seed_everything(args.seed)
    backend = QwenSharedBackend.from_local(model_device)
    backend.transformer.enable_gradient_checkpointing()
    backend.set_trainable_branch("transmission")
    load_initial(backend, args.initial, model_device)
    backend.transformer.train()
    backend.vae.eval()
    dino = DinoPatchLoss(torch.device("cuda:1"), args.dino_layer) if args.mode == "calibrate" or args.arm != "base" else None
    train_data = M1CDataset(args.data_root, train_ids, args.maps, augment=True)
    if args.mode == "calibrate":
        if args.calibration.exists():
            raise FileExistsError(args.calibration)
        if args.initial != INITIAL or args.seed != 2026:
            raise ValueError("Calibration requires the fixed M1c initial checkpoint and seed")
        seed_everything(args.seed)
        sample = M1CDataset(args.data_root, ["13"], args.maps, augment=False)[0]
        result = calibration(backend, dino, sample, model_device, .1)
        result.update({"maps_manifest_sha256": maps_sha, "dino_revision": DINO_REVISION})
        args.calibration.parent.mkdir(parents=True, exist_ok=True)
        args.calibration.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return
    if args.initial != INITIAL:
        raise ValueError("M1c arms must use the same Stage 2 initial LoRA")
    cal = json.loads(args.calibration.read_text())
    if (cal["initial_sha256"] != sha256(args.initial) or cal["maps_manifest_sha256"] != maps_sha
            or cal["seed"] != args.seed or cal["dino_layer"] != args.dino_layer):
        raise RuntimeError("M1c calibration does not match this run")
    sem_weight = 0.0 if args.arm == "base" else cal["sem_weight"]
    parameters = [p for p in backend.transformer.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    seed_everything(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, generator=generator,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(M1BDataset(args.data_root, val_ids, augment=False), batch_size=1)
    total_steps = min(args.epochs * len(train_data), args.max_steps)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"args": vars(args), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "initial_sha256": sha256(args.initial), "maps_manifest_sha256": maps_sha,
                "calibration_sha256": sha256(args.calibration), "train_ids": train_ids, "val_ids": val_ids,
                "sem_weight": sem_weight, "total_steps": total_steps,
                "optimizer": "bitsandbytes.PagedAdamW8bit", "model_gpu": torch.cuda.get_device_name(0),
                "dino_gpu": torch.cuda.get_device_name(1) if dino else None,
                "semantic_definition": "full-image DINO(pred,GT) cosine, fixed training-only DoLP + DINO(I,GT) soft patch weights",
                "control": "fixed toroidal shift per sample; same patch values and local spatial pattern"}
    (args.run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    initial_metrics = validate(backend, val_loader, model_device, args.seed)
    append_jsonl(args.run_dir / "validation.jsonl", {"step": 0, "epoch": 0, **initial_metrics})
    started = time.monotonic()
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        for batch in train_loader:
            image = batch["image"].to(model_device)
            target = batch["target"].to(model_device)
            weight = batch["weight"]
            if args.arm == "shift":
                weight = shifted_weight(weight, batch["id"][0], args.seed)
            prediction = backend.forward_normalized(image, "transmission")
            base, parts = transmission_loss(prediction, target, .2, .1)
            base_grad = torch.autograd.grad(base, prediction, retain_graph=True)[0]
            sem = prediction.new_zeros(())
            sem_grad = torch.zeros_like(base_grad)
            if dino is not None:
                sem = semantic_loss(dino, prediction, target, weight)
                sem_grad = torch.autograd.grad(sem, prediction)[0]
            combined = (base_grad + sem_weight * sem_grad).detach()
            if not torch.isfinite(combined).all():
                raise RuntimeError(f"Non-finite prediction gradient at step {step + 1}")
            # Equivalent first-order gradient to (base + coefficient*semantic).backward().
            prediction.backward(combined)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(grad_norm) or float(grad_norm) == 0:
                raise RuntimeError(f"Invalid adapter gradient at step {step + 1}: {grad_norm}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            base_norm = float(base_grad.float().norm())
            sem_norm = float(sem_grad.float().norm())
            record = {"step": step, "epoch": epoch, "id": batch["id"][0],
                      "loss": float(base.detach() + sem_weight * sem.detach()),
                      "base_loss": float(base.detach()), "semantic_loss": float(sem.detach()),
                      "base_prediction_grad_norm": base_norm,
                      "weighted_sem_prediction_grad_norm": sem_weight * sem_norm,
                      "sem_to_base_prediction_grad_ratio": sem_weight * sem_norm / max(base_norm, 1e-12),
                      "prediction_grad_cosine": float(F.cosine_similarity(base_grad.flatten()[None], sem_grad.flatten()[None])) if dino else 0.,
                      "combined_prediction_grad_norm": float(combined.float().norm()),
                      "grad_norm": float(grad_norm), "lr": args.learning_rate,
                      "elapsed_seconds": time.monotonic() - started,
                      "model_peak_gib": torch.cuda.max_memory_allocated(model_device) / 2**30,
                      "dino_peak_gib": torch.cuda.max_memory_allocated(dino.device) / 2**30 if dino else 0., **parts}
            append_jsonl(args.run_dir / "train.jsonl", record)
            if step == 1 or step % 10 == 0:
                print(json.dumps(record), flush=True)
            if step >= total_steps:
                break
        result = validate(backend, val_loader, model_device, args.seed)
        summary = {"step": step, "epoch": epoch, **result}
        append_jsonl(args.run_dir / "validation.jsonl", summary)
        print(json.dumps(summary), flush=True)
        if step >= total_steps:
            break
    if args.save_final:
        output = args.run_dir / "final_transmission_lora.safetensors"
        safetensors.torch.save_file(adapter_state(backend), output)
        print(f"saved {output} sha256={sha256(output)}", flush=True)
    (args.run_dir / "DONE.json").write_text(json.dumps({"steps": step, "epochs_completed": epoch,
        "elapsed_seconds": time.monotonic() - started, "final_validation": result}, indent=2) + "\n")


if __name__ == "__main__":
    main()
