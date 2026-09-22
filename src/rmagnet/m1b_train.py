"""M1b: matched Stage-2 continuation with optional DoLP-masked DINO loss.

The RGB model receives only I. GT and DoLP are used by the training objective.
Run one arm per process; visible GPU 0 hosts Qwen and GPU 1 hosts frozen DINO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .qwen_backend import QwenSharedBackend
from .stage1_train import DEFAULT_DATA, append_jsonl, discover_ids, image_tensor, seed_everything, ssim
from .stage2_train import adapter_state, transmission_loss


ROOT = Path("/share/linmingheng-local/xuke")
PROJECT = ROOT / "RMagNet"
INITIAL = PROJECT / "runs/stage2_transmission_r128/best_transmission_lora.safetensors"
DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class M1BDataset(Dataset):
    def __init__(self, root: Path, ids: list[str], augment: bool):
        self.root, self.ids, self.augment = root, ids, augment

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = self.ids[index]
        image = image_tensor(self.root / "blended" / f"{sample_id}.png")
        target = image_tensor(self.root / "transmission_layer" / f"{sample_id}.png")
        with Image.open(self.root / "dolp_mask" / f"{sample_id}.png") as loaded:
            mask = torch.from_numpy(np.asarray(loaded.convert("L"), dtype=np.uint8).copy())
        if image.shape != (3, 384, 512) or target.shape != image.shape or mask.shape != (384, 512):
            raise ValueError(f"Incorrect or unaligned shapes for {sample_id}")
        mask = (mask >= 128).float().unsqueeze(0)
        if self.augment and torch.rand(()) < 0.5:
            image, target, mask = image.flip(-1), target.flip(-1), mask.flip(-1)
        return {"id": sample_id, "image": image, "target": target, "mask": mask}


def wrong_mask(mask: torch.Tensor, sample_id: str, seed: int) -> torch.Tensor:
    """Fixed same-area pixel permutation; correct location and shape are removed."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 1000003 + int(sample_id))
    permutation = torch.randperm(mask.numel(), generator=generator).to(mask.device)
    return mask.reshape(-1)[permutation].reshape_as(mask)


class DinoPatchLoss:
    def __init__(self, device: torch.device, layer: int = 6):
        from transformers import Dinov2Model

        self.device = device
        self.model = Dinov2Model.from_pretrained(
            "facebook/dinov2-small", revision=DINO_REVISION, local_files_only=True
        )
        self.model.to(device).eval().requires_grad_(False)
        self.layer = layer
        if not 1 <= layer <= self.model.config.num_hidden_layers:
            raise ValueError(f"DINO layer {layer} exceeds {self.model.config.num_hidden_layers}")
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def features(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = F.interpolate(rgb.to(self.device), size=(336, 448), mode="bilinear", align_corners=False)
        rgb = (rgb - self.mean) / self.std
        states = self.model(pixel_values=rgb, output_hidden_states=True).hidden_states[self.layer]
        return F.normalize(states[:, 1:, :].float(), dim=-1)

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Encode full images before applying the spatial weight. Do not mask inputs.
        pred01 = ((prediction.float() + 1) * 0.5).clamp(0, 1)
        target01 = ((target.float() + 1) * 0.5).clamp(0, 1)
        pred_features = self.features(pred01)
        with torch.no_grad():
            gt_features = self.features(target01)
        weights = F.interpolate(mask.to(self.device), size=(24, 32), mode="area").flatten(1)
        distances = 1 - (pred_features * gt_features).sum(-1)
        valid = weights.sum(1) > 1e-6
        if not valid.any():
            return pred_features.sum().to(prediction.device) * 0
        loss = ((distances * weights).sum(1) / weights.sum(1).clamp_min(1e-6))[valid].mean()
        return loss.to(prediction.device)


def load_initial(backend: QwenSharedBackend, path: Path, device: torch.device) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    weights = safetensors.torch.load_file(path, device=str(device))
    expected = {name for name, p in backend.transformer.named_parameters() if p.requires_grad}
    if set(weights) != expected:
        raise RuntimeError(f"Adapter keys differ: missing={len(expected-set(weights))}, extra={len(set(weights)-expected)}")
    _, unexpected = backend.transformer.load_state_dict(weights, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected adapter keys: {unexpected[:5]}")


@torch.no_grad()
def validate(backend: QwenSharedBackend, loader: DataLoader, device: torch.device, seed: int) -> dict[str, float]:
    cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    backend.transformer.eval()
    rows = []
    for batch in loader:
        prediction = backend.forward_normalized(batch["image"].to(device), "transmission")
        pred = ((prediction.float() + 1) * 0.5).clamp(0, 1)
        gt = ((batch["target"].to(device).float() + 1) * 0.5).clamp(0, 1)
        mask = batch["mask"].to(device)
        mse = F.mse_loss(pred, gt)
        masked_l1 = ((pred - gt).abs().mean(1, keepdim=True) * mask).sum() / mask.sum().clamp_min(1)
        outside = 1 - mask
        outside_l1 = ((pred - gt).abs().mean(1, keepdim=True) * outside).sum() / outside.sum().clamp_min(1)
        rows.append({"id": batch["id"][0], "l1": float(F.l1_loss(pred, gt)),
                     "psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))),
                     "ssim": float(ssim(pred, gt)), "masked_l1": float(masked_l1),
                     "outside_l1": float(outside_l1)})
    backend.transformer.train()
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state, device)
    means = {key: sum(row[key] for row in rows) / len(rows) for key in ("l1", "psnr", "ssim", "masked_l1", "outside_l1")}
    return {"means": means, "per_image": rows}


def calibrate_weight(backend: QwenSharedBackend, dino: DinoPatchLoss,
                     batch: dict[str, object], model_device: torch.device,
                     fraction: float) -> dict[str, float]:
    # One fixed training batch; no optimizer update and no validation data.
    image = batch["image"].to(model_device)
    target = batch["target"].to(model_device)
    mask = batch["mask"].to(model_device)
    prediction = backend.forward_normalized(image, "transmission")
    base, _ = transmission_loss(prediction, target, 0.2, 0.1)
    semantic = dino(prediction, target, mask)
    base_grad = torch.autograd.grad(base, prediction, retain_graph=True)[0]
    sem_grad = torch.autograd.grad(semantic, prediction)[0]
    base_norm = float(base_grad.float().norm())
    sem_norm = float(sem_grad.float().norm())
    if not math.isfinite(base_norm) or not math.isfinite(sem_norm) or sem_norm <= 0:
        raise RuntimeError(f"Invalid calibration gradients: {base_norm}, {sem_norm}")
    weight = fraction * base_norm / sem_norm
    if not 0 < weight <= 10:
        raise RuntimeError(f"Unreasonable calibrated semantic weight {weight}")
    del prediction, base, semantic, base_grad, sem_grad
    torch.cuda.empty_cache()
    return {"sem_weight": weight, "base_grad_norm": base_norm,
            "sem_grad_norm": sem_norm, "target_gradient_fraction": fraction}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("base", "dolp", "shuffle"), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--initial", type=Path, default=INITIAL)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--sem-weight", type=float, default=0.05)
    parser.add_argument("--dino-layer", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-ids", default="11,12,17")
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.max_steps < 0 or args.learning_rate <= 0:
        raise ValueError("Invalid training schedule")
    if args.arm != "base" and args.sem_weight <= 0:
        raise ValueError("Semantic arms need positive --sem-weight")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Run directory already contains files: {args.run_dir}")
    args.run_dir.mkdir(parents=True)
    ids = discover_ids(args.data_root)
    val_ids = [item for item in args.val_ids.split(",") if item]
    train_ids = [item for item in ids if item not in val_ids]
    if len(train_ids) != 50 or len(val_ids) != 3:
        raise RuntimeError(f"Expected 50/3 split, got {len(train_ids)}/{len(val_ids)}")
    manifest = json.loads((args.data_root / "m1b_dolp_manifest.json").read_text())
    if manifest["ids"] != ids or manifest["val_ids"] != sorted(val_ids, key=int):
        raise RuntimeError("DoLP manifest split differs from training split")
    seed_everything(args.seed)
    model_device = torch.device("cuda:0")
    if torch.cuda.device_count() < (2 if args.arm != "base" else 1):
        raise RuntimeError("M1b semantic arms require two visible CUDA devices")
    dino_device = torch.device("cuda:1")
    backend = QwenSharedBackend.from_local(model_device)
    backend.transformer.enable_gradient_checkpointing()
    backend.set_trainable_branch("transmission")
    load_initial(backend, args.initial, model_device)
    backend.transformer.train()
    backend.vae.eval()
    dino = DinoPatchLoss(dino_device, args.dino_layer) if args.arm != "base" else None
    parameters = [p for p in backend.transformer.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    train_data = M1BDataset(args.data_root, train_ids, augment=True)
    val_data = M1BDataset(args.data_root, val_ids, augment=False)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    # Re-seed after model creation so all arms see the same order, flips, VAE latents.
    seed_everything(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, generator=generator,
                              num_workers=0, pin_memory=True)
    total_steps = args.epochs * len(train_data)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    if total_steps < 1:
        raise RuntimeError("No optimizer steps planned")
    initial_lr = args.learning_rate
    metadata = {"args": vars(args), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "initial_sha256": sha256(args.initial), "train_ids": train_ids, "val_ids": val_ids,
                "optimizer_steps_per_epoch": len(train_data), "total_steps": total_steps,
                "trainable_parameters": sum(p.numel() for p in parameters),
                "mask_threshold_uint8": manifest["threshold_uint8"],
                "semantic_definition": "full-image DINOv2-small patch cosine; mask after encoding",
                "dino_revision": DINO_REVISION if dino else None,
                "dino_layer": args.dino_layer if dino else None,
                "optimizer": "bitsandbytes.PagedAdamW8bit", "model_gpu": torch.cuda.get_device_name(0),
                "dino_gpu": torch.cuda.get_device_name(1) if dino else None}
    (args.run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print(json.dumps(metadata, indent=2, default=str), flush=True)
    initial_metrics = validate(backend, val_loader, model_device, args.seed)
    append_jsonl(args.run_dir / "validation.jsonl", {"step": 0, "epoch": 0, **initial_metrics})
    started = time.monotonic()
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        for batch in train_loader:
            image, target, mask = (batch[key].to(model_device) for key in ("image", "target", "mask"))
            prediction = backend.forward_normalized(image, "transmission")
            base_loss, parts = transmission_loss(prediction, target, 0.2, 0.1)
            semantic_loss = prediction.sum() * 0
            if dino is not None:
                if args.arm == "shuffle":
                    mask = wrong_mask(mask, batch["id"][0], args.seed)
                semantic_loss = dino(prediction, target, mask)
            loss = base_loss + (args.sem_weight * semantic_loss if dino else 0)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(grad_norm) or float(grad_norm) == 0:
                raise RuntimeError(f"Invalid LoRA gradient at step {step + 1}: {grad_norm}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            record = {"step": step, "epoch": epoch, "id": batch["id"][0], "loss": float(loss.detach()),
                      "base_loss": float(base_loss.detach()), "semantic_loss": float(semantic_loss.detach()),
                      **parts, "grad_norm": float(grad_norm), "lr": initial_lr,
                      "elapsed_seconds": time.monotonic() - started,
                      "model_peak_gib": torch.cuda.max_memory_allocated(model_device) / 2**30,
                      "dino_peak_gib": torch.cuda.max_memory_allocated(dino_device) / 2**30 if dino else 0}
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
