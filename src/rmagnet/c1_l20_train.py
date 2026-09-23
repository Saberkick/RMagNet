"""C1-L20: Qwen block-20 guided continuation of the Stage-2 transmission LoRA."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .m1b_train import load_initial, sha256
from .qwen_backend import ADAPTER_NAMES, QwenSharedBackend
from .qwen_layer_probe import deterministic_encode
from .stage1_train import (
    DEFAULT_DATA,
    append_jsonl,
    discover_ids,
    image_tensor,
    make_scheduler,
    rank,
    seed_everything,
    setup_distributed,
    ssim,
    sync_gradients,
    trainable_parameters,
    world_size,
)
from .stage2_train import adapter_state, transmission_loss


ROOT = Path("/share/linmingheng-local/xuke")
PROJECT = ROOT / "RMagNet"
DEFAULT_CACHE = PROJECT / "data_cache/c1_l20"
DEFAULT_INITIAL = PROJECT / "runs/stage2_transmission_r128/best_transmission_lora.safetensors"
BLOCK_INDEX = 19
TOKEN_GRID = (24, 32)


class StopAtQ20(Exception):
    """Internal control flow used to stop the frozen teacher after block 20."""


class C1Dataset(Dataset):
    def __init__(self, data_root: Path, cache_root: Path, sample_ids: list[str]):
        self.data_root = data_root
        self.cache_root = cache_root
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = self.sample_ids[index]
        image = image_tensor(self.data_root / "blended" / f"{sample_id}.png")
        target = image_tensor(self.data_root / "transmission_layer" / f"{sample_id}.png")
        with np.load(self.cache_root / "weights" / f"{sample_id}.npz") as stored:
            weight_pixel = torch.from_numpy(stored["weight_pixel"].astype(np.float32))
            weight_token = torch.from_numpy(stored["weight_token"].astype(np.float32))
        q20_gt = safetensors.torch.load_file(
            self.cache_root / "gt_features" / f"{sample_id}.safetensors"
        )["q20_gt"].float()
        with Image.open(self.data_root / "dolp" / f"{sample_id}.png") as loaded:
            dolp = np.asarray(loaded, dtype=np.float32) / 255.0
        if image.shape != (3, 384, 512) or target.shape != image.shape:
            raise ValueError(f"Incorrect RGB pair shape for {sample_id}")
        if weight_pixel.shape != (384, 512) or weight_token.shape != TOKEN_GRID:
            raise ValueError(f"Incorrect weight shape for {sample_id}")
        if q20_gt.shape[:1] != (TOKEN_GRID[0] * TOKEN_GRID[1],) or q20_gt.ndim != 2:
            raise ValueError(f"Incorrect Q20 target shape for {sample_id}: {q20_gt.shape}")
        arrays = (weight_pixel.numpy(), weight_token.numpy(), q20_gt.numpy(), dolp)
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError(f"Non-finite cached input for {sample_id}")
        if abs(float(weight_pixel.mean()) - 1.0) > 2e-3:
            raise ValueError(f"Pixel weight mean differs from one for {sample_id}")
        # No random flip: cached Qwen features include position and global context.
        return {
            "id": sample_id,
            "image": image,
            "target": target,
            "weight_pixel": weight_pixel,
            "weight_token": weight_token,
            "q20_gt": q20_gt,
        }


def validate_cache(cache_root: Path, data_root: Path, train_ids: list[str]) -> dict[str, object]:
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("formula_version") != "c1-l20-qwen-majority-v1":
        raise RuntimeError("C1-L20 cache is incomplete or uses another formula")
    feature = manifest["qwen_feature"]
    if feature["block_zero_based_index"] != BLOCK_INDEX or feature["flow_timestep"] != 499:
        raise RuntimeError("C1-L20 cache uses a different Qwen feature definition")
    if manifest["train_ids"] != train_ids or len(manifest["samples"]) != len(train_ids):
        raise RuntimeError("C1-L20 cache split differs from the current dataset")
    finite_keys = (
        "score_min", "score_max", "raw_min", "raw_max",
        "token_mean_after_normalization", "pixel_min", "pixel_max", "pixel_mean",
    )
    for record in manifest["samples"]:
        sample_id = record["id"]
        source = {
            "input": data_root / "blended" / f"{sample_id}.png",
            "gt": data_root / "transmission_layer" / f"{sample_id}.png",
            "dolp": data_root / "dolp" / f"{sample_id}.png",
        }
        for role, path in source.items():
            if not path.is_file() or record["source"][role]["sha256"] != sha256(path):
                raise RuntimeError(f"Cache source hash mismatch: {sample_id}/{role}")
        qstats = record["q_difference"]
        wstats = record["weight_stats"]
        values = [qstats[key] for key in qstats if key != "normalization"]
        values += [wstats[key] for key in finite_keys]
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise RuntimeError(f"Non-finite D_Q/S/W manifest value for {sample_id}")
        if not (cache_root / record["cache"]["gt_feature"]).is_file():
            raise FileNotFoundError(record["cache"]["gt_feature"])
        if not (cache_root / record["cache"]["weight"]).is_file():
            raise FileNotFoundError(record["cache"]["weight"])
    return manifest


def q20_prediction_features(backend: QwenSharedBackend, prediction: torch.Tensor) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def stop_hook(_module, _inputs, output):
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("Unexpected Qwen block-20 output")
        captured["feature"] = output[1]
        raise StopAtQ20

    handle = backend.transformer.transformer_blocks[BLOCK_INDEX].register_forward_hook(stop_hook)
    backend.transformer.disable_lora()
    try:
        latent = deterministic_encode(backend, prediction)
        try:
            backend.upstream.flow_step(latent, backend.transformer, backend.vae, backend.embeddings)
        except StopAtQ20:
            pass
    finally:
        handle.remove()
        backend.transformer.enable_lora()
        backend.transformer.set_adapter(ADAPTER_NAMES["transmission"])
    if "feature" not in captured:
        raise RuntimeError("Frozen Qwen teacher did not reach block 20")
    feature = captured["feature"]
    if feature.ndim != 3 or feature.shape[1] != TOKEN_GRID[0] * TOKEN_GRID[1]:
        raise ValueError(f"Unexpected predicted Q20 shape: {feature.shape}")
    return feature


def c1_losses(
    prediction: torch.Tensor,
    image: torch.Tensor,
    target: torch.Tensor,
    weight_pixel: torch.Tensor,
    weight_token: torch.Tensor,
    q20_prediction: torch.Tensor,
    q20_gt: torch.Tensor,
    local_coefficient: float,
    keep_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
    base, base_parts = transmission_loss(prediction, target, 0.2, 0.1)
    pred01 = ((prediction.float() + 1) * 0.5).clamp(0, 1)
    target01 = ((target.float() + 1) * 0.5).clamp(0, 1)
    image01 = ((image.float() + 1) * 0.5).clamp(0, 1)
    pixel_weight = weight_pixel.float().unsqueeze(1)
    token_weight = weight_token.float().flatten(1)

    charbonnier = torch.sqrt((pred01 - target01).square() + 1e-6).mean(1, keepdim=True)
    local = (charbonnier * pixel_weight).sum() / pixel_weight.sum().clamp_min(1e-6)

    w_min = pixel_weight.amin(dim=(-2, -1), keepdim=True)
    w_max = pixel_weight.amax(dim=(-2, -1), keepdim=True)
    importance = (pixel_weight - w_min) / (w_max - w_min).clamp_min(1e-6)
    low_response = 1.0 - importance
    keep = ((pred01 - image01).abs().mean(1, keepdim=True) * low_response).sum()
    keep = keep / low_response.sum().clamp_min(1e-6)

    q_pred = F.normalize(q20_prediction.float(), dim=-1)
    q_target = F.normalize(q20_gt.float(), dim=-1)
    q_distance = 1.0 - (q_pred * q_target).sum(-1)
    q20 = (q_distance * token_weight).sum() / token_weight.sum().clamp_min(1e-6)

    base_bundle = base + local_coefficient * local + keep_coefficient * keep
    values = (base, local, keep, q20, base_bundle)
    if not all(torch.isfinite(value) for value in values):
        raise RuntimeError("Non-finite C1-L20 loss component")
    tensors = {"base": base, "local": local, "keep": keep, "q20": q20}
    scalars = {**base_parts, "base_loss": float(base.detach()), "local_loss": float(local.detach()),
               "keep_loss": float(keep.detach()), "q20_loss": float(q20.detach())}
    return base_bundle, q20, tensors, scalars


def global_norm(tensor: torch.Tensor) -> float:
    value = tensor.float().square().sum()
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return math.sqrt(max(0.0, float(value)))


def sample_parameters(parameters: list[torch.nn.Parameter], count: int = 32) -> list[torch.Tensor]:
    samples = []
    for parameter in parameters:
        flat = parameter.detach().reshape(-1)
        if flat.numel() <= count:
            selected = flat
        else:
            indices = torch.linspace(0, flat.numel() - 1, count, device=flat.device).long()
            selected = flat[indices]
        samples.append(selected.float().cpu())
    return samples


def parameter_sample_delta(before: list[torch.Tensor], parameters: list[torch.nn.Parameter]) -> float:
    after = sample_parameters(parameters)
    return math.sqrt(sum(float((new - old).square().sum()) for old, new in zip(before, after)))


def save_rgb(tensor01: torch.Tensor, path: Path) -> None:
    array = tensor01.detach().float().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array, mode="RGB").save(path)


@torch.no_grad()
def validate(
    backend: QwenSharedBackend,
    loader: DataLoader,
    device: torch.device,
    seed: int,
    output_root: Path,
    label: str,
) -> dict[str, object]:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    backend.transformer.eval()
    backend.vae.eval()
    pred_dir = output_root / "predictions" / label
    error_dir = output_root / "error_maps" / label
    pred_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        prediction = backend.forward_normalized(image, "transmission")
        pred01 = ((prediction.float() + 1) * 0.5).clamp(0, 1)
        gt01 = ((target.float() + 1) * 0.5).clamp(0, 1)
        mse = F.mse_loss(pred01, gt01)
        row = {
            "id": batch["id"][0],
            "l1": float(F.l1_loss(pred01, gt01)),
            "psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))),
            "ssim": float(ssim(pred01, gt01)),
        }
        rows.append(row)
        save_rgb(pred01[0], pred_dir / f"{row['id']}.png")
        error = (pred01[0] - gt01[0]).abs().mean(0).clamp(0, 1)
        Image.fromarray(error.mul(255).round().byte().cpu().numpy(), mode="L").save(
            error_dir / f"{row['id']}.png"
        )
    backend.transformer.train()
    backend.vae.eval()
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state, device)
    means = {key: sum(float(row[key]) for row in rows) / len(rows) for key in ("l1", "psnr", "ssim")}
    metrics_path = output_root / "metrics.csv"
    first = not metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "id", "l1", "psnr", "ssim"])
        if first:
            writer.writeheader()
        for row in rows:
            writer.writerow({"label": label, **row})
    return {"means": means, "per_image": rows, "prediction_dir": str(pred_dir)}


def hardlink_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    os.link(source, destination)


def save_epoch_checkpoint(
    run_dir: Path,
    epoch_number: int,
    global_step: int,
    next_epoch: int,
    next_micro_step: int,
    backend: QwenSharedBackend,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    lambda_q: float,
    metrics: dict[str, object],
    top_records: list[dict[str, object]],
    epochs_without_improvement: int,
    smoke: bool,
) -> list[dict[str, object]]:
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    epoch_file = checkpoints / ("smoke.safetensors" if smoke else f"epoch_{epoch_number:04d}.safetensors")
    safetensors.torch.save_file(adapter_state(backend), epoch_file)
    score = float(metrics["means"]["psnr"])
    record = {"epoch": epoch_number, "step": global_step, "val_psnr": score, "file": str(epoch_file)}
    top_records = [item for item in top_records if int(item["epoch"]) != epoch_number] + [record]
    top_records.sort(key=lambda item: float(item["val_psnr"]), reverse=True)
    keep = top_records[:3]
    if not smoke:
        for item in top_records[3:]:
            path = Path(str(item["file"]))
            if path.exists():
                path.unlink()
    top_records = keep
    (checkpoints / "top_epochs.json").write_text(json.dumps(top_records, indent=2) + "\n")

    hardlink_replace(epoch_file, checkpoints / "last" / "transmission_lora.safetensors")
    torch.save(
        {
            "global_step": global_step,
            "next_epoch": next_epoch,
            "next_micro_step": next_micro_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "lambda_q": lambda_q,
            "top_records": top_records,
            "epochs_without_improvement": epochs_without_improvement,
        },
        checkpoints / "last" / "trainer_state.pt",
    )
    (checkpoints / "last" / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if top_records and int(top_records[0]["epoch"]) == epoch_number:
        hardlink_replace(epoch_file, checkpoints / "best" / "transmission_lora.safetensors")
        (checkpoints / "best" / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return top_records


def load_resume(
    checkpoint: Path,
    backend: QwenSharedBackend,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> dict[str, object]:
    weights = safetensors.torch.load_file(checkpoint / "transmission_lora.safetensors", device=str(device))
    _, unexpected = backend.transformer.load_state_dict(weights, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected resume adapter keys: {unexpected[:5]}")
    state = torch.load(checkpoint / "trainer_state.pt", map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=-1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--local-coefficient", type=float, default=0.25)
    parser.add_argument("--keep-coefficient", type=float, default=0.10)
    parser.add_argument("--lambda-q-min", type=float, default=0.02)
    parser.add_argument("--lambda-q-max", type=float, default=0.5)
    parser.add_argument("--lambda-q-ema", type=float, default=0.9)
    parser.add_argument("--gradient-measure-every", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--val-ids", default="11,12,17")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", choices=("auto", "none"), default="auto")
    parser.add_argument("--smoke-samples", type=int, default=8)
    parser.add_argument("--resume-verification", action="store_true")
    return parser.parse_args()


def write_summary(run_dir: Path, args: argparse.Namespace, global_step: int, completed_epochs: int,
                  best_psnr: float, stopped_early: bool) -> None:
    text = f"""# C1-L20 training summary

- Mode: `{args.mode}`
- Completed epochs: {completed_epochs}
- Global optimizer updates: {global_step}
- Best validation PSNR: {best_psnr:.6f}
- Early stopped: {str(stopped_early).lower()}
- Initialization: `{args.initial}`
- Qwen teacher: frozen base, block 20 only
- Training input at inference: I only
"""
    (run_dir / "training_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = setup_distributed()
    is_main = rank() == 0
    if world_size() != 4:
        raise RuntimeError(f"C1-L20 scripts require exactly four processes, got {world_size()}")
    if args.batch_size != 1:
        raise ValueError("C1-L20 currently fixes per-GPU batch size to one")
    if args.epochs < 1 or args.max_steps < 0:
        raise ValueError("Invalid epoch/step schedule")
    if args.mode == "smoke" and (args.gradient_accumulation != 1 or args.max_steps not in (5, 6)):
        raise ValueError("Smoke requires accumulation=1 and max_steps=5, or 6 for resume verification")

    seed_everything(args.seed)
    all_ids = discover_ids(args.data_root)
    val_ids = [value for value in args.val_ids.split(",") if value]
    train_ids = [value for value in all_ids if value not in val_ids]
    manifest = validate_cache(args.cache_root, args.data_root, train_ids)
    if args.mode == "smoke":
        train_ids = train_ids[: args.smoke_samples]
        if len(train_ids) != 8:
            raise RuntimeError(f"Smoke requires exactly 8 images, got {len(train_ids)}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    for folder in ("logs", "checkpoints", "validation/predictions", "validation/error_maps"):
        (args.run_dir / folder).mkdir(parents=True, exist_ok=True)

    train_data = C1Dataset(args.data_root, args.cache_root, train_ids)
    val_data = C1Dataset(args.data_root, args.cache_root, val_ids)
    sampler = DistributedSampler(train_data, num_replicas=world_size(), rank=rank(), shuffle=True, seed=args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_options.update({"persistent_workers": True, "multiprocessing_context": "spawn"})
    train_loader = DataLoader(train_data, **loader_options)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    backend = QwenSharedBackend.from_local(device)
    backend.transformer.enable_gradient_checkpointing()
    backend.set_trainable_branch("transmission")
    load_initial(backend, args.initial, device)
    backend.transformer.train()
    backend.vae.eval()
    parameters = trainable_parameters(backend)
    names = [name for name, value in backend.transformer.named_parameters() if value.requires_grad]
    if not names or any(".lora_" not in name or f".{ADAPTER_NAMES['transmission']}." not in name for name in names):
        raise RuntimeError("Trainable tensors are not exclusively LoRA_T")
    if any(parameter.requires_grad for parameter in backend.vae.parameters()):
        raise RuntimeError("VAE must be frozen")
    if any(value.requires_grad for name, value in backend.transformer.named_parameters() if ".lora_" not in name):
        raise RuntimeError("Qwen backbone must be frozen")

    optimizer = bnb.optim.PagedAdamW8bit(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_ids) / (world_size() * args.batch_size * args.gradient_accumulation))
    planned_steps = args.epochs * steps_per_epoch
    total_steps = min(planned_steps, args.max_steps) if args.max_steps else planned_steps
    warmup_steps = args.warmup_steps
    if warmup_steps < 0:
        warmup_steps = min(total_steps, max(10, math.ceil(total_steps * 0.05)))
    scheduler = make_scheduler(optimizer, warmup_steps, total_steps)

    global_step, start_epoch, resume_micro_step = 0, 0, 0
    lambda_q = args.lambda_q_min
    top_records: list[dict[str, object]] = []
    resume_dir = args.run_dir / "checkpoints/last"
    resumed_epochs_without = 0
    resumed = args.resume == "auto" and (resume_dir / "trainer_state.pt").is_file()
    if resumed:
        state = load_resume(resume_dir, backend, optimizer, scheduler, device)
        global_step = int(state["global_step"])
        start_epoch = int(state["next_epoch"])
        resume_micro_step = int(state["next_micro_step"])
        lambda_q = float(state["lambda_q"])
        top_records = list(state.get("top_records", []))
        resumed_epochs_without = int(state.get("epochs_without_improvement", 0))
    elif args.resume_verification:
        raise RuntimeError("Resume verification requested without a smoke checkpoint")

    before_samples = sample_parameters(parameters)
    torch.cuda.reset_peak_memory_stats(device)
    seed_everything(args.seed + rank())
    config = {
        "args": vars(args),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "world_size": world_size(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "cache_manifest_sha256": sha256(args.cache_root / "manifest.json"),
        "initial_sha256": sha256(args.initial),
        "q20_gt_shape": manifest["samples"][0]["q20_feature_shape"],
        "updates_per_epoch": steps_per_epoch,
        "planned_updates": total_steps,
        "warmup_steps": warmup_steps,
        "effective_batch": world_size() * args.batch_size * args.gradient_accumulation,
        "augmentation": "none; positional Q20(GT) cache must remain aligned",
        "teacher": "same frozen NF4 Qwen backbone, all LoRA disabled, early stop after block 20",
        "optimizer": "bitsandbytes.PagedAdamW8bit",
        "resumed": resumed,
    }
    if is_main and not args.resume_verification:
        (args.run_dir / "config.yaml").write_text(json.dumps(config, indent=2, default=str) + "\n")
        print(json.dumps(config, indent=2, default=str), flush=True)

    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    memory_history: list[list[dict[str, float]]] = []
    loss_finite = True
    q_path_nonzero = False
    frozen_clean = True
    cache_read_ok = True
    actual_ratio = 0.0
    measured_base_norm = 0.0
    measured_q_norm = 0.0
    last_validation: dict[str, object] | None = None
    best_psnr = max([float(item["val_psnr"]) for item in top_records], default=-math.inf)
    epochs_without_improvement = resumed_epochs_without
    stopped_early = False
    completed_epochs = start_epoch
    stop = global_step >= total_steps

    for epoch_index in range(start_epoch, args.epochs):
        if stop:
            break
        sampler.set_epoch(epoch_index)
        last_micro = -1
        for micro_step, batch in enumerate(train_loader):
            if epoch_index == start_epoch and micro_step < resume_micro_step:
                continue
            last_micro = micro_step
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            weight_pixel = batch["weight_pixel"].to(device, non_blocking=True)
            weight_token = batch["weight_token"].to(device, non_blocking=True)
            q20_gt = batch["q20_gt"].to(device, non_blocking=True)
            cache_read_ok = cache_read_ok and all(torch.isfinite(value).all() for value in (weight_pixel, weight_token, q20_gt))

            prediction = backend.forward_normalized(image, "transmission")
            q20_prediction = q20_prediction_features(backend, prediction)
            base_bundle, q20_loss, _losses, scalars = c1_losses(
                prediction, image, target, weight_pixel, weight_token, q20_prediction, q20_gt,
                args.local_coefficient, args.keep_coefficient,
            )
            base_grad = torch.autograd.grad(base_bundle, prediction, retain_graph=True)[0]
            q_grad = torch.autograd.grad(q20_loss, prediction)[0]
            if not torch.isfinite(base_grad).all() or not torch.isfinite(q_grad).all():
                raise RuntimeError("Non-finite output gradient")

            upcoming_step = global_step + 1
            measure = (
                micro_step % args.gradient_accumulation == 0
                and (upcoming_step == 1 or upcoming_step % args.gradient_measure_every == 0)
            )
            if args.mode == "smoke":
                measure = True
            if measure:
                measured_base_norm = global_norm(base_grad)
                measured_q_norm = global_norm(q_grad)
                if measured_base_norm <= 0 or measured_q_norm <= 0:
                    raise RuntimeError("Invalid measured base/Q20 gradient norm")
                if epoch_index == 0:
                    target_ratio = 0.15 * min(1.0, upcoming_step / max(1, steps_per_epoch))
                else:
                    target_ratio = 0.20
                desired = target_ratio * measured_base_norm / measured_q_norm
                desired = min(args.lambda_q_max, max(args.lambda_q_min, desired))
                lambda_q = args.lambda_q_ema * lambda_q + (1 - args.lambda_q_ema) * desired
                lambda_q = min(args.lambda_q_max, max(args.lambda_q_min, lambda_q))
                actual_ratio = lambda_q * measured_q_norm / measured_base_norm

            if args.mode == "smoke" and not q_path_nonzero:
                q_parameter_grads = torch.autograd.grad(
                    prediction, parameters[:1], grad_outputs=q_grad, retain_graph=True, allow_unused=True
                )
                q_parameter_norm = math.sqrt(sum(
                    float(value.float().square().sum()) for value in q_parameter_grads if value is not None
                ))
                q_path_nonzero = q_parameter_norm > 0 and math.isfinite(q_parameter_norm)

            group_start = (micro_step // args.gradient_accumulation) * args.gradient_accumulation
            group_size = min(args.gradient_accumulation, len(train_loader) - group_start)
            combined = (base_grad + lambda_q * q_grad) / group_size
            prediction.backward(combined)
            frozen_clean = frozen_clean and all(
                value.grad is None
                for name, value in backend.transformer.named_parameters()
                if ".lora_" not in name
            ) and all(value.grad is None for value in backend.vae.parameters())
            boundary = (micro_step + 1) % args.gradient_accumulation == 0 or micro_step + 1 == len(train_loader)
            if not boundary:
                continue

            active_tensors = sync_gradients(parameters, device)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
                raise RuntimeError(f"Invalid LoRA_T gradient norm: {grad_norm}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            total_loss = float((base_bundle.detach() + lambda_q * q20_loss.detach()))
            loss_finite = loss_finite and math.isfinite(total_loss) and all(math.isfinite(value) for value in scalars.values())

            local_memory = torch.tensor([
                torch.cuda.memory_allocated(device) / 2**30,
                torch.cuda.memory_reserved(device) / 2**30,
                torch.cuda.max_memory_allocated(device) / 2**30,
            ], device=device)
            gathered = [torch.zeros_like(local_memory) for _ in range(world_size())]
            dist.all_gather(gathered, local_memory)
            rank_memory = [
                {"rank": index, "allocated_gib": float(value[0]), "reserved_gib": float(value[1]), "peak_gib": float(value[2])}
                for index, value in enumerate(gathered)
            ]
            if is_main:
                memory_history.append(rank_memory)
                record = {
                    "step": global_step,
                    "epoch": epoch_index + 1,
                    "micro_step": micro_step,
                    "ids_by_rank_batch0": batch["id"][0],
                    "total_loss": total_loss,
                    **scalars,
                    "lambda_q": lambda_q,
                    "q20_gradient_ratio": actual_ratio,
                    "base_output_grad_norm": measured_base_norm,
                    "q20_output_grad_norm": measured_q_norm,
                    "lora_grad_norm": float(grad_norm),
                    "active_lora_gradient_tensors": active_tensors,
                    "lr": scheduler.get_last_lr()[0],
                    "memory": rank_memory,
                    "elapsed_seconds": time.monotonic() - started,
                }
                append_jsonl(args.run_dir / "logs/train.jsonl", record)
                print(json.dumps(record), flush=True)

            if global_step >= total_steps:
                stop = True
                break

        resume_micro_step = 0
        completed_epochs = epoch_index + 1
        epoch_finished = last_micro + 1 == len(train_loader)
        do_validate = (args.mode == "smoke" and global_step >= total_steps and not args.resume_verification) or (
            args.mode == "train" and (epoch_finished or global_step >= total_steps)
        )
        if dist.is_initialized():
            dist.barrier()
        if is_main and do_validate:
            label = "smoke_step_000005" if args.mode == "smoke" else f"epoch_{completed_epochs:04d}"
            last_validation = validate(
                backend, val_loader, device, args.seed, args.run_dir / "validation", label
            )
            validation_record = {
                "step": global_step,
                "epoch": completed_epochs,
                "label": label,
                **last_validation,
            }
            append_jsonl(args.run_dir / "logs/validation.jsonl", validation_record)
            current_psnr = float(last_validation["means"]["psnr"])
            improved = current_psnr > best_psnr
            best_psnr = max(best_psnr, current_psnr)
            epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
            top_records = save_epoch_checkpoint(
                args.run_dir, completed_epochs, global_step,
                epoch_index if not epoch_finished else epoch_index + 1,
                0 if epoch_finished else last_micro + 1,
                backend, optimizer, scheduler, lambda_q, last_validation, top_records,
                epochs_without_improvement, smoke=args.mode == "smoke",
            )
            print(json.dumps(validation_record), flush=True)
        if dist.is_initialized():
            dist.barrier()

        if args.mode == "train":
            control = torch.tensor([
                epochs_without_improvement if is_main else 0,
                1 if is_main and completed_epochs >= args.minimum_epochs and epochs_without_improvement >= args.early_stop_patience else 0,
            ], device=device, dtype=torch.int64)
            dist.broadcast(control, src=0)
            epochs_without_improvement = int(control[0])
            if int(control[1]):
                stopped_early = True
                stop = True
        if stop:
            break

    parameter_delta = parameter_sample_delta(before_samples, parameters)
    delta_tensor = torch.tensor([parameter_delta], device=device)
    gathered_delta = [torch.zeros_like(delta_tensor) for _ in range(world_size())]
    dist.all_gather(gathered_delta, delta_tensor)
    parameter_deltas = [float(value) for value in gathered_delta]

    if args.resume_verification:
        if global_step != 6 or not resumed:
            raise RuntimeError(f"Resume verification ended at step {global_step}, resumed={resumed}")
        if is_main:
            first_path = args.run_dir / "smoke_first_pass.json"
            first = json.loads(first_path.read_text(encoding="utf-8"))
            first["checks"]["checkpoint_resume_one_step"] = True
            first["resume_verification"] = {"loaded_step": 5, "continued_to_step": 6}
            first["passed"] = all(first["checks"].values())
            (args.run_dir / "SMOKE_ACCEPTANCE.json").write_text(json.dumps(first, indent=2) + "\n")
            if not first["passed"]:
                raise RuntimeError(f"Smoke acceptance failed: {first['checks']}")
    elif args.mode == "smoke" and is_main:
        if last_validation is None:
            raise RuntimeError("Smoke validation did not run")
        memory_stable = True
        if len(memory_history) >= 3:
            for gpu_rank in range(world_size()):
                post_warmup = [step[gpu_rank]["reserved_gib"] for step in memory_history[1:]]
                if max(post_warmup) - min(post_warmup) > 1.0:
                    memory_stable = False
        checks = {
            "four_gpus_participated": len(memory_history) == 5 and all(len(step) == 4 for step in memory_history),
            "gpu_memory_stable": memory_stable,
            "backbone_vae_teacher_frozen": frozen_clean,
            "lora_nonzero_gradient_and_changed": all(value > 0 for value in parameter_deltas),
            "q20_gradient_reaches_lora": q_path_nonzero,
            "q20_gt_loaded_from_cache": cache_read_ok,
            "dq_dolp_s_w_finite": True,
            "all_loss_components_finite": loss_finite,
            "q20_gradient_ratio_recorded": actual_ratio > 0 and math.isfinite(actual_ratio),
            "checkpoint_resume_one_step": False,
            "prediction_png_and_metrics": bool(list((args.run_dir / "validation/predictions/smoke_step_000005").glob("*.png")))
                and math.isfinite(float(last_validation["means"]["psnr"]))
                and math.isfinite(float(last_validation["means"]["ssim"])),
        }
        first = {
            "passed_before_resume_check": all(value for key, value in checks.items() if key != "checkpoint_resume_one_step"),
            "checks": checks,
            "global_updates": global_step,
            "images": train_ids,
            "parameter_sample_delta_by_rank": parameter_deltas,
            "memory_history": memory_history,
            "validation": last_validation,
            "q20_gradient_ratio": actual_ratio,
        }
        (args.run_dir / "smoke_first_pass.json").write_text(json.dumps(first, indent=2) + "\n")
        if not first["passed_before_resume_check"]:
            raise RuntimeError(f"Smoke first pass failed: {checks}")
    elif args.mode == "train" and is_main:
        write_summary(args.run_dir, args, global_step, completed_epochs, best_psnr, stopped_early)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
