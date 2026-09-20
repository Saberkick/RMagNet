"""Stage 2: fine-tune the WindowSeat transmission LoRA on I -> T pairs."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import bitsandbytes as bnb
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .qwen_backend import QwenSharedBackend
from .stage1_train import (
    DEFAULT_DATA,
    append_jsonl,
    discover_ids,
    edge_l1,
    image_tensor,
    make_scheduler,
    rank,
    seed_everything,
    setup_distributed,
    ssim,
    sync_gradients,
    sync_initial_parameters,
    trainable_parameters,
    world_size,
)


DEFAULT_RUN = Path("/share/linmingheng-local/xuke/RMagNet/runs/stage2_transmission_r128")


class TransmissionDataset(Dataset):
    def __init__(self, root: Path, sample_ids: list[str], augment: bool) -> None:
        self.root = root
        self.sample_ids = sample_ids
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = self.sample_ids[index]
        image = image_tensor(self.root / "blended" / f"{sample_id}.png")
        target = image_tensor(self.root / "transmission_layer" / f"{sample_id}.png")
        if image.shape != (3, 384, 512) or target.shape != image.shape:
            raise ValueError(f"Unexpected pair shape for {sample_id}: {image.shape}, {target.shape}")
        if self.augment and torch.rand(()) < 0.5:
            image = image.flip(-1)
            target = target.flip(-1)
        return {"id": sample_id, "image": image, "target": target}


def transmission_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ssim_weight: float,
    edge_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = ((prediction.float() + 1) * 0.5).clamp(0, 1)
    target = ((target.float() + 1) * 0.5).clamp(0, 1)
    l1 = F.l1_loss(prediction, target)
    ssim_loss = 1 - ssim(prediction, target)
    edge = edge_l1(prediction, target)
    total = l1 + ssim_weight * ssim_loss + edge_weight * edge
    return total, {
        "l1": float(l1.detach()),
        "ssim_loss": float(ssim_loss.detach()),
        "edge": float(edge.detach()),
    }


def adapter_state(backend: QwenSharedBackend) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in backend.transformer.named_parameters()
        if parameter.requires_grad
    }


def save_checkpoint(
    run_dir: Path,
    step: int,
    epoch: int,
    next_micro_step: int,
    backend: QwenSharedBackend,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
) -> Path:
    checkpoint = run_dir / f"checkpoint-{step:07d}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    safetensors.torch.save_file(
        adapter_state(backend), checkpoint / "transmission_lora.safetensors"
    )
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "next_micro_step": next_micro_step,
            "scheduler": scheduler.state_dict(),
            "optimizer_state_saved": False,
            "optimizer_resume_policy": "reinitialize PagedAdamW8bit",
        },
        checkpoint / "trainer_state.pt",
    )
    (checkpoint / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8"
    )
    (run_dir / "last_checkpoint.txt").write_text(str(checkpoint) + "\n", encoding="utf-8")
    for old in sorted(run_dir.glob("checkpoint-*"))[: -args.keep_checkpoints]:
        shutil.rmtree(old)
    return checkpoint


def load_checkpoint(
    checkpoint: Path,
    backend: QwenSharedBackend,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, int, int]:
    weights = safetensors.torch.load_file(
        checkpoint / "transmission_lora.safetensors", device=str(device)
    )
    _, unexpected = backend.transformer.load_state_dict(weights, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected T LoRA keys: {unexpected[:5]}")
    state = torch.load(
        checkpoint / "trainer_state.pt", map_location=device, weights_only=False
    )
    scheduler.load_state_dict(state["scheduler"])
    return int(state["step"]), int(state["epoch"]), int(state["next_micro_step"])


@torch.no_grad()
def validate(
    backend: QwenSharedBackend,
    loader: DataLoader,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    backend.transformer.eval()
    backend.vae.eval()
    l1_values, psnr_values, ssim_values = [], [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        prediction = backend.forward_normalized(image, "transmission")
        prediction = ((prediction.float() + 1) * 0.5).clamp(0, 1)
        target = ((target.float() + 1) * 0.5).clamp(0, 1)
        mse = F.mse_loss(prediction, target)
        l1_values.append(float(F.l1_loss(prediction, target)))
        psnr_values.append(float(-10 * torch.log10(mse.clamp_min(1e-12))))
        ssim_values.append(float(ssim(prediction, target)))
    backend.transformer.train()
    backend.vae.eval()
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state, device)
    return {
        "val_l1": sum(l1_values) / len(l1_values),
        "val_psnr": sum(psnr_values) / len(psnr_values),
        "val_ssim": sum(ssim_values) / len(ssim_values),
    }


def maybe_save_best(
    run_dir: Path,
    metrics: dict[str, float],
    step: int,
    backend: QwenSharedBackend,
) -> bool:
    record_path = run_dir / "best_metrics.json"
    previous = json.loads(record_path.read_text()) if record_path.is_file() else None
    if previous is not None and metrics["val_l1"] >= float(previous["val_l1"]):
        return False
    safetensors.torch.save_file(
        adapter_state(backend), run_dir / "best_transmission_lora.safetensors"
    )
    record_path.write_text(
        json.dumps({**metrics, "step": step, "criterion": "minimum val_l1"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--val-ids", default="11,12,17")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = setup_distributed()
    is_main = rank() == 0
    seed_everything(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    all_ids = discover_ids(args.data_root)
    val_ids = [item for item in args.val_ids.split(",") if item]
    train_ids = [item for item in all_ids if item not in val_ids]
    train_data = TransmissionDataset(args.data_root, train_ids, True)
    val_data = TransmissionDataset(args.data_root, val_ids, False)
    sampler = (
        DistributedSampler(train_data, shuffle=True, seed=args.seed)
        if dist.is_initialized()
        else None
    )
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)

    backend = QwenSharedBackend.from_local(device)
    if args.gradient_checkpointing:
        backend.transformer.enable_gradient_checkpointing()
    backend.set_trainable_branch("transmission")
    backend.transformer.train()
    backend.vae.eval()
    parameters = trainable_parameters(backend)
    sync_initial_parameters(parameters)
    seed_everything(args.seed + rank())

    optimizer = bnb.optim.PagedAdamW8bit(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation)
    planned_steps = args.epochs * steps_per_epoch
    total_steps = min(planned_steps, args.max_steps) if args.max_steps else planned_steps
    scheduler = make_scheduler(optimizer, min(args.warmup_steps, total_steps), total_steps)

    global_step, start_epoch, resume_micro_step = 0, 0, 0
    checkpoint = None
    if args.resume == "auto" and (args.run_dir / "last_checkpoint.txt").is_file():
        checkpoint = Path((args.run_dir / "last_checkpoint.txt").read_text().strip())
    elif args.resume not in ("auto", "none"):
        checkpoint = Path(args.resume)
    if checkpoint:
        global_step, start_epoch, resume_micro_step = load_checkpoint(
            checkpoint, backend, scheduler, device
        )
        if dist.is_initialized():
            dist.barrier()

    if is_main:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        metadata = {
            "args": vars(args),
            "git_commit": commit,
            "world_size": world_size(),
            "train_ids": train_ids,
            "val_ids": val_ids,
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "optimizer": "bitsandbytes.PagedAdamW8bit",
            "optimizer_steps_per_epoch": steps_per_epoch,
            "planned_optimizer_steps": total_steps,
            "initialization": "official WindowSeat rank-128 transmission LoRA",
        }
        (args.run_dir / "run_config.json").write_text(
            json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2, default=str), flush=True)

    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    stop = global_step >= total_steps
    last_micro_step = -1
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        if sampler is not None:
            sampler.set_epoch(epoch)
        for micro_step, batch in enumerate(loader):
            if epoch == start_epoch and micro_step < resume_micro_step:
                continue
            last_micro_step = micro_step
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            prediction = backend.forward_normalized(image, "transmission")
            loss, parts = transmission_loss(
                prediction, target, args.ssim_weight, args.edge_weight
            )
            (loss / args.gradient_accumulation).backward()
            boundary = (micro_step + 1) % args.gradient_accumulation == 0 or micro_step + 1 == len(loader)
            if not boundary:
                continue
            active_tensors = sync_gradients(parameters, device)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            values = torch.tensor(
                [
                    float(loss.detach()),
                    parts["l1"],
                    parts["ssim_loss"],
                    parts["edge"],
                    float(grad_norm),
                ],
                device=device,
            )
            if dist.is_initialized():
                dist.all_reduce(values)
                values.div_(world_size())
            if is_main:
                record = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": float(values[0]),
                    "l1": float(values[1]),
                    "ssim_loss": float(values[2]),
                    "edge": float(values[3]),
                    "grad_norm": float(values[4]),
                    "active_gradient_tensors": active_tensors,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                }
                append_jsonl(args.run_dir / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)

            do_validate = args.validate_every > 0 and global_step % args.validate_every == 0
            if do_validate:
                if dist.is_initialized():
                    dist.barrier()
                if is_main:
                    metrics = validate(backend, val_loader, device, args.seed)
                    metrics.update({"step": global_step, "epoch": epoch, "kind": "validation"})
                    metrics["best_updated"] = maybe_save_best(
                        args.run_dir, metrics, global_step, backend
                    )
                    append_jsonl(args.run_dir / "metrics.jsonl", metrics)
                    print(json.dumps(metrics), flush=True)
                if dist.is_initialized():
                    dist.barrier()
            if is_main and args.save_every > 0 and global_step % args.save_every == 0:
                saved = save_checkpoint(
                    args.run_dir,
                    global_step,
                    epoch,
                    micro_step + 1,
                    backend,
                    scheduler,
                    args,
                )
                print(f"saved {saved}", flush=True)
            if dist.is_initialized():
                dist.barrier()
            if global_step >= total_steps:
                stop = True
                break
        resume_micro_step = 0

    if dist.is_initialized():
        dist.barrier()
    if is_main:
        final_dir = args.run_dir / f"checkpoint-{global_step:07d}"
        if not final_dir.exists():
            final_dir = save_checkpoint(
                args.run_dir,
                global_step,
                min(args.epochs - 1, epoch),
                last_micro_step + 1,
                backend,
                scheduler,
                args,
            )
        metrics = validate(backend, val_loader, device, args.seed)
        metrics.update({"step": global_step, "kind": "final", "checkpoint": str(final_dir)})
        metrics["best_updated"] = maybe_save_best(args.run_dir, metrics, global_step, backend)
        append_jsonl(args.run_dir / "metrics.jsonl", metrics)
        print(json.dumps(metrics), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
