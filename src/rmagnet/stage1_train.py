"""Stage 1: train the reflection LoRA on aligned I -> R image pairs.

The Qwen DiT and VAE stay frozen. In distributed runs, every rank owns one
quantized backbone and only the small reflection-adapter gradients are reduced.
This avoids DDP broadcasting the 12.5B frozen backbone.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .qwen_backend import QwenSharedBackend


DEFAULT_DATA = Path(
    "/share/linmingheng-local/xuke/datasets/rmagnet_stage1_512x384"
)


def distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def setup_distributed() -> torch.device:
    if distributed():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=device)
        return device
    torch.cuda.set_device(0)
    return torch.device("cuda", 0)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1.0)


class ReflectionDataset(Dataset):
    def __init__(
        self,
        root: Path,
        sample_ids: list[str],
        augment: bool,
        expected_size: tuple[int, int],
    ) -> None:
        self.root = root
        self.sample_ids = sample_ids
        self.augment = augment
        self.expected_size = expected_size

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = self.sample_ids[index]
        image = image_tensor(self.root / "blended" / f"{sample_id}.png")
        target = image_tensor(self.root / "reflection_layer" / f"{sample_id}.png")
        if tuple(reversed(image.shape[-2:])) != self.expected_size:
            raise ValueError(f"Unexpected input size for {sample_id}: {image.shape}")
        if image.shape != target.shape:
            raise ValueError(f"Pair shape mismatch for {sample_id}")
        if self.augment and torch.rand(()) < 0.5:
            image = image.flip(-1)
            target = target.flip(-1)
        return {"id": sample_id, "image": image, "target": target}


def discover_ids(root: Path) -> list[str]:
    roles = ["blended", "reflection_layer", "transmission_layer"]
    role_ids = []
    for role in roles:
        folder = root / role
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        role_ids.append({path.stem for path in folder.glob("*.png")})
    if not role_ids[0] or any(ids != role_ids[0] for ids in role_ids[1:]):
        raise RuntimeError("Dataset roles do not contain identical sample IDs")
    return sorted(role_ids[0], key=int)


def ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    mu_x = F.avg_pool2d(prediction, 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(target, 11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(prediction * prediction, 11, 1, 5) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, 11, 1, 5) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, 11, 1, 5) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    value = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return value.mean()


def edge_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dx_p = prediction[..., :, 1:] - prediction[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_p = prediction[..., 1:, :] - prediction[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    return 0.5 * (F.l1_loss(dx_p.float(), dx_t.float()) + F.l1_loss(dy_p.float(), dy_t.float()))


def reflection_loss(
    prediction: torch.Tensor, target: torch.Tensor, ssim_weight: float, edge_weight: float
) -> tuple[torch.Tensor, dict[str, float]]:
    pred01 = ((prediction.float() + 1.0) * 0.5).clamp(0, 1)
    target01 = ((target.float() + 1.0) * 0.5).clamp(0, 1)
    l1 = F.l1_loss(pred01, target01)
    ssim_loss = 1.0 - ssim(pred01, target01)
    edge = edge_l1(pred01, target01)
    total = l1 + ssim_weight * ssim_loss + edge_weight * edge
    return total, {
        "l1": float(l1.detach()),
        "ssim_loss": float(ssim_loss.detach()),
        "edge": float(edge.detach()),
    }


def trainable_parameters(backend: QwenSharedBackend) -> list[torch.nn.Parameter]:
    return [parameter for parameter in backend.transformer.parameters() if parameter.requires_grad]


def sync_initial_parameters(parameters: list[torch.nn.Parameter]) -> None:
    if dist.is_initialized():
        for parameter in parameters:
            dist.broadcast(parameter.data, src=0)


def sync_gradients(parameters: list[torch.nn.Parameter], device: torch.device) -> int:
    active = torch.tensor(
        [parameter.grad is not None for parameter in parameters],
        device=device,
        dtype=torch.int32,
    )
    if dist.is_initialized():
        active_min = active.clone()
        active_max = active.clone()
        dist.all_reduce(active_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(active_max, op=dist.ReduceOp.MAX)
        if not torch.equal(active_min, active_max):
            raise RuntimeError("Ranks disagree about which Reflection LoRA tensors have gradients")
        active = active_min
    active_count = int(active.sum())
    if active_count == 0:
        raise RuntimeError("No Reflection LoRA gradients were produced")
    if dist.is_initialized():
        for enabled, parameter in zip(active.tolist(), parameters):
            if enabled:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(world_size())
    return active_count


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
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
) -> Path:
    checkpoint = run_dir / f"checkpoint-{step:07d}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    safetensors.torch.save_file(adapter_state(backend), checkpoint / "reflection_lora.safetensors")
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "next_micro_step": next_micro_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint / "trainer_state.pt",
    )
    (checkpoint / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8"
    )
    (run_dir / "last_checkpoint.txt").write_text(str(checkpoint) + "\n", encoding="utf-8")
    checkpoints = sorted(run_dir.glob("checkpoint-*"))
    for old in checkpoints[: -args.keep_checkpoints]:
        shutil.rmtree(old)
    return checkpoint


def resume_checkpoint(
    path: Path,
    backend: QwenSharedBackend,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, int, int]:
    weights = safetensors.torch.load_file(path / "reflection_lora.safetensors", device=str(device))
    _, unexpected = backend.transformer.load_state_dict(weights, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected resumed LoRA keys: {unexpected[:5]}")
    state = torch.load(path / "trainer_state.pt", map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return (
        int(state["step"]),
        int(state["epoch"]),
        int(state.get("next_micro_step", 0)),
    )


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def maybe_save_best(
    run_dir: Path,
    metrics: dict[str, float],
    step: int,
    backend: QwenSharedBackend,
) -> bool:
    record_path = run_dir / "best_metrics.json"
    previous = None
    if record_path.is_file():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
    if previous is not None and metrics["val_l1"] >= float(previous["val_l1"]):
        return False
    safetensors.torch.save_file(
        adapter_state(backend), run_dir / "best_reflection_lora.safetensors"
    )
    record = {**metrics, "step": step, "criterion": "minimum val_l1"}
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return True


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
        prediction = backend.forward_normalized(image, "reflection")
        pred01 = ((prediction.float() + 1) * 0.5).clamp(0, 1)
        target01 = ((target.float() + 1) * 0.5).clamp(0, 1)
        mse = F.mse_loss(pred01, target01)
        l1_values.append(float(F.l1_loss(pred01, target01)))
        psnr_values.append(float(-10 * torch.log10(mse.clamp_min(1e-12))))
        ssim_values.append(float(ssim(pred01, target01)))
    backend.transformer.train()
    backend.vae.eval()
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state, device)
    return {
        "val_l1": sum(l1_values) / len(l1_values),
        "val_psnr": sum(psnr_values) / len(psnr_values),
        "val_ssim": sum(ssim_values) / len(ssim_values),
    }


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no early stop")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--reflection-rank", type=int, default=8)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--val-ids", default="11,12,17")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", default="auto", help="auto, none, or checkpoint directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = setup_distributed()
    is_main = rank() == 0
    seed_everything(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    all_ids = discover_ids(args.data_root)
    val_ids = [item for item in args.val_ids.split(",") if item]
    unknown = sorted(set(val_ids) - set(all_ids), key=int)
    if unknown:
        raise ValueError(f"Unknown validation IDs: {unknown}")
    train_ids = [sample_id for sample_id in all_ids if sample_id not in val_ids]
    if not train_ids or not val_ids:
        raise ValueError("Both training and validation splits must be non-empty")

    train_dataset = ReflectionDataset(args.data_root, train_ids, True, (512, 384))
    val_dataset = ReflectionDataset(args.data_root, val_ids, False, (512, 384))
    sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed) if distributed() else None
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # All ranks initialize identical adapter weights before rank-specific data RNG.
    backend = QwenSharedBackend.from_local(device, reflection_rank=args.reflection_rank)
    if args.gradient_checkpointing:
        backend.transformer.enable_gradient_checkpointing()
    backend.set_trainable_branch("reflection")
    backend.transformer.train()
    backend.vae.eval()
    parameters = trainable_parameters(backend)
    if not parameters:
        raise RuntimeError("No trainable Reflection LoRA parameters")
    sync_initial_parameters(parameters)
    seed_everything(args.seed + rank())

    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    optimizer_steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation)
    planned_steps = args.epochs * optimizer_steps_per_epoch
    total_steps = min(planned_steps, args.max_steps) if args.max_steps else planned_steps
    scheduler = make_scheduler(optimizer, min(args.warmup_steps, total_steps), total_steps)

    global_step, start_epoch, resume_micro_step = 0, 0, 0
    resume_path = None
    if args.resume == "auto":
        pointer = args.run_dir / "last_checkpoint.txt"
        if pointer.is_file():
            resume_path = Path(pointer.read_text(encoding="utf-8").strip())
    elif args.resume != "none":
        resume_path = Path(args.resume)
    if resume_path is not None:
        global_step, start_epoch, resume_micro_step = resume_checkpoint(
            resume_path, backend, optimizer, scheduler, device
        )
        if dist.is_initialized():
            dist.barrier()

    if is_main:
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:
            git_commit = "unknown"
        metadata = {
            "args": vars(args),
            "git_commit": git_commit,
            "world_size": world_size(),
            "train_ids": train_ids,
            "val_ids": val_ids,
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "planned_optimizer_steps": total_steps,
            "target_semantics": "45_aligned input -> 90 reflection-enhanced target",
        }
        (args.run_dir / "run_config.json").write_text(
            json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2, default=str), flush=True)

    log_path = args.run_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    stop = global_step >= total_steps
    started = time.monotonic()
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
            prediction = backend.forward_normalized(image, "reflection")
            loss, parts = reflection_loss(
                prediction, target, args.ssim_weight, args.edge_weight
            )
            (loss / args.gradient_accumulation).backward()
            boundary = (micro_step + 1) % args.gradient_accumulation == 0 or micro_step + 1 == len(loader)
            if not boundary:
                continue
            active_gradient_tensors = sync_gradients(parameters, device)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            reduced = torch.tensor(
                [float(loss.detach()), parts["l1"], parts["ssim_loss"], parts["edge"], float(grad_norm)],
                device=device,
            )
            if dist.is_initialized():
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                reduced.div_(world_size())
            if is_main:
                record = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": float(reduced[0]),
                    "l1": float(reduced[1]),
                    "ssim_loss": float(reduced[2]),
                    "edge": float(reduced[3]),
                    "grad_norm": float(reduced[4]),
                    "active_gradient_tensors": active_gradient_tensors,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                }
                append_jsonl(log_path, record)
                print(json.dumps(record), flush=True)

            should_validate = args.validate_every > 0 and global_step % args.validate_every == 0
            if should_validate:
                if dist.is_initialized():
                    dist.barrier()
                if is_main:
                    metrics = validate(backend, val_loader, device, args.seed)
                    metrics.update({"step": global_step, "epoch": epoch, "kind": "validation"})
                    metrics["best_updated"] = maybe_save_best(
                        args.run_dir, metrics, global_step, backend
                    )
                    append_jsonl(log_path, metrics)
                    print(json.dumps(metrics), flush=True)
                if dist.is_initialized():
                    dist.barrier()

            if is_main and args.save_every > 0 and global_step % args.save_every == 0:
                path = save_checkpoint(
                    args.run_dir,
                    global_step,
                    epoch,
                    micro_step + 1,
                    backend,
                    optimizer,
                    scheduler,
                    args,
                )
                print(f"saved {path}", flush=True)
            if dist.is_initialized():
                dist.barrier()
            if global_step >= total_steps:
                stop = True
                break
        resume_micro_step = 0

    if dist.is_initialized():
        dist.barrier()
    if is_main:
        final_path = save_checkpoint(
            args.run_dir,
            global_step,
            min(args.epochs - 1, epoch),
            last_micro_step + 1,
            backend,
            optimizer,
            scheduler,
            args,
        ) if not (args.run_dir / f"checkpoint-{global_step:07d}").exists() else args.run_dir / f"checkpoint-{global_step:07d}"
        metrics = validate(backend, val_loader, device, args.seed)
        metrics.update({"step": global_step, "kind": "final", "checkpoint": str(final_path)})
        metrics["best_updated"] = maybe_save_best(
            args.run_dir, metrics, global_step, backend
        )
        append_jsonl(log_path, metrics)
        print(json.dumps(metrics), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
