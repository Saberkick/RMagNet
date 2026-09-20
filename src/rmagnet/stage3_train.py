"""Stage 3: train a latent I/T/R mixer and LoRA_Fuse toward transmission GT."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import bitsandbytes as bnb
import safetensors.torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .conditioning import LatentFusionMixer
from .qwen_backend import QwenSharedBackend
from .stage1_train import (
    DEFAULT_DATA, append_jsonl, discover_ids, image_tensor, make_scheduler, rank,
    seed_everything, setup_distributed, sync_gradients, sync_initial_parameters,
    world_size,
)
from .stage2_train import transmission_loss


ROOT = Path("/share/linmingheng-local/xuke/RMagNet")
DEFAULT_CACHE = ROOT / "cache/stage3_candidates"
DEFAULT_RUN = ROOT / "runs/stage3_fusion_r8"


class FusionDataset(Dataset):
    def __init__(self, data_root: Path, cache_root: Path, ids: list[str], augment: bool):
        self.data_root, self.cache_root, self.ids, self.augment = data_root, cache_root, ids, augment

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = self.ids[index]
        tensors = {
            "image": image_tensor(self.data_root / "blended" / f"{sample_id}.png"),
            "target": image_tensor(self.data_root / "transmission_layer" / f"{sample_id}.png"),
            "transmission": image_tensor(self.cache_root / "transmission" / f"{sample_id}.png"),
            "reflection": image_tensor(self.cache_root / "reflection" / f"{sample_id}.png"),
        }
        shapes = {tuple(value.shape) for value in tensors.values()}
        if shapes != {(3, 384, 512)}:
            raise ValueError(f"Unexpected shapes for {sample_id}: {shapes}")
        if self.augment and torch.rand(()) < 0.5:
            tensors = {key: value.flip(-1) for key, value in tensors.items()}
        return {"id": sample_id, **tensors}


def fuse_state(backend: QwenSharedBackend) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in backend.transformer.named_parameters()
        if value.requires_grad
    }


def mixer_state(mixer: LatentFusionMixer) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous() for name, value in mixer.state_dict().items()}


def save_checkpoint(run_dir, step, epoch, next_micro_step, backend, mixer, scheduler, args):
    checkpoint = run_dir / f"checkpoint-{step:07d}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    safetensors.torch.save_file(fuse_state(backend), checkpoint / "fusion_lora.safetensors")
    safetensors.torch.save_file(mixer_state(mixer), checkpoint / "latent_mixer.safetensors")
    torch.save({
        "step": step, "epoch": epoch, "next_micro_step": next_micro_step,
        "scheduler": scheduler.state_dict(), "optimizer_state_saved": False,
        "optimizer_resume_policy": "reinitialize PagedAdamW8bit",
    }, checkpoint / "trainer_state.pt")
    (checkpoint / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    (run_dir / "last_checkpoint.txt").write_text(str(checkpoint) + "\n")
    for old in sorted(run_dir.glob("checkpoint-*"))[:-args.keep_checkpoints]:
        shutil.rmtree(old)
    return checkpoint


def load_checkpoint(checkpoint, backend, mixer, scheduler, device):
    fuse = safetensors.torch.load_file(checkpoint / "fusion_lora.safetensors", device=str(device))
    _, unexpected = backend.transformer.load_state_dict(fuse, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected Fuse LoRA keys: {unexpected[:5]}")
    mixer.load_state_dict(safetensors.torch.load_file(checkpoint / "latent_mixer.safetensors", device=str(device)))
    state = torch.load(checkpoint / "trainer_state.pt", map_location=device, weights_only=False)
    scheduler.load_state_dict(state["scheduler"])
    return int(state["step"]), int(state["epoch"]), int(state["next_micro_step"])


def forward(backend, mixer, batch, device):
    image = batch["image"].to(device, non_blocking=True)
    transmission = batch["transmission"].to(device, non_blocking=True)
    reflection = batch["reflection"].to(device, non_blocking=True)
    z_i = backend.encode_frozen(image)
    z_t = backend.encode_frozen(transmission)
    z_r = backend.encode_frozen(reflection)
    return backend.forward_fusion_latent(mixer(z_i, z_t, z_r))


@torch.no_grad()
def validate(backend, mixer, loader, device, seed):
    cpu_state, cuda_state = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    backend.transformer.eval(); mixer.eval()
    totals = {"val_l1": 0.0, "val_psnr": 0.0, "val_ssim": 0.0}
    from .stage1_train import ssim
    import torch.nn.functional as F
    for batch in loader:
        pred = ((forward(backend, mixer, batch, device).float() + 1) * 0.5).clamp(0, 1)
        target = ((batch["target"].to(device).float() + 1) * 0.5).clamp(0, 1)
        mse = F.mse_loss(pred, target)
        totals["val_l1"] += float(F.l1_loss(pred, target))
        totals["val_psnr"] += float(-10 * torch.log10(mse.clamp_min(1e-12)))
        totals["val_ssim"] += float(ssim(pred, target))
    totals = {key: value / len(loader) for key, value in totals.items()}
    backend.transformer.train(); mixer.train()
    torch.set_rng_state(cpu_state); torch.cuda.set_rng_state(cuda_state, device)
    return totals


def maybe_save_best(run_dir, metrics, step, backend, mixer):
    path = run_dir / "best_metrics.json"
    previous = json.loads(path.read_text()) if path.is_file() else None
    if previous is not None and metrics["val_l1"] >= previous["val_l1"]:
        return False
    safetensors.torch.save_file(fuse_state(backend), run_dir / "best_fusion_lora.safetensors")
    safetensors.torch.save_file(mixer_state(mixer), run_dir / "best_latent_mixer.safetensors")
    path.write_text(json.dumps({**metrics, "step": step, "criterion": "minimum val_l1"}, indent=2) + "\n")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fusion-rank", type=int, default=8)
    parser.add_argument("--mixer-width", type=int, default=64)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--val-ids", default="11,12,17")
    parser.add_argument("--train-ids", default="all", help="comma list for smoke tests")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", default="auto")
    return parser.parse_args()


def main():
    args = parse_args(); device = setup_distributed(); is_main = rank() == 0
    seed_everything(args.seed); args.run_dir.mkdir(parents=True, exist_ok=True)
    all_ids = discover_ids(args.data_root)
    val_ids = [x for x in args.val_ids.split(",") if x]
    train_ids = ([x for x in all_ids if x not in val_ids] if args.train_ids == "all"
                 else [x.strip() for x in args.train_ids.split(",") if x.strip()])
    selected_ids = train_ids + val_ids
    unknown = sorted(set(selected_ids) - set(all_ids), key=int)
    if unknown:
        raise ValueError(f"Unknown IDs: {unknown}")
    missing = [x for x in selected_ids for role in ("transmission", "reflection") if not (args.cache_root / role / f"{x}.png").is_file()]
    if missing:
        raise FileNotFoundError(f"Stage 3 cache incomplete; first missing ID: {missing[0]}")
    train_data = FusionDataset(args.data_root, args.cache_root, train_ids, True)
    val_data = FusionDataset(args.data_root, args.cache_root, val_ids, False)
    sampler = DistributedSampler(train_data, shuffle=True, seed=args.seed) if dist.is_initialized() else None
    loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)

    backend = QwenSharedBackend.from_local(device, fusion_rank=args.fusion_rank)
    if args.gradient_checkpointing: backend.transformer.enable_gradient_checkpointing()
    backend.set_trainable_fusion(); backend.transformer.train(); backend.vae.eval()
    mixer = LatentFusionMixer(channels=backend.vae.config.z_dim, width=args.mixer_width).to(device)
    parameters = [p for p in backend.transformer.parameters() if p.requires_grad] + list(mixer.parameters())
    sync_initial_parameters(parameters); seed_everything(args.seed + rank())
    optimizer = bnb.optim.PagedAdamW8bit(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation)
    planned = args.epochs * steps_per_epoch; total_steps = min(planned, args.max_steps) if args.max_steps else planned
    scheduler = make_scheduler(optimizer, min(args.warmup_steps, total_steps), total_steps)
    global_step = start_epoch = resume_micro_step = 0
    checkpoint = None
    if args.resume == "auto" and (args.run_dir / "last_checkpoint.txt").is_file(): checkpoint = Path((args.run_dir / "last_checkpoint.txt").read_text().strip())
    elif args.resume not in ("auto", "none"): checkpoint = Path(args.resume)
    if checkpoint: global_step, start_epoch, resume_micro_step = load_checkpoint(checkpoint, backend, mixer, scheduler, device)
    if dist.is_initialized(): dist.barrier()

    if is_main:
        metadata = {"args": vars(args), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "world_size": world_size(), "train_ids": train_ids, "val_ids": val_ids, "trainable_parameters": sum(p.numel() for p in parameters), "architecture": "frozen VAE/DiT + latent I/T/R mixer + LoRA_Fuse", "optimizer": "PagedAdamW8bit (state rebuilt on resume)"}
        (args.run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
        print(json.dumps(metadata, indent=2, default=str), flush=True)

    optimizer.zero_grad(set_to_none=True); started = time.monotonic(); stop = global_step >= total_steps; last_micro = -1
    for epoch in range(start_epoch, args.epochs):
        if stop: break
        if sampler is not None: sampler.set_epoch(epoch)
        for micro, batch in enumerate(loader):
            if epoch == start_epoch and micro < resume_micro_step: continue
            last_micro = micro
            pred = forward(backend, mixer, batch, device)
            loss, parts = transmission_loss(pred, batch["target"].to(device), args.ssim_weight, args.edge_weight)
            (loss / args.gradient_accumulation).backward()
            boundary = (micro + 1) % args.gradient_accumulation == 0 or micro + 1 == len(loader)
            if not boundary: continue
            active = sync_gradients(parameters, device)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
            values = torch.tensor([float(loss.detach()), parts["l1"], parts["ssim_loss"], parts["edge"], float(grad_norm)], device=device)
            if dist.is_initialized(): dist.all_reduce(values); values.div_(world_size())
            if is_main:
                record = {"step": global_step, "epoch": epoch, "loss": float(values[0]), "l1": float(values[1]), "ssim_loss": float(values[2]), "edge": float(values[3]), "grad_norm": float(values[4]), "active_gradient_tensors": active, "lr": scheduler.get_last_lr()[0], "elapsed_seconds": time.monotonic() - started, "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30, "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
                append_jsonl(args.run_dir / "metrics.jsonl", record); print(json.dumps(record), flush=True)
            if args.validate_every > 0 and global_step % args.validate_every == 0:
                if dist.is_initialized(): dist.barrier()
                if is_main:
                    metrics = validate(backend, mixer, val_loader, device, args.seed); metrics.update({"step": global_step, "epoch": epoch, "kind": "validation"}); metrics["best_updated"] = maybe_save_best(args.run_dir, metrics, global_step, backend, mixer); append_jsonl(args.run_dir / "metrics.jsonl", metrics); print(json.dumps(metrics), flush=True)
                if dist.is_initialized(): dist.barrier()
            if is_main and args.save_every > 0 and global_step % args.save_every == 0:
                print(f"saved {save_checkpoint(args.run_dir, global_step, epoch, micro + 1, backend, mixer, scheduler, args)}", flush=True)
            if dist.is_initialized(): dist.barrier()
            if global_step >= total_steps: stop = True; break
        resume_micro_step = 0

    if dist.is_initialized(): dist.barrier()
    if is_main:
        final_dir = args.run_dir / f"checkpoint-{global_step:07d}"
        if not final_dir.exists(): final_dir = save_checkpoint(args.run_dir, global_step, min(args.epochs - 1, epoch), last_micro + 1, backend, mixer, scheduler, args)
        metrics = validate(backend, mixer, val_loader, device, args.seed); metrics.update({"step": global_step, "kind": "final", "checkpoint": str(final_dir)}); metrics["best_updated"] = maybe_save_best(args.run_dir, metrics, global_step, backend, mixer); append_jsonl(args.run_dir / "metrics.jsonl", metrics); print(json.dumps(metrics), flush=True)
    if dist.is_initialized(): dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
