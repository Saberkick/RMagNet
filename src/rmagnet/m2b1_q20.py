"""M2-B1: 70-step M2 Q20 experiment with per-step 30% gradient control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import bitsandbytes as bnb
import lpips
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from .c1_l20_train import StopAtQ20, c1_losses, global_norm, move_optimizer_state
from .m1b_train import load_initial
from .m2a_data_baseline import (
    AspectGroupedDistributedSampler,
    grouped_global_batches,
    load_m2_manifest,
    save_checkpoint,
    validate,
)
from .qwen_backend import ADAPTER_NAMES, QwenSharedBackend
from .qwen_layer_probe import deterministic_encode
from .stage1_train import (
    append_jsonl,
    image_tensor,
    make_scheduler,
    rank,
    seed_everything,
    setup_distributed,
    sync_gradients,
    sync_initial_parameters,
    trainable_parameters,
    world_size,
)


ROOT = Path("/share/linmingheng-local/xuke")
PROJECT = ROOT / "RMagNet"
DEFAULT_DATA = ROOT / "datasets/rmagnet_m2_aspect"
DEFAULT_CACHE = PROJECT / "data_cache/m2a_q20"
DEFAULT_RUN = PROJECT / "runs/m2_corrected_b1_q20grad30"
DEFAULT_INITIAL = PROJECT / "runs/stage2_transmission_r128/best_transmission_lora.safetensors"
EXPECTED_INITIAL_SHA256 = "f5737d4ffb89e86874a96a02bd58a074299ca12e00ec15cac438c403a342085a"
BLOCK_INDEX = 19


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class M2Q20Dataset(Dataset):
    """Aligned M2 I/GT and cached Q20 targets with identical augmentation to M2-A."""

    def __init__(
        self,
        root: Path,
        cache_root: Path,
        records: dict[str, dict],
        cache_records: dict[str, dict],
        sample_ids: list[str],
        augment: bool,
    ) -> None:
        self.root = root
        self.cache_root = cache_root
        self.records = records
        self.cache_records = cache_records
        self.sample_ids = sample_ids
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = self.sample_ids[index]
        record = self.records[sample_id]
        cached = self.cache_records[sample_id]
        image = image_tensor(self.root / "blended" / f"{sample_id}.png")
        target = image_tensor(self.root / "transmission_layer" / f"{sample_id}.png")
        width, height = record["target_size"]
        grid_h, grid_w = cached["token_grid_hw"]
        if image.shape != (3, height, width) or target.shape != image.shape:
            raise ValueError(f"Unexpected pair shape for {sample_id}: {tuple(image.shape)}")

        with np.load(self.cache_root / cached["cache"]["weight"]) as stored:
            weight_pixel = torch.from_numpy(stored["weight_pixel"].astype(np.float32))
            weight_token = torch.from_numpy(stored["weight_token"].astype(np.float32))
        q20_gt = safetensors.torch.load_file(
            self.cache_root / cached["cache"]["gt_feature"]
        )["q20_gt"]
        expected_tokens = grid_h * grid_w
        if weight_pixel.shape != (height, width):
            raise ValueError(f"Pixel weight shape mismatch for {sample_id}: {weight_pixel.shape}")
        if weight_token.shape != (grid_h, grid_w):
            raise ValueError(f"Token weight shape mismatch for {sample_id}: {weight_token.shape}")
        if q20_gt.shape != (expected_tokens, 3072) or q20_gt.dtype != torch.bfloat16:
            raise ValueError(f"Q20 target mismatch for {sample_id}: {q20_gt.shape}/{q20_gt.dtype}")
        if not all(torch.isfinite(x).all() for x in (weight_pixel, weight_token, q20_gt)):
            raise ValueError(f"Non-finite cached data for {sample_id}")
        if abs(float(weight_pixel.mean()) - 1.0) > 2e-3:
            raise ValueError(f"Pixel weight mean differs from one for {sample_id}")

        flipped = False
        if self.augment and torch.rand(()) < 0.5:
            image = image.flip(-1)
            target = target.flip(-1)
            weight_pixel = weight_pixel.flip(-1)
            weight_token = weight_token.flip(-1)
            q20_gt = q20_gt.reshape(grid_h, grid_w, 3072).flip(1).reshape(expected_tokens, 3072)
            flipped = True
        return {
            "id": sample_id,
            "image": image,
            "target": target,
            "weight_pixel": weight_pixel,
            "weight_token": weight_token,
            "q20_gt": q20_gt,
            "token_grid": torch.tensor([grid_h, grid_w]),
            "bucket": record["aspect_bucket"],
            "flipped": flipped,
        }


class M2ValidationDataset(Dataset):
    def __init__(self, root: Path, records: dict[str, dict], sample_ids: list[str]) -> None:
        self.root = root
        self.records = records
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = self.sample_ids[index]
        record = self.records[sample_id]
        image = image_tensor(self.root / "blended" / f"{sample_id}.png")
        target = image_tensor(self.root / "transmission_layer" / f"{sample_id}.png")
        width, height = record["target_size"]
        if image.shape != (3, height, width) or target.shape != image.shape:
            raise ValueError(f"Unexpected validation shape for {sample_id}")
        return {"id": sample_id, "image": image, "target": target, "bucket": record["aspect_bucket"]}


def load_and_validate_cache(
    cache_root: Path,
    data_root: Path,
    train_ids: list[str],
) -> tuple[dict, dict[str, dict]]:
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest.get("identity", {})
    source = manifest.get("source_dataset", {})
    if not manifest.get("complete"):
        raise RuntimeError("M2 Q20 cache is incomplete")
    if identity.get("formula_version") != "m2a-q20-variable-aspect-v1":
        raise RuntimeError("Unexpected M2 Q20 formula version")
    if identity.get("block_zero_based_index") != BLOCK_INDEX or identity.get("flow_timestep") != 499:
        raise RuntimeError("Unexpected Q20 layer or timestep")
    if source.get("split") != "train" or source.get("cached_count") != 144:
        raise RuntimeError("Cache is not the complete M2 train split")
    if identity.get("selected_ids") != train_ids:
        raise RuntimeError("Cache sample order differs from M2 train split")
    if source.get("manifest_sha256") != sha256(data_root / "manifest.json"):
        raise RuntimeError("M2 dataset manifest hash differs from cache identity")
    records = {record["id"]: record for record in manifest["samples"]}
    if set(records) != set(train_ids):
        raise RuntimeError("Cache record IDs differ from M2 train IDs")
    for sample_id in train_ids:
        record = records[sample_id]
        for key in ("gt_feature", "weight"):
            path = cache_root / record["cache"][key]
            expected = record["cache"][f"{key}_sha256"]
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"Cache file hash mismatch: {sample_id}/{key}")
    return manifest, records


def q20_prediction_features(
    backend: QwenSharedBackend,
    prediction: torch.Tensor,
    expected_tokens: int,
) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def stop_hook(_module, _inputs, output):
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("Unexpected Qwen block-20 output")
        captured["feature"] = output[1]
        raise StopAtQ20

    handle = backend.transformer.transformer_blocks[BLOCK_INDEX].register_forward_hook(stop_hook)
    try:
        latent = deterministic_encode(backend, prediction)
        try:
            backend.upstream.flow_step(latent, backend.transformer, backend.vae, backend.embeddings)
        except StopAtQ20:
            pass
    finally:
        handle.remove()
    if "feature" not in captured:
        raise RuntimeError("Frozen Qwen teacher did not reach block 20")
    feature = captured["feature"]
    if feature.ndim != 3 or feature.shape[1:] != (expected_tokens, 3072):
        raise ValueError(f"Unexpected predicted Q20 shape: {feature.shape}")
    return feature


def check_initial_hash(path: Path, expected: str, device: torch.device) -> str:
    actual = ""
    ok = torch.ones((), dtype=torch.int32, device=device)
    if rank() == 0:
        actual = sha256(path)
        if actual != expected:
            ok.zero_()
    dist.broadcast(ok, src=0)
    if not bool(ok.item()):
        raise RuntimeError("Stage-2 best LoRA SHA-256 mismatch")
    return actual if rank() == 0 else expected


def link_best(run_dir: Path, report: dict, step: int, source: Path) -> bool:
    record_path = run_dir / "best_metrics.json"
    previous = json.loads(record_path.read_text()) if record_path.is_file() else None
    current_l1 = float(report["means"]["l1"])
    if previous is not None and current_l1 >= float(previous["val_l1"]):
        return False
    destination = run_dir / "best_transmission_lora.safetensors"
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    os.link(source, destination)
    record_path.write_text(json.dumps({
        "val_l1": current_l1,
        "val_psnr": report["means"]["psnr"],
        "val_ssim": report["means"]["ssim"],
        "val_lpips_squeeze": report["means"]["lpips_squeeze"],
        "step": step,
        "criterion": "minimum saved-PNG macro validation L1",
    }, indent=2) + "\n", encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--initial-sha256", default=EXPECTED_INITIAL_SHA256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--local-coefficient", type=float, default=0.25)
    parser.add_argument("--keep-coefficient", type=float, default=0.10)
    parser.add_argument("--target-q-gradient-ratio", type=float, default=0.30)
    parser.add_argument("--lambda-q-min", type=float, default=0.0)
    parser.add_argument("--lambda-q-max", type=float, default=0.5)
    parser.add_argument("--gradient-ratio-tolerance", type=float, default=5e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=35)
    parser.add_argument("--save-every", type=int, default=35)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.cache_root = args.cache_root.resolve()
    args.run_dir = args.run_dir.resolve()
    args.initial = args.initial.resolve()
    manifest, records, splits = load_m2_manifest(args.data_root)
    cache_manifest, cache_records = load_and_validate_cache(
        args.cache_root, args.data_root, splits["train"]
    )
    train_data = M2Q20Dataset(
        args.data_root, args.cache_root, records, cache_records, splits["train"], True
    )
    val_data = M2ValidationDataset(args.data_root, records, splits["validation"])
    preflight_batches = grouped_global_batches(train_data, args.seed, 0, 4)
    preflight = {
        "status": "preflight_ok",
        "experiment": "M2-B1-corrected-q20grad30",
        "train": len(train_data),
        "validation": len(val_data),
        "sealed_test": len(splits["test"]),
        "global_batches_per_epoch": len(preflight_batches),
        "unique_samples_epoch0": len({i for batch in preflight_batches for i in batch}),
        "bucket_counts": dict(Counter(records[i]["aspect_bucket"] for i in splits["train"])),
        "cache_samples": len(cache_records),
        "cache_complete": bool(cache_manifest["complete"]),
        "q20_block": 20,
        "p90_read": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return

    device = setup_distributed()
    is_main = rank() == 0
    if world_size() != 4 or args.batch_size != 1 or args.gradient_accumulation != 1:
        raise RuntimeError("M2-B1 strict comparison requires 4 GPUs, batch 1, accumulation 1")
    if args.max_steps != 70 or args.validate_every != 35 or args.save_every != 35:
        raise RuntimeError("M2-B1 strict comparison requires 70 steps and checkpoints at 35/70")
    if not (0.0 < args.target_q_gradient_ratio < 1.0):
        raise ValueError("target-q-gradient-ratio must be between zero and one")
    if args.lambda_q_min < 0.0 or args.lambda_q_min >= args.lambda_q_max:
        raise ValueError("lambda-q bounds are invalid")
    seed_everything(args.seed)
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    sampler = AspectGroupedDistributedSampler(train_data, args.seed)
    loader = DataLoader(
        train_data,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

    backend = QwenSharedBackend.from_local(device)
    backend.transformer.enable_gradient_checkpointing()
    backend.set_trainable_branch("transmission")
    initial_sha = check_initial_hash(args.initial, args.initial_sha256, device)
    if is_main:
        load_initial(backend, args.initial, device)
    parameters = trainable_parameters(backend)
    sync_initial_parameters(parameters)
    names = [name for name, value in backend.transformer.named_parameters() if value.requires_grad]
    if not names or any(".lora_" not in name or f".{ADAPTER_NAMES['transmission']}." not in name for name in names):
        raise RuntimeError("Trainable tensors are not exclusively LoRA_T")
    if any(value.requires_grad for value in backend.vae.parameters()):
        raise RuntimeError("VAE must be frozen")
    if any(value.requires_grad for name, value in backend.transformer.named_parameters() if ".lora_" not in name):
        raise RuntimeError("Qwen backbone must be frozen")
    backend.transformer.train()
    backend.vae.eval()
    seed_everything(args.seed + rank())

    optimizer = bnb.optim.PagedAdamW8bit(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, min(args.warmup_steps, args.max_steps), args.max_steps)
    lpips_model = None
    if is_main:
        lpips_model = lpips.LPIPS(net="squeeze", verbose=False).eval().cpu()
        for parameter in lpips_model.parameters():
            parameter.requires_grad_(False)
        metadata = {
            "experiment": "M2-B1-corrected-q20grad30",
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "args": vars(args),
            "world_size": world_size(),
            "effective_batch": 4,
            "optimizer_steps_per_epoch": len(loader),
            "planned_optimizer_steps": 70,
            "initialization": str(args.initial),
            "initial_sha256": initial_sha,
            "loss": "L_base + 0.25*L_weighted_charbonnier + 0.10*L_low_response_keep + lambda_q*L_Q20",
            "spatial_weight": "S=D_Q*(0.7+0.3*D_DoLP); W=clip(1+2*S,1,3), per-image mean 1",
            "q20_gradient_control": "per-step exact 30% output-gradient ratio; direct lambda; no EMA; lambda [0,0.5]",
            "augmentation": "same seeded horizontal flip as M2-A; cached weights and Q20 token grid flip with RGB pair",
            "p90_read": False,
            "split_ids": splits,
            "dataset_manifest_sha256": sha256(args.data_root / "manifest.json"),
            "cache_manifest_sha256": sha256(args.cache_root / "manifest.json"),
            "preflight": preflight,
        }
        (args.run_dir / "run_config.json").write_text(
            json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2, default=str), flush=True)

    dist.barrier()
    if is_main:
        baseline = validate(backend, val_loader, device, lpips_model, args.run_dir, 0, args.seed)
        updated = link_best(args.run_dir, baseline, 0, args.initial)
        event = {"kind": "M2-S0-stage2-init", "step": 0, "means": baseline["means"], "best_updated": updated}
        append_jsonl(args.run_dir / "metrics.jsonl", event)
        print(json.dumps(event), flush=True)
    dist.barrier()

    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    global_step = 0
    lambda_q = 0.0
    actual_ratio = measured_base_norm = measured_q_norm = 0.0
    last_validation = None
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        sampler.set_epoch(epoch)
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            weight_pixel = batch["weight_pixel"].to(device, non_blocking=True)
            weight_token = batch["weight_token"].to(device, non_blocking=True)
            q20_gt = batch["q20_gt"].to(device, non_blocking=True)
            grid = batch["token_grid"][0]
            expected_tokens = int(grid[0]) * int(grid[1])
            if not all(torch.isfinite(x).all() for x in (weight_pixel, weight_token, q20_gt)):
                raise RuntimeError("Non-finite cached supervision")

            prediction = backend.forward_normalized(image, "transmission")
            backend.transformer.disable_lora()
            try:
                q20_prediction = q20_prediction_features(backend, prediction, expected_tokens)
                base_bundle, q20_loss, _losses, scalars = c1_losses(
                    prediction, image, target, weight_pixel, weight_token,
                    q20_prediction, q20_gt, args.local_coefficient, args.keep_coefficient,
                )
                q_grad = torch.autograd.grad(q20_loss, prediction)[0]
            finally:
                backend.transformer.enable_lora()
                backend.transformer.set_adapter(ADAPTER_NAMES["transmission"])
            base_grad = torch.autograd.grad(base_bundle, prediction)[0]
            if not torch.isfinite(base_grad).all() or not torch.isfinite(q_grad).all():
                raise RuntimeError("Non-finite output gradient")

            measured_base_norm = global_norm(base_grad)
            measured_q_norm = global_norm(q_grad)
            if measured_base_norm <= 0 or measured_q_norm <= 0:
                raise RuntimeError("Invalid base/Q20 gradient norm")
            desired = args.target_q_gradient_ratio * measured_base_norm / measured_q_norm
            lambda_q = min(args.lambda_q_max, max(args.lambda_q_min, desired))
            actual_ratio = lambda_q * measured_q_norm / measured_base_norm
            if abs(actual_ratio - args.target_q_gradient_ratio) > args.gradient_ratio_tolerance:
                raise RuntimeError(
                    f"Q20 gradient ratio control failed: target={args.target_q_gradient_ratio}, "
                    f"actual={actual_ratio}, lambda_q={lambda_q}, unclamped={desired}"
                )

            combined = base_grad + lambda_q * q_grad
            prediction.backward(combined)
            frozen_clean = all(
                value.grad is None for name, value in backend.transformer.named_parameters()
                if ".lora_" not in name
            ) and all(value.grad is None for value in backend.vae.parameters())
            if not frozen_clean:
                raise RuntimeError("Frozen backbone/VAE received gradients")
            del prediction, q20_prediction, base_grad, q_grad, combined, _losses
            del image, target, weight_pixel, weight_token, q20_gt

            active_tensors = sync_gradients(parameters, device)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
                raise RuntimeError(f"Invalid LoRA_T gradient norm: {grad_norm}")
            move_optimizer_state(optimizer, device)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            move_optimizer_state(optimizer, torch.device("cpu"))
            torch.cuda.empty_cache()
            global_step += 1

            total_loss = float(base_bundle.detach() + lambda_q * q20_loss.detach())
            values = torch.tensor([
                total_loss, scalars["l1"], scalars["ssim_loss"], scalars["edge"],
                scalars["local_loss"], scalars["keep_loss"], scalars["q20_loss"], float(grad_norm),
            ], device=device)
            dist.all_reduce(values)
            values.div_(world_size())
            if is_main:
                record = {
                    "kind": "train", "step": global_step, "epoch": epoch,
                    "loss": float(values[0]), "l1": float(values[1]),
                    "ssim_loss": float(values[2]), "edge": float(values[3]),
                    "weighted_charbonnier": float(values[4]),
                    "low_response_keep": float(values[5]), "q20_loss": float(values[6]),
                    "lambda_q": lambda_q, "q20_gradient_ratio": actual_ratio,
                    "base_output_grad_norm": measured_base_norm, "q20_output_grad_norm": measured_q_norm,
                    "grad_norm": float(values[7]), "active_gradient_tensors": active_tensors,
                    "lr": scheduler.get_last_lr()[0], "elapsed_seconds": time.monotonic() - started,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                }
                append_jsonl(args.run_dir / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)
            del base_bundle, q20_loss, scalars, grad_norm, values
            torch.cuda.empty_cache()

            if global_step % args.validate_every == 0:
                dist.barrier()
                if is_main:
                    last_validation = validate(
                        backend, val_loader, device, lpips_model, args.run_dir, global_step, args.seed
                    )
                    checkpoint = save_checkpoint(
                        args.run_dir, global_step, epoch, backend, scheduler, args
                    )
                    updated = link_best(
                        args.run_dir, last_validation, global_step,
                        checkpoint / "transmission_lora.safetensors",
                    )
                    event = {"kind": "validation", "step": global_step, "epoch": epoch,
                             "means": last_validation["means"], "best_updated": updated}
                    append_jsonl(args.run_dir / "metrics.jsonl", event)
                    print(json.dumps(event), flush=True)
                    print(f"saved {checkpoint}", flush=True)
                dist.barrier()
            if global_step >= args.max_steps:
                stop = True
                break

    if global_step != 70:
        raise RuntimeError(f"M2-B1 stopped at {global_step}, expected 70")
    dist.barrier()
    if is_main:
        final_checkpoint = args.run_dir / "checkpoint-0000070"
        if not final_checkpoint.is_dir():
            final_checkpoint = save_checkpoint(args.run_dir, 70, epoch, backend, scheduler, args)
        if last_validation is None or int(last_validation["step"]) != 70:
            last_validation = validate(backend, val_loader, device, lpips_model, args.run_dir, 70, args.seed)
        summary = {
            "status": "complete", "experiment": "M2-B1-corrected-q20grad30", "optimizer_updates": 70,
            "final_checkpoint": str(final_checkpoint),
            "best_metrics": json.loads((args.run_dir / "best_metrics.json").read_text()),
            "final_validation": last_validation["means"],
            "final_lambda_q": lambda_q, "final_q20_gradient_ratio": actual_ratio,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (args.run_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
