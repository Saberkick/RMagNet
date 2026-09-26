"""M2-A: 70-step data-only continuation from the Stage-2 best Transmission LoRA.

This arm deliberately does not read Q20, DoLP, or P90.  It uses the original
Stage-2 reconstruction objective on the grouped M2 train split and evaluates
saved 8-bit PNG predictions on the fixed validation split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import bitsandbytes as bnb
import lpips
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from .m1b_train import load_initial
from .qwen_backend import QwenSharedBackend
from .stage1_train import (
    append_jsonl,
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
from .stage2_train import adapter_state, transmission_loss


ROOT = Path("/share/linmingheng-local/xuke")
PROJECT = ROOT / "RMagNet"
DEFAULT_DATA = ROOT / "datasets/rmagnet_m2_aspect"
DEFAULT_RUN = PROJECT / "runs/m2_corrected_a_data70_noq20"
DEFAULT_INITIAL = PROJECT / "runs/stage2_transmission_r128/best_transmission_lora.safetensors"
EXPECTED_INITIAL_SHA256 = "f5737d4ffb89e86874a96a02bd58a074299ca12e00ec15cac438c403a342085a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    ids = path.read_text(encoding="utf-8").split()
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate IDs in {path}")
    return ids


def load_m2_manifest(data_root: Path) -> tuple[dict, dict[str, dict], dict[str, list[str]]]:
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("version") != "m2-variable-aspect-v2-corrected-labels":
        raise RuntimeError("M2 dataset manifest is incomplete or incompatible")
    records = {record["id"]: record for record in manifest["samples"]}
    splits = {
        name: read_ids(data_root / "splits" / f"{name}.txt")
        for name in ("train", "validation", "test")
    }
    if [len(splits[name]) for name in ("train", "validation", "test")] != [144, 18, 18]:
        raise RuntimeError(f"Unexpected M2 split sizes: { {k: len(v) for k, v in splits.items()} }")
    sets = {name: set(ids) for name, ids in splits.items()}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise RuntimeError("M2 splits overlap")
    if set().union(*sets.values()) != set(records):
        raise RuntimeError("M2 split union differs from manifest IDs")
    groups = {
        name: {records[sample_id]["group"] for sample_id in ids}
        for name, ids in splits.items()
    }
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
        raise RuntimeError("M2 capture groups leak across splits")
    return manifest, records, splits


class M2TransmissionDataset(Dataset):
    def __init__(
        self,
        root: Path,
        records: dict[str, dict],
        sample_ids: list[str],
        augment: bool,
    ) -> None:
        self.root = root
        self.records = records
        self.sample_ids = sample_ids
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = self.sample_ids[index]
        record = self.records[sample_id]
        image = image_tensor(self.root / "blended" / f"{sample_id}.png")
        target = image_tensor(self.root / "transmission_layer" / f"{sample_id}.png")
        width, height = record["target_size"]
        if image.shape != (3, height, width) or target.shape != image.shape:
            raise ValueError(
                f"Unexpected M2 pair shape for {sample_id}: {tuple(image.shape)}, "
                f"expected (3,{height},{width})"
            )
        flipped = False
        if self.augment and torch.rand(()) < 0.5:
            image = image.flip(-1)
            target = target.flip(-1)
            flipped = True
        return {
            "id": sample_id,
            "image": image,
            "target": target,
            "bucket": record["aspect_bucket"],
            "flipped": flipped,
        }


def grouped_global_batches(
    dataset: M2TransmissionDataset,
    seed: int,
    epoch: int,
    replicas: int,
) -> list[list[int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, sample_id in enumerate(dataset.sample_ids):
        buckets[dataset.records[sample_id]["aspect_bucket"]].append(index)
    rng = random.Random(seed + epoch)
    batches: list[list[int]] = []
    leftovers: list[int] = []
    for bucket in sorted(buckets):
        indices = buckets[bucket]
        rng.shuffle(indices)
        full = len(indices) // replicas * replicas
        batches.extend(indices[offset : offset + replicas] for offset in range(0, full, replicas))
        leftovers.extend(indices[full:])
    rng.shuffle(leftovers)
    if len(leftovers) % replicas:
        raise RuntimeError("M2 grouped sampler leftovers do not form complete global batches")
    batches.extend(
        leftovers[offset : offset + replicas]
        for offset in range(0, len(leftovers), replicas)
    )
    rng.shuffle(batches)
    flat = [index for batch in batches for index in batch]
    if len(flat) != len(dataset) or len(set(flat)) != len(dataset):
        raise RuntimeError("M2 grouped sampler did not cover each sample exactly once")
    return batches


class AspectGroupedDistributedSampler(Sampler[int]):
    def __init__(self, dataset: M2TransmissionDataset, seed: int) -> None:
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self.replicas = world_size()
        self.current_rank = rank()
        if len(dataset) % self.replicas:
            raise RuntimeError(
                f"Dataset size {len(dataset)} is not divisible by world size {self.replicas}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        batches = grouped_global_batches(self.dataset, self.seed, self.epoch, self.replicas)
        return iter(batch[self.current_rank] for batch in batches)

    def __len__(self) -> int:
        return len(self.dataset) // self.replicas


def tensor_from_saved_png(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = torch.from_numpy(__import__("numpy").asarray(image.convert("RGB"), dtype="float32").copy())
    return array.permute(2, 0, 1).unsqueeze(0).div_(255.0)


def save_prediction(prediction01: torch.Tensor, path: Path) -> None:
    array = (
        prediction01[0]
        .detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(array, mode="RGB").save(path, format="PNG", compress_level=6)


def macro(rows: list[dict], keys: tuple[str, ...]) -> dict[str, float]:
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


@torch.no_grad()
def validate(
    backend: QwenSharedBackend,
    loader: DataLoader,
    device: torch.device,
    lpips_model: torch.nn.Module,
    run_dir: Path,
    step: int,
    seed: int,
) -> dict:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    backend.transformer.eval()
    backend.vae.eval()

    prediction_dir = run_dir / "validation" / f"step_{step:06d}" / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for batch in loader:
        sample_id = batch["id"][0]
        bucket = batch["bucket"][0]
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        prediction = backend.forward_normalized(image, "transmission")
        prediction01 = ((prediction.float() + 1) * 0.5).clamp(0, 1)
        output_path = prediction_dir / f"{sample_id}.png"
        save_prediction(prediction01, output_path)

        pred_saved = tensor_from_saved_png(output_path)
        gt_saved = ((batch["target"].float() + 1) * 0.5).clamp(0, 1)
        input_saved = ((batch["image"].float() + 1) * 0.5).clamp(0, 1)
        error = (pred_saved - gt_saved).abs().mean(1, keepdim=True)
        change = (input_saved - gt_saved).abs().mean(1, keepdim=True)
        low_threshold = torch.quantile(change.flatten(), 0.25)
        high_threshold = torch.quantile(change.flatten(), 0.75)
        low_mask = change <= low_threshold
        high_mask = change >= high_threshold
        mse = F.mse_loss(pred_saved, gt_saved)
        lpips_value = float(
            lpips_model(pred_saved.mul(2).sub(1), gt_saved.mul(2).sub(1)).mean()
        )
        rows.append(
            {
                "id": sample_id,
                "bucket": bucket,
                "width": int(pred_saved.shape[-1]),
                "height": int(pred_saved.shape[-2]),
                "l1": float(F.l1_loss(pred_saved, gt_saved)),
                "psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))),
                "ssim": float(ssim(pred_saved, gt_saved)),
                "lpips_squeeze": lpips_value,
                "low_change_keep_l1": float(error[low_mask].mean()),
                "high_change_restore_l1": float(error[high_mask].mean()),
            }
        )
        del image, target, prediction, prediction01
        torch.cuda.empty_cache()

    keys = (
        "l1",
        "psnr",
        "ssim",
        "lpips_squeeze",
        "low_change_keep_l1",
        "high_change_restore_l1",
    )
    means = macro(rows, keys)
    bucket_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        bucket_rows[row["bucket"]].append(row)
    by_bucket = {
        bucket: {"count": len(values), **macro(values, keys)}
        for bucket, values in sorted(bucket_rows.items())
    }
    report = {
        "step": step,
        "metric_domain": "saved 8-bit RGB PNG at each sample's aspect-preserved M2 size",
        "lpips_network": "SqueezeNet v1.1, LPIPS v0.1",
        "change_regions": "per-image bottom/top quartile of mean RGB |I-GT|",
        "means": means,
        "by_bucket": by_bucket,
        "per_image": rows,
    }
    report_dir = prediction_dir.parent
    (report_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (report_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    backend.transformer.train()
    backend.vae.eval()
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state, device)
    return report


def atomic_save_adapter(path: Path, backend: QwenSharedBackend) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    safetensors.torch.save_file(adapter_state(backend), temporary)
    os.replace(temporary, path)


def maybe_save_best(
    run_dir: Path,
    report: dict,
    step: int,
    backend: QwenSharedBackend,
    initial: Path,
) -> bool:
    record_path = run_dir / "best_metrics.json"
    previous = json.loads(record_path.read_text()) if record_path.is_file() else None
    current_l1 = float(report["means"]["l1"])
    if previous is not None and current_l1 >= float(previous["val_l1"]):
        return False
    best_path = run_dir / "best_transmission_lora.safetensors"
    if step == 0:
        if best_path.exists():
            best_path.unlink()
        os.link(initial, best_path)
    else:
        atomic_save_adapter(best_path, backend)
    record_path.write_text(
        json.dumps(
            {
                "val_l1": current_l1,
                "val_psnr": report["means"]["psnr"],
                "val_ssim": report["means"]["ssim"],
                "val_lpips_squeeze": report["means"]["lpips_squeeze"],
                "step": step,
                "criterion": "minimum saved-PNG macro validation L1",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return True


def save_checkpoint(
    run_dir: Path,
    step: int,
    epoch: int,
    backend: QwenSharedBackend,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args: argparse.Namespace,
) -> Path:
    checkpoint = run_dir / f"checkpoint-{step:07d}"
    checkpoint.mkdir(parents=True, exist_ok=False)
    atomic_save_adapter(checkpoint / "transmission_lora.safetensors", backend)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "scheduler": scheduler.state_dict(),
            "optimizer_state_saved": False,
            "resume_policy": "experiment is short; restart from Stage-2 best for strict comparison",
        },
        checkpoint / "trainer_state.pt",
    )
    (checkpoint / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8"
    )
    return checkpoint


def check_initial_hash(path: Path, expected: str, device: torch.device) -> str:
    actual = ""
    ok = torch.ones((), dtype=torch.int32, device=device)
    if rank() == 0:
        actual = sha256(path)
        if actual != expected:
            ok.zero_()
    if dist.is_initialized():
        dist.broadcast(ok, src=0)
    if not bool(ok.item()):
        raise RuntimeError("Stage-2 best LoRA SHA-256 mismatch")
    return actual if rank() == 0 else expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
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
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=35)
    parser.add_argument("--save-every", type=int, default=35)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.run_dir = args.run_dir.resolve()
    args.initial = args.initial.resolve()
    manifest, records, splits = load_m2_manifest(args.data_root)
    train_data = M2TransmissionDataset(args.data_root, records, splits["train"], True)
    val_data = M2TransmissionDataset(args.data_root, records, splits["validation"], False)

    preflight_batches = grouped_global_batches(train_data, args.seed, 0, 4)
    preflight = {
        "status": "preflight_ok",
        "train": len(train_data),
        "validation": len(val_data),
        "sealed_test": len(splits["test"]),
        "world_size_expected": 4,
        "global_batches_per_epoch": len(preflight_batches),
        "samples_per_global_batch": 4,
        "unique_samples_epoch0": len({index for batch in preflight_batches for index in batch}),
        "bucket_counts": dict(Counter(records[sample_id]["aspect_bucket"] for sample_id in splits["train"])),
        "q20_read": False,
        "dolp_read": False,
        "p90_read": False,
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return

    device = setup_distributed()
    is_main = rank() == 0
    if world_size() != 4 or args.batch_size != 1 or args.gradient_accumulation != 1:
        raise RuntimeError("M2-A strict comparison requires 4 GPUs, per-GPU batch 1, accumulation 1")
    if args.max_steps != 70:
        raise RuntimeError("M2-A strict comparison requires exactly 70 optimizer updates")
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
    if args.gradient_checkpointing:
        backend.transformer.enable_gradient_checkpointing()
    backend.set_trainable_branch("transmission")
    initial_sha = check_initial_hash(args.initial, args.initial_sha256, device)
    if is_main:
        load_initial(backend, args.initial, device)
    parameters = trainable_parameters(backend)
    sync_initial_parameters(parameters)
    backend.transformer.train()
    backend.vae.eval()
    seed_everything(args.seed + rank())

    optimizer = bnb.optim.PagedAdamW8bit(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = make_scheduler(
        optimizer, min(args.warmup_steps, args.max_steps), args.max_steps
    )
    lpips_model = None
    if is_main:
        lpips_model = lpips.LPIPS(net="squeeze", verbose=False).eval().cpu()
        for parameter in lpips_model.parameters():
            parameter.requires_grad_(False)
        metadata = {
            "experiment": "M2-A-corrected-data70-noq20",
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "args": vars(args),
            "world_size": world_size(),
            "effective_batch": world_size() * args.batch_size * args.gradient_accumulation,
            "optimizer_steps_per_epoch": len(loader),
            "planned_optimizer_steps": args.max_steps,
            "initialization": str(args.initial),
            "initial_sha256": initial_sha,
            "loss": "L1 + 0.2*(1-SSIM) + 0.1*edge_L1",
            "forbidden_inputs": {
                "q20_cache": False,
                "dolp": False,
                "p90": False,
                "weighted_charbonnier": False,
                "low_response_keep": False,
                "q20_feature_loss": False,
            },
            "split_ids": splits,
            "dataset_manifest_sha256": sha256(args.data_root / "manifest.json"),
            "preflight": preflight,
        }
        (args.run_dir / "run_config.json").write_text(
            json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2, default=str), flush=True)

    if dist.is_initialized():
        dist.barrier()
    if is_main:
        baseline = validate(
            backend, val_loader, device, lpips_model, args.run_dir, 0, args.seed
        )
        best_updated = maybe_save_best(
            args.run_dir, baseline, 0, backend, args.initial
        )
        event = {
            "kind": "M2-S0-stage2-init",
            "step": 0,
            "means": baseline["means"],
            "best_updated": best_updated,
        }
        append_jsonl(args.run_dir / "metrics.jsonl", event)
        print(json.dumps(event), flush=True)
    if dist.is_initialized():
        dist.barrier()

    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    global_step = 0
    last_validation = None
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        sampler.set_epoch(epoch)
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            prediction = backend.forward_normalized(image, "transmission")
            loss, parts = transmission_loss(
                prediction, target, args.ssim_weight, args.edge_weight
            )
            loss.backward()
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
                    "kind": "train",
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

            if args.validate_every > 0 and global_step % args.validate_every == 0:
                if dist.is_initialized():
                    dist.barrier()
                if is_main:
                    last_validation = validate(
                        backend,
                        val_loader,
                        device,
                        lpips_model,
                        args.run_dir,
                        global_step,
                        args.seed,
                    )
                    best_updated = maybe_save_best(
                        args.run_dir,
                        last_validation,
                        global_step,
                        backend,
                        args.initial,
                    )
                    event = {
                        "kind": "validation",
                        "step": global_step,
                        "epoch": epoch,
                        "means": last_validation["means"],
                        "best_updated": best_updated,
                    }
                    append_jsonl(args.run_dir / "metrics.jsonl", event)
                    print(json.dumps(event), flush=True)
                if dist.is_initialized():
                    dist.barrier()
            if is_main and args.save_every > 0 and global_step % args.save_every == 0:
                checkpoint = save_checkpoint(
                    args.run_dir, global_step, epoch, backend, scheduler, args
                )
                print(f"saved {checkpoint}", flush=True)
            if dist.is_initialized():
                dist.barrier()
            if global_step >= args.max_steps:
                stop = True
                break

    if global_step != args.max_steps:
        raise RuntimeError(f"M2-A stopped at {global_step}, expected {args.max_steps}")
    if dist.is_initialized():
        dist.barrier()
    if is_main:
        final_checkpoint = args.run_dir / f"checkpoint-{global_step:07d}"
        if not final_checkpoint.is_dir():
            final_checkpoint = save_checkpoint(
                args.run_dir, global_step, epoch, backend, scheduler, args
            )
        if last_validation is None or int(last_validation["step"]) != global_step:
            last_validation = validate(
                backend,
                val_loader,
                device,
                lpips_model,
                args.run_dir,
                global_step,
                args.seed,
            )
        summary = {
            "status": "complete",
            "experiment": "M2-A-corrected-data70-noq20",
            "optimizer_updates": global_step,
            "final_checkpoint": str(final_checkpoint),
            "best_metrics": json.loads((args.run_dir / "best_metrics.json").read_text()),
            "final_validation": last_validation["means"],
            "completed_at_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
        }
        (args.run_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
