"""M3-noLrec: 70-step auxiliary-only semantic-separation ablation.

L_rec is deliberately absent.  The trainable objective is exactly
0.25*L_cluster + 0.10*L_relation + 0.10*L_cons + 0.05*L_boundary.
The shared Transmission LoRA starts from the Stage-2 best checkpoint.
"""

from __future__ import annotations

import argparse
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
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .c1_l20_train import move_optimizer_state
from .m1b_train import load_initial
from .m2a_data_baseline import (
    AspectGroupedDistributedSampler,
    check_initial_hash,
    grouped_global_batches,
    load_m2_manifest,
    maybe_save_best,
    save_checkpoint,
    sha256,
    validate,
)
from .m2b1_q20 import M2ValidationDataset, q20_prediction_features
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
DEFAULT_CACHE = PROJECT / "data_cache/m3_semantic_v1"
DEFAULT_RUN = PROJECT / "runs/m3_nolrec_70"
DEFAULT_INITIAL = PROJECT / "runs/stage2_transmission_r128/best_transmission_lora.safetensors"
EXPECTED_INITIAL_SHA256 = "f5737d4ffb89e86874a96a02bd58a074299ca12e00ec15cac438c403a342085a"
CACHE_VERSION = "m3-semantic-separation-v1"
CLUSTERS = 4
BLOCK_INDEX = 19


class M3NoLrecDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        cache_root: Path,
        records: dict[str, dict],
        cache_records: dict[str, dict],
        sample_ids: list[str],
        augment: bool,
    ) -> None:
        self.root = data_root
        self.cache_root = cache_root
        self.records = records
        self.cache_records = cache_records
        self.sample_ids = sample_ids
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sample_ids)

    @staticmethod
    def flip_edges(edge_index: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        row = edge_index // grid_w
        column = edge_index % grid_w
        return row * grid_w + (grid_w - 1 - column)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample_id = self.sample_ids[index]
        record = self.records[sample_id]
        cached_record = self.cache_records[sample_id]
        width, height = record["target_size"]
        image = image_tensor(self.root / "blended" / f"{sample_id}.png")
        p90 = image_tensor(self.root / "reflection_90" / f"{sample_id}.png")
        target = image_tensor(self.root / "transmission_layer" / f"{sample_id}.png")
        expected = (3, height, width)
        if image.shape != expected or p90.shape != expected or target.shape != expected:
            raise ValueError(f"M3 RGB shape mismatch for {sample_id}")

        stored = safetensors.torch.load_file(self.cache_root / cached_record["cache"]["sample"])
        posterior = stored["posterior"].float()
        confidence = stored["confidence"].float()
        reflection = stored["reflection_evidence"].float()
        boundary = stored["boundary"].float()
        edge_index = stored["edge_index"].long()
        edge_target = stored["edge_target"].float()
        edge_confidence = stored["edge_confidence"].float()
        grid_h, grid_w = (int(value) for value in stored["token_grid_hw"])
        tokens = grid_h * grid_w
        if posterior.shape != (CLUSTERS, tokens):
            raise ValueError(f"Posterior shape mismatch for {sample_id}: {posterior.shape}")
        if any(value.shape != (tokens,) for value in (confidence, reflection, boundary)):
            raise ValueError(f"Token map shape mismatch for {sample_id}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Relation edge shape mismatch for {sample_id}")
        if edge_target.numel() != edge_index.shape[1] or edge_confidence.numel() != edge_index.shape[1]:
            raise ValueError(f"Relation target shape mismatch for {sample_id}")
        values = (posterior, confidence, reflection, boundary, edge_target, edge_confidence)
        if not all(torch.isfinite(value).all() for value in values):
            raise RuntimeError(f"Non-finite M3 cache for {sample_id}")
        if float((posterior.sum(0) - 1).abs().max()) > 2e-3:
            raise RuntimeError(f"Posterior normalization failed for {sample_id}")

        flipped = False
        if self.augment and torch.rand(()) < 0.5:
            image = image.flip(-1)
            p90 = p90.flip(-1)
            target = target.flip(-1)
            posterior = posterior.reshape(CLUSTERS, grid_h, grid_w).flip(-1).reshape(CLUSTERS, tokens)
            confidence = confidence.reshape(grid_h, grid_w).flip(-1).reshape(tokens)
            reflection = reflection.reshape(grid_h, grid_w).flip(-1).reshape(tokens)
            boundary = boundary.reshape(grid_h, grid_w).flip(-1).reshape(tokens)
            edge_index = self.flip_edges(edge_index, grid_h, grid_w)
            flipped = True
        return {
            "id": sample_id,
            "image": image,
            "p90": p90,
            "target": target,
            "posterior": posterior,
            "confidence": confidence,
            "reflection": reflection,
            "boundary": boundary,
            "edge_index": edge_index,
            "edge_target": edge_target,
            "edge_confidence": edge_confidence,
            "token_grid": torch.tensor([grid_h, grid_w], dtype=torch.int32),
            "bucket": record["aspect_bucket"],
            "flipped": flipped,
        }


def load_cache(cache_root: Path, data_root: Path, train_ids: list[str]) -> tuple[dict, dict[str, dict]]:
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source_dataset", {})
    teacher = manifest.get("teacher", {})
    if not manifest.get("complete") or manifest.get("cache_version") != CACHE_VERSION:
        raise RuntimeError("M3 semantic cache is incomplete or incompatible")
    if source.get("train_ids") != train_ids or source.get("sample_count") != 144:
        raise RuntimeError("M3 cache train split differs from the dataset")
    if source.get("manifest_sha256") != sha256(data_root / "manifest.json"):
        raise RuntimeError("M3 cache dataset manifest hash mismatch")
    if teacher.get("block_zero_based_index") != BLOCK_INDEX or teacher.get("flow_timestep") != 499:
        raise RuntimeError("M3 cache teacher identity mismatch")
    records = {record["id"]: record for record in manifest["samples"]}
    if set(records) != set(train_ids):
        raise RuntimeError("M3 cache record IDs differ from train IDs")
    for sample_id in train_ids:
        path = cache_root / records[sample_id]["cache"]["sample"]
        if not path.is_file() or sha256(path) != records[sample_id]["cache"]["sample_sha256"]:
            raise RuntimeError(f"M3 compact cache hash mismatch: {sample_id}")
    return manifest, records


def forward_from_latent(backend: QwenSharedBackend, latent: torch.Tensor) -> torch.Tensor:
    backend.transformer.set_adapter(ADAPTER_NAMES["transmission"])
    edited = backend.upstream.flow_step(latent, backend.transformer, backend.vae, backend.embeddings)
    return backend.upstream.decode(edited, backend.vae)


def pixel_maps(batch: dict, device: torch.device, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grid_h, grid_w = (int(value) for value in batch["token_grid"][0])
    posterior = batch["posterior"].to(device, non_blocking=True).reshape(1, CLUSTERS, grid_h, grid_w)
    reflection = batch["reflection"].to(device, non_blocking=True).reshape(1, 1, grid_h, grid_w)
    boundary = batch["boundary"].to(device, non_blocking=True).reshape(1, 1, grid_h, grid_w)
    posterior = F.interpolate(posterior, size=(height, width), mode="bilinear", align_corners=False)
    posterior /= posterior.sum(1, keepdim=True).clamp_min(1e-6)
    reflection = F.interpolate(reflection, size=(height, width), mode="bilinear", align_corners=False).clamp(0, 1)
    boundary = F.interpolate(boundary, size=(height, width), mode="bilinear", align_corners=False).clamp(0, 1)
    return posterior, reflection, boundary


def cluster_loss(prediction: torch.Tensor, target: torch.Tensor, posterior: torch.Tensor) -> torch.Tensor:
    prediction01 = ((prediction.float() + 1) * 0.5).clamp(0, 1)
    target01 = ((target.float() + 1) * 0.5).clamp(0, 1)
    error = torch.sqrt((prediction01 - target01).square() + 1e-6).mean(1, keepdim=True)
    numerator = (posterior * error).sum(dim=(-2, -1))
    denominator = posterior.sum(dim=(-2, -1)).clamp_min(1e-6)
    return (numerator / denominator).mean()


def consistency_loss(first: torch.Tensor, second_detached: torch.Tensor, reflection: torch.Tensor) -> torch.Tensor:
    first01 = ((first.float() + 1) * 0.5).clamp(0, 1)
    second01 = ((second_detached.float() + 1) * 0.5).clamp(0, 1)
    error = torch.sqrt((first01 - second01).square() + 1e-6).mean(1, keepdim=True)
    weight = 1.0 + reflection
    return (weight * error).sum() / (weight.sum().clamp_min(1e-6))


def boundary_loss(prediction: torch.Tensor, target: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
    prediction01 = ((prediction.float() + 1) * 0.5).clamp(0, 1)
    target01 = ((target.float() + 1) * 0.5).clamp(0, 1)
    dx = (prediction01[..., :, 1:] - prediction01[..., :, :-1]) - (target01[..., :, 1:] - target01[..., :, :-1])
    dy = (prediction01[..., 1:, :] - prediction01[..., :-1, :]) - (target01[..., 1:, :] - target01[..., :-1, :])
    error_x = torch.sqrt(dx.square() + 1e-6).mean(1, keepdim=True)
    error_y = torch.sqrt(dy.square() + 1e-6).mean(1, keepdim=True)
    weight_x = 0.5 * (boundary[..., :, 1:] + boundary[..., :, :-1])
    weight_y = 0.5 * (boundary[..., 1:, :] + boundary[..., :-1, :])
    loss_x = (weight_x * error_x).sum() / weight_x.sum().clamp_min(1e-6)
    loss_y = (weight_y * error_y).sum() / weight_y.sum().clamp_min(1e-6)
    return 0.5 * (loss_x + loss_y)


def relation_loss(
    q20_prediction: torch.Tensor,
    edge_index: torch.Tensor,
    edge_target: torch.Tensor,
    edge_confidence: torch.Tensor,
) -> torch.Tensor:
    if q20_prediction.ndim != 3 or q20_prediction.shape[0] != 1:
        raise ValueError(f"Unexpected Q20 prediction shape: {q20_prediction.shape}")
    feature = F.normalize(q20_prediction[0].float(), dim=-1, eps=1e-6)
    edge_index = edge_index.long()
    similarity = (feature[edge_index[0]] * feature[edge_index[1]]).sum(1)
    centered = similarity - similarity.detach().median()
    predicted_relation = torch.sigmoid(centered / 0.07)
    per_edge = F.smooth_l1_loss(predicted_relation, edge_target.float(), reduction="none")
    weight = edge_confidence.float()
    return (per_edge * weight).sum() / weight.sum().clamp_min(1e-6)


def finite_losses(losses: dict[str, torch.Tensor]) -> None:
    if not all(torch.isfinite(value) for value in losses.values()):
        bad = {name: float(value.detach()) for name, value in losses.items()}
        raise RuntimeError(f"Non-finite M3-noLrec losses: {bad}")


def output_norm(value: torch.Tensor) -> float:
    return float(value.float().square().sum().sqrt())


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
    parser.add_argument("--cluster-coefficient", type=float, default=0.25)
    parser.add_argument("--relation-coefficient", type=float, default=0.10)
    parser.add_argument("--consistency-coefficient", type=float, default=0.10)
    parser.add_argument("--boundary-coefficient", type=float, default=0.05)
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
    cache_manifest, cache_records = load_cache(args.cache_root, args.data_root, splits["train"])
    train_data = M3NoLrecDataset(
        args.data_root, args.cache_root, records, cache_records, splits["train"], True
    )
    val_data = M2ValidationDataset(args.data_root, records, splits["validation"])
    preflight_batches = grouped_global_batches(train_data, args.seed, 0, 4)
    preflight = {
        "status": "preflight_ok",
        "experiment": "M3-noLrec-70",
        "train": len(train_data),
        "validation": len(val_data),
        "sealed_test": len(splits["test"]),
        "world_size_expected": 4,
        "global_batches_per_epoch": len(preflight_batches),
        "unique_samples_epoch0": len({index for batch in preflight_batches for index in batch}),
        "bucket_counts": dict(Counter(records[sample_id]["aspect_bucket"] for sample_id in splits["train"])),
        "cache_version": cache_manifest["cache_version"],
        "cache_samples": len(cache_records),
        "loss_rec_present": False,
        "loss_formula": "0.25*L_cluster + 0.10*L_relation + 0.10*L_cons + 0.05*L_boundary",
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return

    device = setup_distributed()
    is_main = rank() == 0
    if world_size() != 4 or args.batch_size != 1 or args.gradient_accumulation != 1:
        raise RuntimeError("M3-noLRec requires 4 GPUs, per-GPU batch 1, accumulation 1")
    if args.max_steps != 70 or args.validate_every != 35 or args.save_every != 35:
        raise RuntimeError("M3-noLRec strict run requires 70 steps and validation/checkpoints at 35/70")
    expected_coefficients = (0.25, 0.10, 0.10, 0.05)
    actual_coefficients = (
        args.cluster_coefficient,
        args.relation_coefficient,
        args.consistency_coefficient,
        args.boundary_coefficient,
    )
    if actual_coefficients != expected_coefficients:
        raise RuntimeError(f"M3-noLRec coefficients changed: {actual_coefficients}")
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
    trainable_names = [name for name, value in backend.transformer.named_parameters() if value.requires_grad]
    if not trainable_names or any(
        ".lora_" not in name or f".{ADAPTER_NAMES['transmission']}." not in name
        for name in trainable_names
    ):
        raise RuntimeError("Trainable tensors are not exclusively LoRA_T")
    if any(value.requires_grad for value in backend.vae.parameters()):
        raise RuntimeError("VAE must remain frozen")
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
            "experiment": "M3-noLrec-70",
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "args": vars(args),
            "world_size": world_size(),
            "effective_batch": 4,
            "optimizer_steps_per_epoch": len(loader),
            "planned_optimizer_steps": 70,
            "initialization": str(args.initial),
            "initial_sha256": initial_sha,
            "loss_rec_present": False,
            "loss": preflight["loss_formula"],
            "cluster": "soft K=4 state-balanced Charbonnier against GT, averaged over I/P90",
            "relation": "Q20(T_I) sparse relation targets from fixed Q20(GT)",
            "consistency": "reflection-evidence-weighted Charbonnier between T_I and T_90",
            "boundary": "soft-boundary weighted gradient reconstruction, averaged over I/P90",
            "p90_role": "training-only second reflection observation; never Reflection GT",
            "teacher": "fixed base Qwen block 20, all LoRA disabled during teacher forward",
            "cache_manifest_sha256": sha256(args.cache_root / "manifest.json"),
            "dataset_manifest_sha256": sha256(args.data_root / "manifest.json"),
            "split_ids": splits,
            "preflight": preflight,
        }
        (args.run_dir / "run_config.json").write_text(
            json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata, indent=2, default=str), flush=True)

    dist.barrier()
    if is_main:
        baseline = validate(backend, val_loader, device, lpips_model, args.run_dir, 0, args.seed)
        updated = maybe_save_best(args.run_dir, baseline, 0, backend, args.initial)
        event = {"kind": "M2-S0-stage2-init", "step": 0, "means": baseline["means"], "best_updated": updated}
        append_jsonl(args.run_dir / "metrics.jsonl", event)
        print(json.dumps(event), flush=True)
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
            p90 = batch["p90"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            height, width = image.shape[-2:]
            posterior, reflection, boundary = pixel_maps(batch, device, height, width)
            edge_index = batch["edge_index"][0].to(device, non_blocking=True)
            edge_target = batch["edge_target"][0].to(device, non_blocking=True)
            edge_confidence = batch["edge_confidence"][0].to(device, non_blocking=True)
            expected_tokens = int(batch["token_grid"][0, 0]) * int(batch["token_grid"][0, 1])

            with torch.no_grad():
                latent_i = deterministic_encode(backend, image)
                latent_90 = deterministic_encode(backend, p90)
                prediction_90_reference = forward_from_latent(backend, latent_90).detach()

            prediction_i = forward_from_latent(backend, latent_i)
            cluster_i = cluster_loss(prediction_i, target, posterior)
            boundary_i = boundary_loss(prediction_i, target, boundary)
            consistency_i = consistency_loss(prediction_i, prediction_90_reference, reflection)
            backend.transformer.disable_lora()
            try:
                q20_prediction = q20_prediction_features(backend, prediction_i, expected_tokens)
                relation = relation_loss(q20_prediction, edge_index, edge_target, edge_confidence)
                losses_i = {
                    "cluster_i": cluster_i,
                    "relation": relation,
                    "consistency_i": consistency_i,
                    "boundary_i": boundary_i,
                }
                finite_losses(losses_i)
                loss_i = (
                    0.5 * args.cluster_coefficient * cluster_i
                    + args.relation_coefficient * relation
                    + args.consistency_coefficient * consistency_i
                    + 0.5 * args.boundary_coefficient * boundary_i
                )
                output_gradient_i = torch.autograd.grad(loss_i, prediction_i)[0]
            finally:
                backend.transformer.enable_lora()
                backend.transformer.set_adapter(ADAPTER_NAMES["transmission"])
            if not torch.isfinite(output_gradient_i).all() or output_norm(output_gradient_i) <= 0:
                raise RuntimeError("Invalid I output gradient")
            prediction_i_reference = prediction_i.detach()
            prediction_i.backward(output_gradient_i)
            output_gradient_i_norm = output_norm(output_gradient_i)
            del prediction_i, q20_prediction, output_gradient_i, loss_i
            torch.cuda.empty_cache()

            prediction_90 = forward_from_latent(backend, latent_90)
            cluster_90 = cluster_loss(prediction_90, target, posterior)
            boundary_90 = boundary_loss(prediction_90, target, boundary)
            consistency_90 = consistency_loss(prediction_90, prediction_i_reference, reflection)
            losses_90 = {
                "cluster_90": cluster_90,
                "consistency_90": consistency_90,
                "boundary_90": boundary_90,
            }
            finite_losses(losses_90)
            loss_90 = (
                0.5 * args.cluster_coefficient * cluster_90
                + args.consistency_coefficient * consistency_90
                + 0.5 * args.boundary_coefficient * boundary_90
            )
            output_gradient_90 = torch.autograd.grad(loss_90, prediction_90)[0]
            if not torch.isfinite(output_gradient_90).all() or output_norm(output_gradient_90) <= 0:
                raise RuntimeError("Invalid P90 output gradient")
            prediction_90.backward(output_gradient_90)
            output_gradient_90_norm = output_norm(output_gradient_90)

            frozen_clean = all(
                value.grad is None
                for name, value in backend.transformer.named_parameters()
                if ".lora_" not in name
            ) and all(value.grad is None for value in backend.vae.parameters())
            if not frozen_clean:
                raise RuntimeError("Frozen Qwen/VAE parameters received gradients")
            active_tensors = sync_gradients(parameters, device)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
                raise RuntimeError(f"Invalid LoRA_T gradient norm: {grad_norm}")
            move_optimizer_state(optimizer, device)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            move_optimizer_state(optimizer, torch.device("cpu"))
            global_step += 1

            cluster_mean = 0.5 * (cluster_i.detach() + cluster_90.detach())
            consistency_mean = 0.5 * (consistency_i.detach() + consistency_90.detach())
            boundary_mean = 0.5 * (boundary_i.detach() + boundary_90.detach())
            total_value = (
                args.cluster_coefficient * cluster_mean
                + args.relation_coefficient * relation.detach()
                + args.consistency_coefficient * consistency_mean
                + args.boundary_coefficient * boundary_mean
            )
            values = torch.tensor([
                float(total_value), float(cluster_mean), float(relation.detach()),
                float(consistency_mean), float(boundary_mean), float(grad_norm),
                output_gradient_i_norm, output_gradient_90_norm,
            ], device=device)
            dist.all_reduce(values)
            values.div_(world_size())
            if is_main:
                record = {
                    "kind": "train",
                    "step": global_step,
                    "epoch": epoch,
                    "loss": float(values[0]),
                    "l_rec": None,
                    "cluster": float(values[1]),
                    "relation": float(values[2]),
                    "consistency": float(values[3]),
                    "boundary": float(values[4]),
                    "grad_norm": float(values[5]),
                    "output_gradient_i_norm": float(values[6]),
                    "output_gradient_p90_norm": float(values[7]),
                    "active_gradient_tensors": active_tensors,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                }
                append_jsonl(args.run_dir / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)

            del image, p90, target, posterior, reflection, boundary
            del edge_index, edge_target, edge_confidence, latent_i, latent_90
            del prediction_90_reference, prediction_i_reference, prediction_90
            del cluster_i, cluster_90, consistency_i, consistency_90, boundary_i, boundary_90, relation
            del output_gradient_90, loss_90, losses_i, losses_90, grad_norm, values
            torch.cuda.empty_cache()

            if global_step % args.validate_every == 0:
                dist.barrier()
                if is_main:
                    last_validation = validate(backend, val_loader, device, lpips_model, args.run_dir, global_step, args.seed)
                    updated = maybe_save_best(args.run_dir, last_validation, global_step, backend, args.initial)
                    event = {
                        "kind": "validation",
                        "step": global_step,
                        "epoch": epoch,
                        "means": last_validation["means"],
                        "best_updated": updated,
                    }
                    append_jsonl(args.run_dir / "metrics.jsonl", event)
                    print(json.dumps(event), flush=True)
                dist.barrier()
            if is_main and global_step % args.save_every == 0:
                checkpoint = save_checkpoint(args.run_dir, global_step, epoch, backend, scheduler, args)
                print(f"saved {checkpoint}", flush=True)
            dist.barrier()
            if global_step >= args.max_steps:
                stop = True
                break

    if global_step != 70:
        raise RuntimeError(f"M3-noLRec stopped at {global_step}, expected 70")
    dist.barrier()
    if is_main:
        final_checkpoint = args.run_dir / "checkpoint-0000070"
        if not final_checkpoint.is_dir():
            final_checkpoint = save_checkpoint(args.run_dir, 70, epoch, backend, scheduler, args)
        if last_validation is None or int(last_validation["step"]) != 70:
            last_validation = validate(backend, val_loader, device, lpips_model, args.run_dir, 70, args.seed)
        summary = {
            "status": "complete",
            "experiment": "M3-noLrec-70",
            "loss_rec_present": False,
            "optimizer_updates": 70,
            "final_checkpoint": str(final_checkpoint),
            "best_metrics": json.loads((args.run_dir / "best_metrics.json").read_text()),
            "final_validation": last_validation["means"],
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

