"""SMA clean-content memory reads inside entirely frozen M4.

Reuses the M4 output losses and gradient control; only new readers train.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
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
    atomic_save_adapter,
    check_initial_hash,
    grouped_global_batches,
    maybe_save_best,
    sha256,
    validate,
)
from .m2b1_q20 import M2ValidationDataset
from .m3_with_lrec_100e import consistency_loss
from .sma_joint import install_teacher, teacher_context, lora_parameters
from .sma_conditioned import save_joint, calibrate, diagnostics
from .sma_conditioned import ConditionedSMA as SMA, install, C_VERSION as SMA_VERSION, forward as forward_from_latent
from .sma_data import load_manifest as load_m2_manifest
from .m4_cache import (
    CACHE_VERSION,
    EARLY_BLOCKS,
    FLOW_TIMESTEP,
    HIDDEN_SIZE,
    LATE_BLOCKS,
    MID_BLOCKS,
)
from .qwen_backend import ADAPTER_NAMES, QwenSharedBackend
from .qwen_layer_probe import deterministic_encode
from .stage2_train import transmission_loss
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
DEFAULT_CACHE = PROJECT / "data_cache/sma_m4final_v1"
DEFAULT_RUN = PROJECT / "runs/sma_e20"
DEFAULT_INITIAL = PROJECT / "runs/m4_best_newcache_e20_p4/best_transmission_lora.safetensors"
EXPECTED_INITIAL_SHA256 = "897282b1bb9cfe61f96530df72edcf8a44a066bb819a3663e9100862aefdb2a3"
MAX_ONLINE_BLOCK = max(MID_BLOCKS)


class StopAtOnlineBlock(Exception):
    pass


class M4Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        cache_root: Path,
        records: dict[str, dict],
        cache_records: dict[str, dict],
        sample_ids: list[str],
        augment: bool,
    ) -> None:
        self.data_root = data_root
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
        cache_record = self.cache_records[sample_id]
        width, height = record["target_size"]
        image = image_tensor(self.data_root / "blended" / f"{sample_id}.png")
        p90 = image_tensor(self.data_root / "reflection_90" / f"{sample_id}.png")
        target = image_tensor(
            self.data_root / "transmission_layer" / f"{sample_id}.png"
        )
        if image.shape != (3, height, width) or p90.shape != image.shape or target.shape != image.shape:
            raise ValueError(f"M4 RGB shape mismatch for {sample_id}")

        stored = safetensors.torch.load_file(
            self.cache_root / cache_record["cache"]
        )
        grid_h, grid_w = (int(value) for value in stored["token_grid_hw"])
        tokens = grid_h * grid_w
        if (grid_h, grid_w) != (height // 16, width // 16):
            raise ValueError(f"M4 token grid mismatch for {sample_id}")
        result: dict[str, object] = {
            "id": sample_id,
            "image": image,
            "p90": p90,
            "target": target,
            "late_gate": stored["late_gate"].float(),
            "late_agreement": stored["late_agreement"].float(),
            "token_grid": torch.tensor([grid_h, grid_w], dtype=torch.int32),
            "bucket": record["aspect_bucket"],
            "intentional_noisy_gt": "intentional_wrong_gt_donor" in record,
        }
        for block in EARLY_BLOCKS:
            result[f"q{block}_input"] = stored[f"q{block}_input"]
            result[f"q{block}_gt"] = stored[f"q{block}_gt"]
        for block in MID_BLOCKS:
            result[f"q{block}_gt"] = stored[f"q{block}_gt"]

        expected_feature_shape = (tokens, HIDDEN_SIZE)
        features = [
            result[f"q{block}_{role}"]
            for block in EARLY_BLOCKS
            for role in ("input", "gt")
        ] + [result[f"q{block}_gt"] for block in MID_BLOCKS]
        if any(tuple(value.shape) != expected_feature_shape for value in features):
            raise ValueError(f"M4 cached feature shape mismatch for {sample_id}")
        if result["late_gate"].shape != (tokens,) or result["late_agreement"].shape != (tokens,):
            raise ValueError(f"M4 gate shape mismatch for {sample_id}")
        if not all(torch.isfinite(value.float()).all() for value in features):
            raise ValueError(f"Non-finite M4 features for {sample_id}")
        for name in ("late_gate", "late_agreement"):
            value = result[name]
            if not torch.isfinite(value).all() or float(value.min()) < 0 or float(value.max()) > 1:
                raise ValueError(f"Invalid M4 {name} for {sample_id}")

        flipped = False
        if self.augment and torch.rand(()) < 0.5:
            result["image"] = image.flip(-1)
            result["p90"] = p90.flip(-1)
            result["target"] = target.flip(-1)
            for name in ("late_gate", "late_agreement"):
                result[name] = result[name].reshape(grid_h, grid_w).flip(1).reshape(tokens)
            for block in EARLY_BLOCKS:
                for role in ("input", "gt"):
                    name = f"q{block}_{role}"
                    result[name] = result[name].reshape(grid_h, grid_w, HIDDEN_SIZE).flip(1).reshape(tokens, HIDDEN_SIZE)
            for block in MID_BLOCKS:
                name = f"q{block}_gt"
                result[name] = result[name].reshape(grid_h, grid_w, HIDDEN_SIZE).flip(1).reshape(tokens, HIDDEN_SIZE)
            flipped = True
        result["flipped"] = flipped
        return result


def load_cache(
    cache_root: Path, data_root: Path, train_ids: list[str]
) -> tuple[dict, dict[str, dict]]:
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source_dataset", {})
    teacher = manifest.get("teacher", {})
    if not manifest.get("complete") or manifest.get("cache_version") not in {CACHE_VERSION, "m4-best-lora-multilayer-v1"}:
        raise RuntimeError("M4 cache is incomplete or incompatible")
    if source.get("train_ids") != train_ids or source.get("sample_count") != len(train_ids):
        raise RuntimeError("M4 cache train split mismatch")
    if source.get("manifest_sha256") != sha256(data_root / "manifest.json"):
        raise RuntimeError("M4 cache dataset manifest hash mismatch")
    if (
        teacher.get("early_blocks") != list(EARLY_BLOCKS)
        or teacher.get("mid_blocks") != list(MID_BLOCKS)
        or teacher.get("late_blocks") != list(LATE_BLOCKS)
        or teacher.get("flow_timestep") != FLOW_TIMESTEP
    ):
        raise RuntimeError("M4 cache teacher identity mismatch")
    records = {record["id"]: record for record in manifest["samples"]}
    if set(records) != set(train_ids):
        raise RuntimeError("M4 cache IDs differ from the training split")
    for sample_id in train_ids:
        path = cache_root / records[sample_id]["cache"]
        if not path.is_file() or sha256(path) != records[sample_id]["cache_sha256"]:
            raise RuntimeError(f"M4 cache hash mismatch for {sample_id}")
    return manifest, records


def online_prediction_features(
    backend: QwenSharedBackend,
    prediction_leaf: torch.Tensor,
    expected_tokens: int,
) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    online_blocks = EARLY_BLOCKS + MID_BLOCKS
    for block_number in online_blocks:
        index = block_number - 1

        def hook(_module, _inputs, output, block_number=block_number):
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError(f"Unexpected online block {block_number} output")
            captured[block_number] = output[1]
            if block_number == MAX_ONLINE_BLOCK:
                raise StopAtOnlineBlock

        handles.append(
            backend.transformer.transformer_blocks[index].register_forward_hook(hook)
        )
    # The caller keeps LoRA disabled until every teacher-derived VJP finishes.
    # Gradient checkpointing recomputes these blocks during autograd; changing the
    # adapter state between the forward and recomputation corrupts checkpoint metadata.
    try:
        latent = deterministic_encode(backend, prediction_leaf)
        try:
            backend.upstream.flow_step(
                latent, backend.transformer, backend.vae, backend.embeddings
            )
        except StopAtOnlineBlock:
            pass
        else:
            raise RuntimeError("Online teacher did not stop at block 41")
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(online_blocks):
        raise RuntimeError(f"Missing online features: {set(online_blocks)-set(captured)}")
    for block, feature in captured.items():
        if feature.ndim != 3 or feature.shape[1:] != (expected_tokens, HIDDEN_SIZE):
            raise ValueError(f"Unexpected online Q{block} shape: {feature.shape}")
    return captured


def cosine_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return 1.0 - (
        F.normalize(first.float(), dim=-1, eps=1e-6)
        * F.normalize(second.float(), dim=-1, eps=1e-6)
    ).sum(-1)


def token_weights(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate = batch["late_gate"].to(device, non_blocking=True).float()
    agreement = batch["late_agreement"].to(device, non_blocking=True).float()
    confidence = 0.5 + 0.5 * agreement
    keep = (1.0 - gate) * confidence
    restore = gate * confidence
    return gate, keep, restore


def texture_loss(
    features: dict[int, torch.Tensor],
    batch: dict,
    device: torch.device,
    keep: torch.Tensor,
    restore: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for block in EARLY_BLOCKS:
        q_prediction = features[block]
        q_input = batch[f"q{block}_input"].to(device, non_blocking=True)
        q_gt = batch[f"q{block}_gt"].to(device, non_blocking=True)
        d_keep = cosine_distance(q_prediction, q_input)
        d_restore = cosine_distance(q_prediction, q_gt)
        numerator = (keep * d_keep + restore * d_restore).sum()
        denominator = (keep + restore).sum().clamp_min(1e-6)
        losses.append(numerator / denominator)
    return torch.stack(losses).mean()


def centered_normalized(value: torch.Tensor) -> torch.Tensor:
    centered = value.float() - value.float().mean(dim=1, keepdim=True)
    return F.normalize(centered, dim=-1, eps=1e-6)


def neighbor_relation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    prediction = centered_normalized(prediction).reshape(1, grid_h, grid_w, HIDDEN_SIZE)
    target = centered_normalized(target).reshape(1, grid_h, grid_w, HIDDEN_SIZE)
    confidence = confidence.reshape(1, grid_h, grid_w)
    pred_x = (prediction[:, :, 1:] * prediction[:, :, :-1]).sum(-1)
    gt_x = (target[:, :, 1:] * target[:, :, :-1]).sum(-1)
    weight_x = 0.5 * (confidence[:, :, 1:] + confidence[:, :, :-1])
    pred_y = (prediction[:, 1:] * prediction[:, :-1]).sum(-1)
    gt_y = (target[:, 1:] * target[:, :-1]).sum(-1)
    weight_y = 0.5 * (confidence[:, 1:] + confidence[:, :-1])
    loss_x = (
        F.smooth_l1_loss(pred_x, gt_x, reduction="none") * weight_x
    ).sum() / weight_x.sum().clamp_min(1e-6)
    loss_y = (
        F.smooth_l1_loss(pred_y, gt_y, reduction="none") * weight_y
    ).sum() / weight_y.sum().clamp_min(1e-6)
    return 0.5 * (loss_x + loss_y)


def semantic_loss(
    features: dict[int, torch.Tensor],
    batch: dict,
    device: torch.device,
    confidence: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    content_losses = []
    relation_losses = []
    for block in MID_BLOCKS:
        prediction = features[block]
        target = batch[f"q{block}_gt"].to(device, non_blocking=True)
        pred_centered = centered_normalized(prediction)
        target_centered = centered_normalized(target)
        content = 1.0 - (pred_centered * target_centered).sum(-1)
        content_losses.append(
            (content * confidence).sum() / confidence.sum().clamp_min(1e-6)
        )
        relation_losses.append(
            neighbor_relation_loss(
                prediction, target, confidence, grid_h, grid_w
            )
        )
    content = torch.stack(content_losses).mean()
    relation = torch.stack(relation_losses).mean()
    return 0.7 * content + 0.3 * relation, content, relation


def spatial_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    source: torch.Tensor,
    gate: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = prediction.shape[-2:]
    gate_pixel = F.interpolate(
        gate.reshape(1, 1, grid_h, grid_w),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).clamp(0, 1)
    prediction01 = ((prediction.float() + 1) * 0.5).clamp(0, 1)
    target01 = ((target.float() + 1) * 0.5).clamp(0, 1)
    source01 = ((source.float() + 1) * 0.5).clamp(0, 1)
    error_gt = torch.sqrt((prediction01 - target01).square() + 1e-6).mean(1, keepdim=True)
    error_keep = torch.sqrt((prediction01 - source01).square() + 1e-6).mean(1, keepdim=True)
    weight = 1.0 + 2.0 * gate_pixel
    weight = weight / weight.mean().clamp_min(1e-6)
    weighted_restore = (weight * error_gt).mean()
    low_response_keep = ((1.0 - gate_pixel) * error_keep).sum() / (
        (1.0 - gate_pixel).sum().clamp_min(1e-6)
    )
    return 0.75 * weighted_restore + 0.25 * low_response_keep, weighted_restore, low_response_keep


def tensor_norm(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().sum().sqrt()


class GradientController:
    def __init__(
        self,
        targets: dict[str, float],
        warmup_steps: int,
        ema_decay: float = 0.9,
        minimum: float = 1e-4,
        maximum: float = 10.0,
        total_cap: float = 0.25,
    ) -> None:
        self.targets = targets
        self.warmup_steps = warmup_steps
        self.ema_decay = ema_decay
        self.minimum = minimum
        self.maximum = maximum
        self.total_cap = total_cap
        self.ema: dict[str, float] = {}

    def combine(
        self,
        base: torch.Tensor,
        auxiliaries: dict[str, torch.Tensor],
        step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        base_norm = float(tensor_norm(base).detach())
        ramp = min(1.0, step / max(self.warmup_steps, 1))
        scaled = {}
        logs = {}
        for name, gradient in auxiliaries.items():
            gradient_norm = float(tensor_norm(gradient).detach())
            target = self.targets[name] * ramp
            raw = target * base_norm / max(gradient_norm, 1e-12)
            raw = min(self.maximum, max(self.minimum, raw))
            previous = self.ema.get(name, raw)
            scale = self.ema_decay * previous + (1.0 - self.ema_decay) * raw
            self.ema[name] = scale
            scaled[name] = scale * gradient
            logs[f"{name}_raw_norm"] = gradient_norm
            logs[f"{name}_scale"] = scale

        auxiliary = sum(scaled.values(), torch.zeros_like(base))
        auxiliary_norm = float(tensor_norm(auxiliary).detach())
        cap = self.total_cap * base_norm
        cap_scale = min(1.0, cap / max(auxiliary_norm, 1e-12))
        combined = base + cap_scale * auxiliary
        logs["base_output_gradient_norm"] = base_norm
        logs["aux_output_gradient_norm_before_cap"] = auxiliary_norm
        logs["aux_cap_scale"] = cap_scale
        logs["actual_aux_base_ratio"] = float(
            tensor_norm(cap_scale * auxiliary).detach()
        ) / max(base_norm, 1e-12)
        return combined, logs


def finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value):
        raise RuntimeError(f"Non-finite M4 loss {name}: {float(value.detach())}")


def save_latest(
    run_dir: Path,
    report: dict,
    step: int,
    epoch: int,
    backend: QwenSharedBackend,
) -> None:
    atomic_save_adapter(run_dir / "latest_sma.safetensors", backend)
    prediction_path = run_dir / "validation" / f"step_{step:06d}"
    if prediction_path.exists():
        latest_dir = run_dir / "latest_validation"
        if latest_dir.exists(): shutil.rmtree(latest_dir)
        shutil.copytree(prediction_path, latest_dir)
    (run_dir / "latest_metrics.json").write_text(
        json.dumps(
            {
                "val_l1": float(report["means"]["l1"]),
                "val_psnr": report["means"]["psnr"],
                "val_ssim": report["means"]["ssim"],
                "val_lpips_squeeze": report["means"]["lpips_squeeze"],
                "step": step,
                "epoch": epoch,
                "criterion": "latest completed epoch validation",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )



def atomic_save_adapter(path, backend):
    if getattr(backend, "joint_training", False):
        return save_joint(path, backend, EXPECTED_INITIAL_SHA256)
    from .m4_cache import atomic_safetensors
    atomic_safetensors(path, backend.sma.state_dict(), {"experiment": "SMA", "architecture": SMA_VERSION, "base_m4_sha256": EXPECTED_INITIAL_SHA256})


def maybe_save_best(run_dir, report, step, backend, initial):
    record_path = run_dir / "best_metrics.json"
    previous = json.loads(record_path.read_text()) if record_path.is_file() else None
    if previous is not None and report["means"]["l1"] >= previous["val_l1"]: return False
    atomic_save_adapter(run_dir / "best_sma.safetensors", backend)
    prediction_path = run_dir / "validation" / f"step_{step:06d}"
    if step == 0 and prediction_path.exists():
        (run_dir / "initial_prediction_hashes.json").write_text(json.dumps({p.name: sha256(p) for p in (prediction_path / "predictions").glob("*.png")}, indent=2)+"\n")
    if prediction_path.exists():
        best_dir = run_dir / "best_validation"
        if best_dir.exists(): shutil.rmtree(best_dir)
        shutil.copytree(prediction_path, best_dir)
    record_path.write_text(json.dumps({"val_l1": report["means"]["l1"], "val_psnr": report["means"]["psnr"], "val_ssim": report["means"]["ssim"], "val_lpips_squeeze": report["means"]["lpips_squeeze"], "step": step, "criterion": "minimum saved-PNG validation macro L1"}, indent=2)+"\n")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--initial-sha256", default=EXPECTED_INITIAL_SHA256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--gradient-ramp-steps", type=int, default=36)
    parser.add_argument("--spatial-gradient-ratio", type=float, default=0.08)
    parser.add_argument("--texture-gradient-ratio", type=float, default=0.08)
    parser.add_argument("--semantic-gradient-ratio", type=float, default=0.08)
    parser.add_argument("--aux-gradient-cap", type=float, default=0.25)
    parser.add_argument("--consistency-coefficient", type=float, default=0.10)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--edge-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--memory-file", type=Path, default=PROJECT / "runs/sma_memory_pretrain/memory.safetensors")
    parser.add_argument("--continue-run", type=Path, help="Completed parent run; restore latest weights only, epochs denotes cumulative target")
    parser.add_argument("--joint-lora", action="store_true")
    parser.add_argument("--lora-learning-rate", type=float, default=5e-6)
    parser.add_argument("--gamma-learning-rate", type=float, default=5e-3)
    parser.add_argument("--calibration-start-epoch", type=int, default=1)
    parser.add_argument("--calibration-interval", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.joint_lora or args.continue_run: raise ValueError("C requires fresh joint training")
    if args.calibration_interval < 1 or args.calibration_start_epoch < 0: raise ValueError("Invalid calibration schedule")
    if args.joint_lora and args.continue_run:raise ValueError("Joint B starts fresh, no legacy continuation")
    if args.joint_lora and args.lora_learning_rate <= 0:raise ValueError("Invalid LoRA LR")
    args.data_root = args.data_root.resolve()
    args.cache_root = args.cache_root.resolve()
    args.run_dir = args.run_dir.resolve()
    args.initial = args.initial.resolve()
    manifest, records, splits = load_m2_manifest(args.data_root)
    cache_manifest, cache_records = load_cache(
        args.cache_root, args.data_root, splits["train"]
    )
    train_data = M4Dataset(
        args.data_root,
        args.cache_root,
        records,
        cache_records,
        splits["train"],
        False,
    )
    val_data = M2ValidationDataset(
        args.data_root, records, splits["validation"]
    )
    batches = grouped_global_batches(train_data, args.seed, 0, 4)
    updates_per_epoch = len(batches)
    continuation = None
    if args.continue_run:
        from .sma_continue import audit_parent
        continuation = audit_parent(args.continue_run, args.data_root, args.cache_root,
                                    args.initial_sha256, updates_per_epoch, args.epochs)
    initial_step = continuation['initial_global_step'] if continuation else 0
    start_epoch = continuation['completed_epochs'] if continuation else 0
    full_schedule_steps = updates_per_epoch * args.epochs
    planned_steps = (
        min(args.max_steps, full_schedule_steps)
        if args.max_steps > 0
        else full_schedule_steps
    )
    validate_every = args.validate_every or updates_per_epoch
    if planned_steps <= initial_step:
        raise ValueError("The cumulative step target must exceed the parent latest step")
    preflight = {
        "status": "preflight_ok",
        "experiment": "C1-semantic-conditioned-edit",
        "label_noise": manifest.get("label_noise"),
        "train": len(train_data),
        "validation": len(val_data),
        "sealed_test": len(splits["test"]),
        "world_size_expected": 4,
        "updates_per_epoch": updates_per_epoch,
        "epochs": args.epochs,
        "planned_steps": planned_steps,
        "initial_global_step": initial_step,
        "new_optimizer_updates": planned_steps-initial_step,
        "continuation": continuation,
        "validate_every": validate_every,
        "early_stopping_patience_epochs": args.early_stopping_patience,
        "checkpoint_policy": ("best/latest joint LoRA+SMA only; fixed teacher referenced by SHA256" if args.joint_lora else "best/latest SMA only; fixed M4 weights referenced by SHA256"),
        "cache_version": cache_manifest["cache_version"],
        "cache_gib": cache_manifest["storage"]["total_gib"],
        "early_blocks": list(EARLY_BLOCKS),
        "mid_blocks": list(MID_BLOCKS),
        "late_blocks": list(LATE_BLOCKS),
        "initialization": str(args.initial),
        "joint_lora": args.joint_lora,
        "lora_learning_rate": args.lora_learning_rate if args.joint_lora else None,
        "augmentation": "disabled: cached Qwen features are tied to absolute token positions",
        "auxiliary_gradient_targets": {
            "spatial": args.spatial_gradient_ratio,
            "texture": args.texture_gradient_ratio,
            "semantic": args.semantic_gradient_ratio,
            "combined_cap": args.aux_gradient_cap,
        },
    }
    if not continuation and not args.memory_file.is_file(): raise FileNotFoundError(args.memory_file)
    if cache_manifest["teacher"].get("adapter_sha256") != args.initial_sha256:
        raise RuntimeError("Output teacher cache must match frozen final M4")
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return

    device = setup_distributed()
    is_main = rank() == 0
    if world_size() != 4 or args.batch_size != 1 or args.gradient_accumulation != 1:
        raise RuntimeError("M4 requires four GPUs, per-GPU batch 1, accumulation 1")
    if planned_steps <= 0 or validate_every <= 0:
        raise ValueError("M4 planned steps and validation interval must be positive")
    if args.early_stopping_patience < 0:
        raise ValueError("Early-stopping patience must be non-negative")
    if args.early_stopping_patience and validate_every != updates_per_epoch:
        raise ValueError("Epoch-based early stopping requires validation once per epoch")
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
    initial_sha = check_initial_hash(args.initial, args.initial_sha256, device)
    # Every rank loads the fixed final M4; synchronizing only new readers would
    # otherwise leave nonzero ranks running the official WindowSeat adapter.
    backend.set_trainable_branch("transmission")
    load_initial(backend, args.initial, device)
    backend.set_trainable_branch(None)
    backend.joint_training = args.joint_lora
    teacher_audit = install_teacher(backend) if args.joint_lora else None
    sma = SMA().to(device)
    if continuation:
        initial_sma_path = Path(continuation['checkpoint'])
        if sha256(initial_sma_path) != continuation['checkpoint_sha256']:
            raise RuntimeError('Parent checkpoint changed before load')
    else:
        initial_sma_path = args.memory_file
        memory_report = json.loads((args.memory_file.parent / "report.json").read_text())
        if memory_report["teacher_sha256"] != initial_sha or memory_report["cache_manifest_sha256"] != sha256(args.cache_root / "manifest.json"):
            raise RuntimeError("Memory / teacher / cache identity mismatch")
        if memory_report["memory_sha256"] != sha256(args.memory_file):
            raise RuntimeError("Pretrained memory checksum mismatch")
    sma.load_memory(safetensors.torch.load_file(initial_sma_path, device=str(device)))
    if any(float(m.gamma) != 0 for m in sma.readers.values()):
        raise RuntimeError('C must start with zero modulation')
    sma.freeze_memory()
    runtime = install(backend, sma)
    reader_parameters = list(sma.readers.parameters())
    student_parameters = lora_parameters(backend) if args.joint_lora else []
    parameters = reader_parameters + student_parameters
    sync_initial_parameters(parameters)
    if any(p.requires_grad and id(p) not in {id(v) for v in student_parameters} for p in backend.transformer.parameters()) or any(p.requires_grad for p in backend.vae.parameters()):
        raise RuntimeError("Only student LoRA may train in the backbone")
    backend.transformer.train()
    backend.vae.eval()
    seed_everything(args.seed + rank())

    optimizer = torch.optim.AdamW(
        [{"params": [p for n,p in sma.readers.named_parameters() if not n.endswith("gamma")], "lr": args.learning_rate},
         {"params": [p for n,p in sma.readers.named_parameters() if n.endswith("gamma")], "lr": args.gamma_learning_rate, "weight_decay": 0.0}],
        lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = make_scheduler(
        optimizer, min(args.warmup_steps, planned_steps-initial_step), planned_steps-initial_step
    )
    lora_optimizer = bnb.optim.PagedAdamW8bit(student_parameters, lr=args.lora_learning_rate, weight_decay=args.weight_decay) if args.joint_lora else None
    lora_scheduler = make_scheduler(lora_optimizer, min(args.warmup_steps, planned_steps-initial_step), planned_steps-initial_step) if lora_optimizer else None
    if lora_optimizer:lora_optimizer.zero_grad(set_to_none=True)
    controller = GradientController(
        {
            "spatial": args.spatial_gradient_ratio,
            "texture": args.texture_gradient_ratio,
            "semantic": args.semantic_gradient_ratio,
        },
        args.gradient_ramp_steps,
        total_cap=args.aux_gradient_cap,
    )
    lpips_model = None
    if is_main:
        lpips_model = lpips.LPIPS(net="squeeze", verbose=False).eval().cpu()
        for parameter in lpips_model.parameters():
            parameter.requires_grad_(False)
        config = {
            "experiment": "C1-semantic-conditioned-edit",
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "args": vars(args),
            "world_size": world_size(),
            "effective_batch": 4,
            "updates_per_epoch": updates_per_epoch,
            "planned_steps": planned_steps,
            "initial_sha256": initial_sha,
            "cache_manifest_sha256": sha256(args.cache_root / "manifest.json"),
            "dataset_manifest_sha256": sha256(args.data_root / "manifest.json"),
            "roles": {
                "late": "Q52/Q54/Q56 offline ensemble gate",
                "early": "Q16/Q20 keep-input outside gate and restore-GT inside gate",
                "middle": "Q37/Q39/Q41 centered content plus 4-neighbor relation",
            },
            "base": f"mean I/P90 of L1 + {args.ssim_weight}*(1-SSIM) + {args.edge_weight}*edge; {args.consistency_coefficient} polar consistency",
            "gradient_control": preflight["auxiliary_gradient_targets"],
            "memory_strategy": "frozen Q37 clean memory; four Q39/Q41 image FFN rank128 modulators; reliability calibrated by current-student off/on counterfactuals; fixed M4 loss teacher",
            "trainable_parameters": sum(p.numel() for p in parameters),
            "memory_file_sha256": sha256(initial_sma_path),
            "continuation": continuation,
            "sma_architecture": SMA_VERSION,
            "m4_frozen": not args.joint_lora,
            "teacher_audit": teacher_audit,
            "optimizer_policy": "condition/gates FP32 AdamW; gamma LR5e-3; student LoRA paged 8-bit AdamW with CPU state offload" if args.joint_lora else "readers FP32 AdamW",
            "student_lora_parameters": sum(p.numel() for p in student_parameters),
            "reader_parameters": sum(p.numel() for p in reader_parameters),
            "memory_frozen": True,
            "condition_mode": "C1; no incorrect conditions or GT",
            "calibration_start_epoch_zero_indexed": args.calibration_start_epoch,
            "calibration_interval": args.calibration_interval,
            "gamma_policy": "zero init; 0.25*tanh(raw); raw gamma LR configurable, default5e-3",
            "reliability_cap": "10% global DDP-averaged gate main gradient; coefficient <=1",
            "preflight": preflight,
        }
        (args.run_dir / "run_config.json").write_text(
            json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps(config, indent=2, default=str), flush=True)

    dist.barrier()
    if is_main:
        if continuation:
            parent = Path(continuation['parent_run'])
            for name in ('best_sma.safetensors', 'best_metrics.json'):
                shutil.copy2(parent/name, args.run_dir/name)
            shutil.copytree(parent/'best_validation', args.run_dir/'best_validation')
        baseline = validate(
            backend, val_loader, device, lpips_model, args.run_dir, initial_step, args.seed
        )
        if continuation:
            parent_metrics = json.loads((Path(continuation['parent_run'])/'latest_metrics.json').read_text())
            if abs(baseline['means']['psnr']-parent_metrics['val_psnr']) > 1e-4 or abs(baseline['means']['ssim']-parent_metrics['val_ssim']) > 1e-5:
                raise RuntimeError('Loaded continuation does not reproduce parent latest validation')
        updated = maybe_save_best(
            args.run_dir, baseline, initial_step, backend, args.initial
        )
        if continuation:
            save_latest(args.run_dir, baseline, initial_step, start_epoch-1, backend)
        event = {
            "kind": "continuation_init" if continuation else "m4_frozen_init",
            "step": initial_step,
            "means": baseline["means"],
            "best_updated": updated,
        }
        append_jsonl(args.run_dir / "metrics.jsonl", event)
        print(json.dumps(event), flush=True)
        shutil.rmtree(
            args.run_dir / "validation" / f"step_{initial_step:06d}", ignore_errors=True
        )
    dist.barrier()

    student_parameter_ids = {id(p) for p in student_parameters}
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    global_step = initial_step
    last_validation = None
    stop = False
    early_stopped = False
    epochs_without_improvement = continuation['epochs_without_improvement'] if continuation else 0
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        sampler.set_epoch(epoch)
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            p90 = batch["p90"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            grid_h, grid_w = (int(value) for value in batch["token_grid"][0])
            expected_tokens = grid_h * grid_w
            gate, keep, restore = token_weights(batch, device)
            confidence = 0.5 + 0.5 * batch["late_agreement"].to(
                device, non_blocking=True
            ).float()

            with torch.no_grad():
                latent_i = deterministic_encode(backend, image)
                latent_90 = deterministic_encode(backend, p90)
                prediction_i_probe = forward_from_latent(
                    backend, latent_i
                ).detach()
                prediction_90_reference = forward_from_latent(
                    backend, latent_90
                ).detach()

            leaf = prediction_i_probe.detach().requires_grad_(True)
            rec_i_probe, rec_i_parts = transmission_loss(
                leaf, target, args.ssim_weight, args.edge_weight
            )
            gate_pixel_i = F.interpolate(
                gate.reshape(1, 1, grid_h, grid_w),
                size=leaf.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            cons_i = consistency_loss(
                leaf, prediction_90_reference, gate_pixel_i
            )
            base_i = 0.5 * (rec_i_probe + args.consistency_coefficient * cons_i)
            spatial, spatial_restore, spatial_keep = spatial_loss(
                leaf, target, image, gate, grid_h, grid_w
            )
            # Same fixed M4 teacher as the cache; only new modules are bypassed.
            with teacher_context(backend, runtime):
              try:
                  features = online_prediction_features(
                      backend, leaf, expected_tokens
                  )
                  texture = texture_loss(
                      features, batch, device, keep, restore
                  )
                  semantic, semantic_content, semantic_relation = semantic_loss(
                      features,
                      batch,
                      device,
                      confidence,
                      grid_h,
                      grid_w,
                  )
                  for name, value in {
                      "base_i": base_i,
                      "spatial": spatial,
                      "texture": texture,
                      "semantic": semantic,
                  }.items():
                      finite(name, value)
                  base_gradient = torch.autograd.grad(
                      base_i, leaf, retain_graph=True
                  )[0]
                  spatial_gradient = torch.autograd.grad(
                      spatial, leaf, retain_graph=True
                  )[0]
                  texture_gradient = torch.autograd.grad(
                      texture, leaf, retain_graph=True
                  )[0]
                  semantic_gradient = torch.autograd.grad(
                      semantic, leaf
                  )[0]
              finally:
                  pass
            output_gradient_i, gradient_log = controller.combine(
                base_gradient,
                {
                    "spatial": spatial_gradient,
                    "texture": texture_gradient,
                    "semantic": semantic_gradient,
                },
                global_step + 1,
            )
            if not torch.isfinite(output_gradient_i).all() or float(tensor_norm(output_gradient_i)) <= 0:
                raise RuntimeError("Invalid M4 I output gradient")
            rec_i_log = rec_i_probe.detach()
            cons_i_log = cons_i.detach()
            spatial_log = spatial.detach()
            spatial_restore_log = spatial_restore.detach()
            spatial_keep_log = spatial_keep.detach()
            texture_log = texture.detach()
            semantic_log = semantic.detach()
            semantic_content_log = semantic_content.detach()
            semantic_relation_log = semantic_relation.detach()
            del features, leaf, base_gradient, spatial_gradient, texture_gradient, semantic_gradient
            del base_i, rec_i_probe, rec_i_parts, cons_i, gate_pixel_i
            del spatial, spatial_restore, spatial_keep
            del texture, semantic, semantic_content, semantic_relation
            torch.cuda.empty_cache()

            prediction_i = forward_from_latent(backend, latent_i)
            prediction_i.backward(output_gradient_i)
            prediction_i_reference = prediction_i.detach()
            del prediction_i, output_gradient_i
            torch.cuda.empty_cache()

            prediction_90 = forward_from_latent(backend, latent_90)
            rec_90, rec_90_parts = transmission_loss(
                prediction_90, target, args.ssim_weight, args.edge_weight
            )
            cons_90 = consistency_loss(
                prediction_90,
                prediction_i_reference,
                F.interpolate(
                    gate.reshape(1, 1, grid_h, grid_w),
                    size=prediction_90.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ),
            )
            base_90 = 0.5 * (rec_90 + args.consistency_coefficient * cons_90)
            finite("base_90", base_90)
            output_gradient_90 = torch.autograd.grad(
                base_90, prediction_90
            )[0]
            prediction_90.backward(output_gradient_90)

            calibration_log = {"calibration": False}
            if epoch >= args.calibration_start_epoch and (global_step+1) % args.calibration_interval == 0:
                calibration_log = calibrate(backend, latent_i, target, global_step+1, epoch)

            frozen_clean = all(
                value.grad is None
                for name, value in backend.transformer.named_parameters()
                if id(value) not in student_parameter_ids
            ) and all(
                value.grad is None for value in backend.vae.parameters()
            )
            if not frozen_clean:
                raise RuntimeError("Frozen Qwen/VAE parameters received gradients")
            active_tensors = sync_gradients(parameters, device)
            reader_norm = torch.stack([p.grad.detach().float().square().sum() for p in reader_parameters if p.grad is not None]).sum().sqrt()
            lora_norm = torch.stack([p.grad.detach().float().square().sum() for p in student_parameters if p.grad is not None]).sum().sqrt() if student_parameters else torch.zeros((),device=device)
            if args.joint_lora and (not torch.isfinite(lora_norm) or lora_norm <= 0):raise RuntimeError("Missing student LoRA gradients")
            probe_parameter = max(student_parameters, key=lambda p:float(p.grad.float().square().sum()) if p.grad is not None else -1) if args.joint_lora and global_step == initial_step else None
            lora_before = probe_parameter.detach().clone() if probe_parameter is not None else None
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, args.max_grad_norm
            )
            if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
                raise RuntimeError(f"Invalid SMA reader gradient norm: {grad_norm}")
            before_update = sma.readers["39_in"].gamma.detach().clone() if global_step == initial_step else None
            optimizer.step()
            if lora_optimizer:
                move_optimizer_state(lora_optimizer, device)
                lora_optimizer.step()
                lora_optimizer.zero_grad(set_to_none=True)
                move_optimizer_state(lora_optimizer, torch.device('cpu'))
            if before_update is not None and torch.equal(before_update, sma.readers["39_in"].gamma):
                raise RuntimeError("First SMA reader output projection failed to update")
            if lora_before is not None and torch.equal(lora_before, probe_parameter):raise RuntimeError("Student LoRA failed first update")
            scheduler.step()
            if lora_scheduler:lora_scheduler.step()
            if any(p.grad is not None for p in sma.memory.parameters()):
                raise RuntimeError("Frozen content memory received gradients")
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            values = torch.tensor(
                [
                    float(rec_i_log),
                    float(rec_90.detach()),
                    float(cons_i_log),
                    float(cons_90.detach()),
                    float(spatial_log),
                    float(spatial_restore_log),
                    float(spatial_keep_log),
                    float(texture_log),
                    float(semantic_log),
                    float(semantic_content_log),
                    float(semantic_relation_log),
                    float(grad_norm),
                    gradient_log["actual_aux_base_ratio"],
                    gradient_log["spatial_scale"],
                    gradient_log["texture_scale"],
                    gradient_log["semantic_scale"],
                ],
                device=device,
            )
            dist.all_reduce(values)
            values.div_(world_size())
            noisy_count = batch["intentional_noisy_gt"].to(device, dtype=torch.int32).sum()
            dist.all_reduce(noisy_count)
            if is_main:
                event = {
                    "kind": "train",
                    "step": global_step,
                    "epoch": epoch,
                    "noisy_gt_samples_in_update": int(noisy_count.item()),
                    "rec_i": float(values[0]),
                    "rec_p90": float(values[1]),
                    "consistency_i": float(values[2]),
                    "consistency_p90": float(values[3]),
                    "spatial": float(values[4]),
                    "spatial_restore": float(values[5]),
                    "spatial_keep": float(values[6]),
                    "texture_q16_q20": float(values[7]),
                    "semantic_q37_q39_q41": float(values[8]),
                    "semantic_content": float(values[9]),
                    "semantic_relation": float(values[10]),
                    "grad_norm": float(values[11]),
                    "actual_aux_base_ratio": float(values[12]),
                    "spatial_scale": float(values[13]),
                    "texture_scale": float(values[14]),
                    "semantic_scale": float(values[15]),
                    "active_gradient_tensors": active_tensors,
                    "backbone_vae_memory_frozen": True,
                    "reader_first_update_verified": True,
                    "joint_lora": args.joint_lora,
                    "student_lora_grad_norm": float(lora_norm),
                    "reader_grad_norm": float(reader_norm),
                    "student_lora_update_verified": args.joint_lora,
                    "lora_lr": lora_scheduler.get_last_lr()[0] if lora_scheduler else None,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.monotonic() - started,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                }
                event["gamma_lr"] = scheduler.get_last_lr()[1]
                event.update(calibration_log)
                event.update(diagnostics(sma, runtime))
                append_jsonl(args.run_dir / "metrics.jsonl", event)
                print(json.dumps(event), flush=True)

            del image, p90, target, gate, keep, restore, confidence
            del latent_i, latent_90, prediction_i_probe, prediction_90_reference
            del prediction_i_reference, prediction_90, output_gradient_90
            del rec_90, rec_90_parts, cons_90, base_90
            del rec_i_log, cons_i_log, spatial_log, spatial_restore_log, spatial_keep_log
            del texture_log, semantic_log, semantic_content_log, semantic_relation_log
            del grad_norm, values
            torch.cuda.empty_cache()

            if global_step % validate_every == 0:
                dist.barrier()
                early_stop_flag = torch.zeros((), dtype=torch.int32, device=device)
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
                    updated = maybe_save_best(
                        args.run_dir,
                        last_validation,
                        global_step,
                        backend,
                        args.initial,
                    )
                    save_latest(
                        args.run_dir,
                        last_validation,
                        global_step,
                        epoch,
                        backend,
                    )
                    epochs_without_improvement = (
                        0 if updated else epochs_without_improvement + 1
                    )
                    should_early_stop = (
                        args.early_stopping_patience > 0
                        and epochs_without_improvement
                        >= args.early_stopping_patience
                    )
                    if should_early_stop:
                        early_stop_flag.fill_(1)
                    event = {
                        "kind": "validation",
                        "step": global_step,
                        "epoch": epoch,
                        "means": last_validation["means"],
                        "best_updated": updated,
                        "epochs_without_improvement": epochs_without_improvement,
                        "early_stop_triggered": should_early_stop,
                    }
                    append_jsonl(args.run_dir / "metrics.jsonl", event)
                    print(json.dumps(event), flush=True)
                    shutil.rmtree(
                        args.run_dir
                        / "validation"
                        / f"step_{global_step:06d}",
                        ignore_errors=True,
                    )
                dist.broadcast(early_stop_flag, src=0)
                if bool(early_stop_flag.item()):
                    early_stopped = True
                    stop = True
                dist.barrier()

            if global_step >= planned_steps:
                stop = True
                break

    if global_step != planned_steps and not early_stopped:
        raise RuntimeError(f"M4 stopped at {global_step}, expected {planned_steps}")
    dist.barrier()
    if global_step % validate_every != 0:
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
            updated = maybe_save_best(
                args.run_dir,
                last_validation,
                global_step,
                backend,
                args.initial,
            )
            save_latest(
                args.run_dir,
                last_validation,
                global_step,
                epoch,
                backend,
            )
            event = {
                "kind": "final_validation",
                "step": global_step,
                "epoch": epoch,
                "means": last_validation["means"],
                "best_updated": updated,
            }
            append_jsonl(args.run_dir / "metrics.jsonl", event)
            print(json.dumps(event), flush=True)
            shutil.rmtree(
                args.run_dir / "validation" / f"step_{global_step:06d}",
                ignore_errors=True,
            )
        dist.barrier()
    if is_main:
        summary = {
            "status": "complete",
            "experiment": "C1-semantic-conditioned-edit",
            "epochs_requested": args.epochs,
            "epochs_completed": epoch + 1,
            "optimizer_updates": global_step,
            "joint_lora": args.joint_lora,
            "optimizer_updates_this_run": global_step-initial_step,
            "epochs_completed_before_start": start_epoch,
            "epochs_completed_this_run": epoch+1-start_epoch,
            "continuation": continuation,
            "early_stopped": early_stopped,
            "early_stopping_patience_epochs": args.early_stopping_patience,
            "epochs_without_improvement": epochs_without_improvement,
            "checkpoint_policy": ("best/latest joint LoRA+SMA only; fixed teacher referenced by SHA256" if args.joint_lora else "best/latest SMA only; fixed M4 weights referenced by SHA256"),
            "best_lora": str(
                args.run_dir / "best_sma.safetensors"
            ),
            "best_metrics": json.loads(
                (args.run_dir / "best_metrics.json").read_text()
            ),
            "latest_lora": str(
                args.run_dir / "latest_sma.safetensors"
            ),
            "latest_metrics": json.loads(
                (args.run_dir / "latest_metrics.json").read_text()
            ),
            "final_validation": (
                last_validation["means"] if last_validation else None
            ),
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
