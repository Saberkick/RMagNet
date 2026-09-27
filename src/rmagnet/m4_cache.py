"""Build the M4 multi-layer frozen-Qwen supervision cache."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F

from .m2a_prepare import dataset_state, sample_paths
from .qwen_backend import QwenSharedBackend
from .qwen_layer_probe import deterministic_encode
from .stage1_train import image_tensor


ROOT = Path("/share/linmingheng-local/xuke")
PROJECT = ROOT / "RMagNet"
DEFAULT_DATA = ROOT / "datasets/rmagnet_m2_aspect"
DEFAULT_OUTPUT = PROJECT / "data_cache/m4_multilayer_v1"
CACHE_VERSION = "m4-multilayer-v1"
EARLY_BLOCKS = (16, 20)
MID_BLOCKS = (37, 39, 41)
LATE_BLOCKS = (52, 54, 56)
ALL_BLOCKS = EARLY_BLOCKS + MID_BLOCKS + LATE_BLOCKS
MAX_BLOCK = max(ALL_BLOCKS)
HIDDEN_SIZE = 3072
FLOW_TIMESTEP = 499


class StopAfterSelectedBlocks(Exception):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_safetensors(
    path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    safetensors.torch.save_file(
        {name: value.detach().contiguous().cpu() for name, value in tensors.items()},
        temporary,
        metadata=metadata,
    )
    os.replace(temporary, path)


@torch.inference_mode()
def selected_features(
    backend: QwenSharedBackend, image: torch.Tensor
) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for block_number in ALL_BLOCKS:
        index = block_number - 1

        def hook(_module, _inputs, output, block_number=block_number):
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError(f"Unexpected Qwen block {block_number} output")
            captured[block_number] = output[1].detach().to(
                device="cpu", dtype=torch.bfloat16
            )
            if block_number == MAX_BLOCK:
                raise StopAfterSelectedBlocks

        handles.append(
            backend.transformer.transformer_blocks[index].register_forward_hook(hook)
        )
    backend.transformer.disable_lora()
    try:
        latent = deterministic_encode(backend, image)
        try:
            backend.upstream.flow_step(
                latent, backend.transformer, backend.vae, backend.embeddings
            )
        except StopAfterSelectedBlocks:
            pass
        else:
            raise RuntimeError("Teacher forward did not stop at the configured late block")
    finally:
        backend.transformer.enable_lora()
        for handle in handles:
            handle.remove()
    if set(captured) != set(ALL_BLOCKS):
        raise RuntimeError(f"Missing selected blocks: {set(ALL_BLOCKS)-set(captured)}")
    return captured


def cosine_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return 1.0 - (
        F.normalize(first.float(), dim=-1, eps=1e-6)
        * F.normalize(second.float(), dim=-1, eps=1e-6)
    ).sum(-1)


def robust_unit(values: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    flat = values.float().reshape(-1)
    low = float(torch.quantile(flat, 0.02))
    high = float(torch.quantile(flat, 0.98))
    normalized = ((values.float() - low) / max(high - low, 1e-8)).clamp(0, 1)
    return normalized, low, high


def build_gate(
    input_features: dict[int, torch.Tensor],
    gt_features: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    normalized = []
    statistics = {}
    for block in LATE_BLOCKS:
        raw = cosine_distance(input_features[block], gt_features[block])[0]
        unit, low, high = robust_unit(raw)
        normalized.append(unit)
        statistics[str(block)] = {
            "raw_mean": float(raw.mean()),
            "raw_p02": low,
            "raw_p98": high,
        }
    stacked = torch.stack(normalized)
    mean = stacked.mean(0)
    agreement = (1.0 - 2.0 * stacked.std(0, unbiased=False)).clamp(0, 1)
    gate = (mean * (0.75 + 0.25 * agreement)).clamp(0, 1)
    return gate.to(torch.float16), agreement.to(torch.float16), statistics


def valid_record(output: Path, sample_id: str) -> bool:
    data_path = output / "samples" / f"{sample_id}.safetensors"
    record_path = output / "records" / f"{sample_id}.json"
    if not data_path.is_file() or not record_path.is_file():
        return False
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        return (
            record.get("id") == sample_id
            and record.get("cache_sha256") == sha256(data_path)
        )
    except Exception:
        return False


def extract(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    manifest, records = dataset_state(data_root)
    train_ids = (data_root / "splits/train.txt").read_text(encoding="utf-8").split()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    selected = train_ids[args.shard_index :: args.num_shards]
    remaining = [sample_id for sample_id in selected if not valid_record(output, sample_id)]
    print(
        json.dumps(
            {
                "status": "resume" if len(remaining) != len(selected) else "start",
                "shard": args.shard_index,
                "num_shards": args.num_shards,
                "selected": len(selected),
                "remaining": len(remaining),
            }
        ),
        flush=True,
    )
    if not remaining:
        return

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(2026 + args.shard_index)
    torch.cuda.manual_seed(2026 + args.shard_index)
    backend = QwenSharedBackend.from_local(device)
    backend.set_trainable_branch(None)
    backend.transformer.eval()
    backend.vae.eval()
    if any(parameter.requires_grad for parameter in backend.transformer.parameters()):
        raise RuntimeError("M4 cache extraction requires a frozen Qwen transformer")

    records_by_id = {record["id"]: record for record in records}
    try:
        for position, sample_id in enumerate(remaining, start=1):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            source = records_by_id[sample_id]
            width, height = source["target_size"]
            tokens = (height // 16) * (width // 16)
            paths = sample_paths(data_root, sample_id)
            input_features = selected_features(
                backend, image_tensor(paths["input"])[None]
            )
            torch.cuda.empty_cache()
            gt_features = selected_features(
                backend, image_tensor(paths["gt"])[None]
            )
            torch.cuda.synchronize(device)

            expected = (1, tokens, HIDDEN_SIZE)
            for block in ALL_BLOCKS:
                if tuple(input_features[block].shape) != expected:
                    raise ValueError(
                        f"Unexpected input block {block} shape for {sample_id}: "
                        f"{tuple(input_features[block].shape)}"
                    )
                if tuple(gt_features[block].shape) != expected:
                    raise ValueError(
                        f"Unexpected GT block {block} shape for {sample_id}: "
                        f"{tuple(gt_features[block].shape)}"
                    )

            gate, agreement, late_stats = build_gate(input_features, gt_features)
            tensors: dict[str, torch.Tensor] = {
                "late_gate": gate,
                "late_agreement": agreement,
                "token_grid_hw": torch.tensor(
                    [height // 16, width // 16], dtype=torch.int16
                ),
            }
            for block in EARLY_BLOCKS:
                tensors[f"q{block}_input"] = input_features[block][0]
                tensors[f"q{block}_gt"] = gt_features[block][0]
            for block in MID_BLOCKS:
                tensors[f"q{block}_gt"] = gt_features[block][0]

            if not all(torch.isfinite(value.float()).all() for value in tensors.values()):
                raise RuntimeError(f"Non-finite M4 cache tensor for {sample_id}")
            data_path = output / "samples" / f"{sample_id}.safetensors"
            atomic_safetensors(
                data_path,
                tensors,
                {
                    "sample_id": sample_id,
                    "cache_version": CACHE_VERSION,
                    "early_blocks": ",".join(map(str, EARLY_BLOCKS)),
                    "mid_blocks": ",".join(map(str, MID_BLOCKS)),
                    "late_blocks": ",".join(map(str, LATE_BLOCKS)),
                    "flow_timestep": str(FLOW_TIMESTEP),
                },
            )
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            sidecar = {
                "id": sample_id,
                "shard": args.shard_index,
                "image_size_wh": [width, height],
                "token_grid_hw": [height // 16, width // 16],
                "cache": str(data_path.relative_to(output)),
                "cache_bytes": data_path.stat().st_size,
                "cache_sha256": sha256(data_path),
                "source_sha256": {
                    "input": sha256(paths["input"]),
                    "gt": sha256(paths["gt"]),
                },
                "late_statistics": late_stats,
                "gate": {
                    "min": float(gate.min()),
                    "mean": float(gate.mean()),
                    "max": float(gate.max()),
                    "agreement_mean": float(agreement.mean()),
                },
                "peak_allocated_gib": peak,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(output / "records" / f"{sample_id}.json", sidecar)
            print(
                json.dumps(
                    {
                        "sample": sample_id,
                        "progress": f"{position}/{len(remaining)}",
                        "shard": args.shard_index,
                        "cache_mib": round(data_path.stat().st_size / 2**20, 2),
                        "peak_allocated_gib": round(peak, 3),
                    }
                ),
                flush=True,
            )
            del input_features, gt_features, tensors, gate, agreement
    finally:
        del backend
        gc.collect()
        torch.cuda.empty_cache()


def finalize(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    dataset_manifest, _records = dataset_state(data_root)
    train_ids = (data_root / "splits/train.txt").read_text(encoding="utf-8").split()
    missing = [sample_id for sample_id in train_ids if not valid_record(output, sample_id)]
    if missing:
        raise RuntimeError(f"M4 cache is incomplete; missing/broken: {missing[:10]}")
    records = [
        json.loads((output / "records" / f"{sample_id}.json").read_text(encoding="utf-8"))
        for sample_id in train_ids
    ]
    total_bytes = sum(record["cache_bytes"] for record in records)
    result = {
        "complete": True,
        "cache_version": CACHE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_git_commit": subprocess.check_output(
            ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_dataset": {
            "version": dataset_manifest["version"],
            "manifest": str(data_root / "manifest.json"),
            "manifest_sha256": sha256(data_root / "manifest.json"),
            "split": "train",
            "sample_count": len(train_ids),
            "train_ids": train_ids,
        },
        "teacher": {
            "model": "Qwen/Qwen-Image-Edit-2509",
            "adapter": "all LoRA disabled",
            "vae": "posterior mode; deterministic",
            "flow_timestep": FLOW_TIMESTEP,
            "early_blocks": list(EARLY_BLOCKS),
            "mid_blocks": list(MID_BLOCKS),
            "late_blocks": list(LATE_BLOCKS),
            "hidden_size": HIDDEN_SIZE,
        },
        "late_gate": {
            "per_layer": "robust p02/p98 normalized cosine distance Q_l(I),Q_l(GT)",
            "ensemble": "mean(D52,D54,D56)*(0.75+0.25*agreement)",
            "agreement": "clip(1-2*std(D52,D54,D56),0,1)",
            "range": [0, 1],
        },
        "storage": {
            "total_bytes": total_bytes,
            "total_gib": total_bytes / 2**30,
            "dtype": "bfloat16 features; float16 gates",
        },
        "samples": records,
    }
    atomic_json(output / "manifest.json", result)
    print(
        json.dumps(
            {
                "status": "complete",
                "samples": len(records),
                "total_gib": round(total_bytes / 2**30, 3),
                "manifest": str(output / "manifest.json"),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("extract", "finalize"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "extract":
        extract(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
