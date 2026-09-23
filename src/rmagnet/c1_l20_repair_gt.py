"""Rebuild only C1-L20 Q20(GT) features as BF16 after the FP16 overflow bug."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import safetensors.torch
import torch
import torch.distributed as dist

from .c1_l20_prepare import BLOCK_INDEX, BLOCK_NUMBER, DEFAULT_OUTPUT, sha256
from .qwen_backend import QwenSharedBackend
from .qwen_layer_probe import FeatureCapture, features
from .stage1_train import DEFAULT_DATA, image_tensor, rank, setup_distributed, world_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--restart-staging", action="store_true")
    args = parser.parse_args()

    device = setup_distributed()
    is_main = rank() == 0
    manifest_path = args.cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_ids = list(manifest["train_ids"])
    if len(train_ids) != 50 or manifest["qwen_feature"]["block_zero_based_index"] != BLOCK_INDEX:
        raise RuntimeError("Unexpected C1-L20 manifest")

    staging = args.cache_root / "gt_features_bf16_staging"
    backup = args.cache_root / "gt_features_fp16_invalid_backup"
    if is_main:
        if staging.exists():
            if not args.restart_staging:
                raise FileExistsError(f"Staging directory already exists: {staging}")
            shutil.rmtree(staging)
        if backup.exists():
            raise FileExistsError(f"Backup directory already exists: {backup}")
        staging.mkdir(parents=True)
    dist.barrier()

    backend = QwenSharedBackend.from_local(device)
    backend.set_trainable_branch(None)
    backend.transformer.eval()
    backend.vae.eval()
    capture = FeatureCapture(backend.transformer, (BLOCK_INDEX,))
    local_records = []
    try:
        for sample_id in train_ids[rank() :: world_size()]:
            gt_path = args.data_root / "transmission_layer" / f"{sample_id}.png"
            q_gt = features(
                backend, capture, image_tensor(gt_path)[None], (BLOCK_INDEX,)
            )[BLOCK_INDEX][0]
            if q_gt.shape != (768, 3072) or not torch.isfinite(q_gt).all():
                raise ValueError(f"Invalid float32 Q20(GT) for {sample_id}: {q_gt.shape}")
            stored = q_gt.to(torch.bfloat16).contiguous()
            if not torch.isfinite(stored).all():
                raise ValueError(f"BF16 conversion produced non-finite values for {sample_id}")
            output = staging / f"{sample_id}.safetensors"
            safetensors.torch.save_file(
                {"q20_gt": stored}, output,
                metadata={"sample_id": sample_id, "block_one_based": str(BLOCK_NUMBER), "dtype": "bfloat16"},
            )
            local_records.append({
                "id": sample_id,
                "sha256": sha256(output),
                "dtype": str(stored.dtype),
                "shape": list(stored.shape),
                "min": float(stored.float().min()),
                "max": float(stored.float().max()),
                "abs_max": float(stored.float().abs().max()),
            })
            print(json.dumps({"rank": rank(), **local_records[-1]}), flush=True)
    finally:
        capture.close()

    gathered: list[list[dict[str, object]] | None] = [None for _ in range(world_size())]
    dist.all_gather_object(gathered, local_records)
    dist.barrier()
    if is_main:
        records = [record for group in gathered for record in (group or [])]
        records.sort(key=lambda item: int(item["id"]))
        if [record["id"] for record in records] != train_ids:
            raise RuntimeError("Repaired feature IDs differ from manifest train IDs")
        for record in records:
            path = staging / f"{record['id']}.safetensors"
            loaded = safetensors.torch.load_file(path)["q20_gt"]
            if loaded.dtype != torch.bfloat16 or loaded.shape != (768, 3072) or not torch.isfinite(loaded).all():
                raise RuntimeError(f"Final staging verification failed: {path}")
            if sha256(path) != record["sha256"]:
                raise RuntimeError(f"Staging hash changed: {path}")

        old = args.cache_root / "gt_features"
        os.replace(old, backup)
        os.replace(staging, old)
        try:
            by_id = {record["id"]: record for record in records}
            manifest["qwen_feature"]["stored_gt_dtype"] = "bfloat16"
            manifest["qwen_feature"]["storage_reason"] = "BF16 preserves Qwen activation range; FP16 overflowed"
            manifest["gt_feature_repair"] = {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "world_size": world_size(),
                "old_dtype": "float16",
                "new_dtype": "bfloat16",
                "records": records,
            }
            for sample in manifest["samples"]:
                repaired = by_id[sample["id"]]
                sample["cache"]["gt_feature_sha256"] = repaired["sha256"]
                sample["q20_feature_dtype"] = "bfloat16"
                sample["q20_feature_abs_max"] = repaired["abs_max"]
            temporary = manifest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, manifest_path)
        except Exception:
            shutil.rmtree(old)
            os.replace(backup, old)
            raise
        shutil.rmtree(backup)
        print(json.dumps({"status": "complete", "features": len(records), "dtype": "bfloat16"}, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
