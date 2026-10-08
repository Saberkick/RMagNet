"""Create an immutable training-only GT noise overlay and reuse clean caches.

Original data and cache files are never rewritten. A fixed seven recipients
receive a different capture group's training GT at exactly the same resolution.
"""
import argparse
import copy
import json
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .m4_cache import sha256, atomic_json


def audit_overlay(root, manifest, records, splits):
    noise = manifest.get("label_noise")
    if noise is None:
        return
    if noise.get("type") != "fixed-training-GT-replacement-v1":
        raise RuntimeError("Unsupported intentional label noise")
    base = Path(noise["source_data_root"]).resolve()
    if base == root.resolve() or sha256(base / "manifest.json") != noise["source_manifest_sha256"]:
        raise RuntimeError("Original dataset identity changed")
    original = json.loads((base / "manifest.json").read_text())
    if original.get("label_noise"):
        raise RuntimeError("Noise overlays must not be stacked")
    original_records = {r["id"]: r for r in original["samples"]}
    mapping = noise["mapping"]
    if len(mapping) != 7 or noise["count"] != 7 or noise["seed"] != 20261008:
        raise RuntimeError("This experiment requires the recorded fixed seven labels")
    for split, ids in splits.items():
        if ids != (base / "splits" / f"{split}.txt").read_text().split():
            raise RuntimeError("Noise experiment changed a split")
    train = set(splits["train"])
    for sid, donor in mapping.items():
        if sid not in train or donor not in train or sid == donor:
            raise RuntimeError("Noise mapping leaked a split or retained original GT")
        a, b = original_records[sid], original_records[donor]
        if a["group"] == b["group"] or a["target_size"] != b["target_size"]:
            raise RuntimeError("GT donor must be a different capture at identical size")
        if a["processed"]["gt"]["sha256"] == b["processed"]["gt"]["sha256"]:
            raise RuntimeError("GT donor is identical")
    for sid, rec in records.items():
        original_rec = original_records[sid]
        for role in ("input", "reflection_90", "dolp", "gt"):
            source = original_records[mapping.get(sid, sid)] if role == "gt" else original_rec
            if rec["processed"][role]["sha256"] != source["processed"][role]["sha256"]:
                raise RuntimeError(f"Undeclared noise or changed observation: {sid}/{role}")
        if sha256(root / "transmission_layer" / f"{sid}.png") != rec["processed"]["gt"]["sha256"]:
            raise RuntimeError(f"Overlay target file changed: {sid}")
    if noise["validation_changed"] or noise["test_changed"]:
        raise RuntimeError("Only training targets may change")


def prepare(args):
    from .sma_data import load_manifest
    source = args.source.resolve()
    original, records, splits = load_manifest(source)
    if original.get("label_noise"):
        raise RuntimeError("Use the original corrected dataset")
    target, cache = args.output.resolve(), args.cache_output.resolve()
    personal = Path('/share/linmingheng-local/xuke').resolve()
    for path in (target, cache):
        if not path.is_relative_to(personal) or path in (personal, source, args.cache_source.resolve()):
            raise RuntimeError("Unsafe output location")
    if target.exists():
        current, _, _ = load_manifest(target)
        expected = current.get("label_noise", {})
        if expected.get("seed") != args.seed or expected.get("count") != args.count:
            raise RuntimeError("Existing overlay belongs to another experiment")
        mapping = expected["mapping"]
    else:
        sizes = defaultdict(list)
        for sid in splits["train"]:
            sizes[tuple(records[sid]["target_size"])].append(sid)
        choices = {}
        for sid in splits["train"]:
            choices[sid] = [donor for donor in sizes[tuple(records[sid]["target_size"])]
                            if records[donor]["group"] != records[sid]["group"]
                            and records[donor]["processed"]["gt"]["sha256"] != records[sid]["processed"]["gt"]["sha256"]]
        eligible = sorted(sid for sid in choices if choices[sid])
        rng = random.Random(args.seed)
        selected = rng.sample(eligible, args.count)
        mapping = {sid: rng.choice(sorted(choices[sid])) for sid in selected}
        current = copy.deepcopy(original)
        current["label_noise"] = {
            "type": "fixed-training-GT-replacement-v1", "count": args.count,
            "requested_fraction": .05, "actual_fraction": args.count / len(splits["train"]),
            "seed": args.seed, "mapping": mapping, "eligible_recipients": len(eligible),
            "selection": "uniform among training samples with a different-capture exact-size donor",
            "donors": "training split only; fixed for all four epochs; donor reuse permitted",
            "source_data_root": str(source), "source_manifest_sha256": sha256(source / "manifest.json"),
            "validation_changed": False, "test_changed": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        target.mkdir(parents=True)
        for folder in ("blended", "reflection_90", "dolp", "splits"):
            (target / folder).symlink_to(source / folder, target_is_directory=True)
        (target / "transmission_layer").mkdir()
        for rec in current["samples"]:
            sid = rec["id"]; donor = mapping.get(sid, sid)
            (target / "transmission_layer" / f"{sid}.png").symlink_to(source / "transmission_layer" / f"{donor}.png")
            if donor != sid:
                rec["original_correct_gt"] = copy.deepcopy(rec["processed"]["gt"])
                rec["source"]["gt"] = copy.deepcopy(records[donor]["source"]["gt"])
                rec["processed"]["gt"] = copy.deepcopy(records[donor]["processed"]["gt"])
                rec["processed"]["gt"]["path"] = f"transmission_layer/{sid}.png"
                rec["intentional_wrong_gt_donor"] = donor
        atomic_json(target / "manifest.json", current)
        atomic_json(target / "noise_mapping.json", current["label_noise"])
        load_manifest(target)

    original_cache = args.cache_source.resolve()
    cm = json.loads((original_cache / "manifest.json").read_text())
    if not cm["complete"] or cm["source_dataset"]["manifest_sha256"] != sha256(source / "manifest.json"):
        raise RuntimeError("Clean cache dataset mismatch")
    if cm["source_dataset"]["train_ids"] != splits["train"]:
        raise RuntimeError("Clean cache split mismatch")
    (cache / "samples").mkdir(parents=True, exist_ok=True)
    (cache / "records").mkdir(exist_ok=True)
    reused = 0
    for rec in cm["samples"]:
        sid = rec["id"]
        if sid in mapping:
            continue  # Re-extract every GT-dependent tensor and late gate.
        old, new = original_cache / rec["cache"], cache / rec["cache"]
        if sha256(old) != rec["cache_sha256"]:
            raise RuntimeError("Clean cache checksum mismatch")
        if not new.exists():
            os.link(old, new)
        elif not os.path.samefile(old, new):
            raise RuntimeError("Unchanged cache must share the immutable original file")
        sidecar = cache / "records" / f"{sid}.json"
        if not sidecar.exists():
            shutil.copy2(original_cache / "records" / f"{sid}.json", sidecar)
        reused += 1
    atomic_json(cache / "noise_setup.json", {"label_noise": current["label_noise"],
                "cache_reused_hardlinks": reused, "cache_rebuild_ids": sorted(mapping),
                "source_cache_manifest_sha256": sha256(original_cache / "manifest.json")})
    print(json.dumps({"status": "overlay_ready", "data_root": str(target), "cache_root": str(cache),
                      "changed": len(mapping), "reused": reused, "mapping": mapping}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache-source', type=Path, required=True)
    p.add_argument('--cache-output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261008)
    p.add_argument('--count', type=int, default=7)
    a = p.parse_args()
    if a.count != 7 or a.seed != 20261008:
        raise ValueError('This fixed 5% experiment uses count=7, seed=20261008')
    prepare(a)


if __name__ == '__main__':
    main()
