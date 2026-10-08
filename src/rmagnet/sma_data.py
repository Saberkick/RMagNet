"""Strict SMA data audit; honors the existing v3 misregistration exclusion."""
import json
from pathlib import Path
from PIL import Image
from .m4_cache import sha256


def load_manifest(root):
    manifest = json.loads((root/"manifest.json").read_text())
    versions = {"m2-variable-aspect-v2-corrected-labels": 18,
                "m2-variable-aspect-v3-excluded-misaligned": 17}
    version = manifest.get("version")
    if not manifest.get("complete") or version not in versions:
        raise RuntimeError("Unsupported or incomplete corrected dataset")
    if manifest.get("label_correction", {}).get("status") != "applied":
        raise RuntimeError("Label correction must be recorded")
    records = {r["id"]: r for r in manifest["samples"]}
    splits = {s: (root/"splits"/f"{s}.txt").read_text().split() for s in ("train", "validation", "test")}
    if list(map(len, splits.values())) != [144, 18, versions[version]]:
        raise RuntimeError("Unexpected split sizes")
    ids = [x for values in splits.values() for x in values]
    if len(ids) != len(set(ids)) or set(ids) != set(records):
        raise RuntimeError("Split union/identity mismatch")
    group_sets = [{records[x]["group"] for x in splits[s]} for s in splits]
    if any(group_sets[i]&group_sets[j] for i in range(3) for j in range(i)):
        raise RuntimeError("Capture groups overlap")
    excluded = {r["id"] for r in manifest.get("excluded_samples", [])}
    if excluded & set(ids): raise RuntimeError("Excluded scene returned to dataset")
    for name in ("train", "validation"):
        for sid in splits[name]:
            r = records[sid]
            w,h = r["target_size"]
            if w%16 or h%16: raise RuntimeError("Unsupported image grid")
            for role, folder in (("input","blended"),("gt","transmission_layer"),("reflection_90","reflection_90"),("dolp","dolp")):
                p = root/folder/f"{sid}.png"
                if sha256(p) != r["processed"][role]["sha256"]:
                    raise RuntimeError(f"Source hash mismatch: {sid}/{role}")
                with Image.open(p) as im:
                    if im.size != (w,h) or im.mode != ("L" if role=="dolp" else "RGB"):
                        raise RuntimeError(f"Source geometry/mode mismatch: {sid}/{role}")
    return manifest, records, splits


def dataset_state(root):
    manifest, records, splits = load_manifest(root)
    return manifest, [records[x] for x in splits["train"]]
