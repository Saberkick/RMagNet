"""Prepare and compare the M1b 30%-gradient semantic-strength control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PROJECT = Path("/share/linmingheng-local/xuke/RMagNet")
RUNS = PROJECT / "runs/m1b"
SOURCE = RUNS / "sem_calibration.json"
DERIVED = RUNS / "sem_calibration_f30.json"
OLD_GROUP = RUNS / "probe_e2"
NEW_GROUP = RUNS / "strength30_e2"
ARMS = ("base", "dolp", "shuffle")


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derived_calibration() -> dict:
    source = read_json(SOURCE)
    expected = {
        "sample_id": "13",
        "seed": 2026,
        "dino_model": "facebook/dinov2-small",
        "dino_layer": 6,
        "mask_threshold_uint8": 64,
    }
    for key, value in expected.items():
        if source.get(key) != value:
            raise RuntimeError(f"Original calibration {key} differs: {source.get(key)} != {value}")
    if not math.isclose(source["target_gradient_fraction"], 0.1, abs_tol=1e-12):
        raise RuntimeError("Source calibration is not the 10% experiment")
    if source["sem_weight"] <= 0:
        raise RuntimeError("Source semantic weight must be positive")
    return {
        **source,
        "sem_weight": 3.0 * source["sem_weight"],
        "target_gradient_fraction": 0.3,
        "derivation": "exactly 3 times the verified 10% coefficient; same sample and initial state",
        "source_calibration": str(SOURCE),
        "source_calibration_sha256": digest(SOURCE),
    }


def prepare() -> dict:
    expected = derived_calibration()
    if DERIVED.is_file():
        existing = read_json(DERIVED)
        if existing != expected:
            raise RuntimeError(f"Existing 30% calibration differs: {DERIVED}")
    else:
        DERIVED.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return expected


def compare() -> Path:
    calibration = prepare()
    source = read_json(SOURCE)
    configs = {}
    for group_name, folder, semantic_weight in (
        ("10%", OLD_GROUP, source["sem_weight"]),
        ("30%", NEW_GROUP, calibration["sem_weight"]),
    ):
        for arm in ARMS:
            config = read_json(folder / arm / "run_config.json")
            done = read_json(folder / arm / "DONE.json")
            if done["steps"] != 100 or done["epochs_completed"] != 2:
                raise RuntimeError(f"Incomplete {group_name}/{arm}: {done['steps']} steps")
            expected_weight = 0.0 if arm == "base" else semantic_weight
            actual_weight = config["args"]["sem_weight"]
            if not math.isclose(actual_weight, expected_weight, rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError(f"Wrong semantic coefficient in {group_name}/{arm}: {actual_weight}")
            configs[(group_name, arm)] = config
    reference = configs[("10%", "base")]
    for (group_name, arm), config in configs.items():
        for key in ("initial_sha256", "train_ids", "val_ids", "mask_threshold_uint8"):
            if config[key] != reference[key]:
                raise RuntimeError(f"Unmatched {key} in {group_name}/{arm}")
        for key in ("seed", "epochs", "max_steps", "learning_rate", "weight_decay", "warmup_steps"):
            if config["args"][key] != reference["args"][key]:
                raise RuntimeError(f"Unmatched {key} in {group_name}/{arm}")
    evaluations = {
        "10%": read_json(OLD_GROUP / "evaluation/evaluation.json"),
        "30%": read_json(NEW_GROUP / "evaluation/evaluation.json"),
    }
    for name, evaluation in evaluations.items():
        if evaluation["ids"] != ["11", "12", "17"] or evaluation["metric_domain"] != "saved 8-bit RGB PNG":
            raise RuntimeError(f"Unexpected evaluation settings in {name}")
    columns = ("psnr", "ssim", "masked_psnr", "masked_l1", "outside_l1")
    lines = [
        "# M1b semantic strength: 10% versus 30%",
        "",
        "Both groups used the same Stage 2 initial LoRA, 50 training images, seed 2026,",
        "2 epochs and 100 optimizer steps. The only intended change is the semantic",
        "coefficient, tripled from the original single-batch 10% gradient calibration.",
        "Values below come from saved 8-bit PNGs at 512×384 on IDs 11/12/17.",
        "",
        "| Group | Arm | PSNR ↑ | SSIM ↑ | Mask PSNR ↑ | Mask L1 ↓ | Outside L1 ↓ |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group_name, evaluation in evaluations.items():
        for arm in ARMS:
            values = evaluation["means"][arm]
            lines.append(
                f"| {group_name} | {arm} | "
                + " | ".join(f"{values[key]:.6f}" for key in columns)
                + " |"
            )
    lines.extend([
        "",
        "## Matched differences (B − A, B − C)",
        "",
        "Positive PSNR/SSIM and negative L1 mean B improved over its comparator.",
        "",
        "| Group | Contrast | ΔPSNR | ΔSSIM | ΔMask PSNR | ΔMask L1 | ΔOutside L1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for group_name, evaluation in evaluations.items():
        for other in ("base", "shuffle"):
            diff = {
                key: evaluation["means"]["dolp"][key] - evaluation["means"][other][key]
                for key in columns
            }
            lines.append(
                f"| {group_name} | dolp − {other} | "
                + " | ".join(f"{diff[key]:+.6f}" for key in columns)
                + " |"
            )
    lines.extend([
        "",
        "Inspect the three saved image triplets before interpreting small metric changes.",
        "A stronger DINO gradient does not itself establish semantic or text fidelity.",
        "",
    ])
    output = NEW_GROUP / "STRENGTH_COMPARISON.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "weight", "compare"))
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(), indent=2))
    elif args.action == "weight":
        print(prepare()["sem_weight"])
    else:
        print(compare())


if __name__ == "__main__":
    main()
