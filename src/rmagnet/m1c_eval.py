"""Evaluate saved 8-bit M1c outputs against GT on fixed 11/12/17 validation images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from .m1b_eval import load_adapter, measure, png_tensor
from .m1b_train import INITIAL, PROJECT, sha256
from .qwen_backend import QwenSharedBackend
from .stage1_train import DEFAULT_DATA, image_tensor
from .stage2_eval import to_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-root", type=Path, default=PROJECT / "runs/m1c/probe_e2")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "runs/m1c/probe_e2/evaluation")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    ids = ["11", "12", "17"]
    arms = ("base", "soft", "shift")
    variants = {"stage2": INITIAL, **{arm: args.run_root / arm / "final_transmission_lora.safetensors" for arm in arms}}
    configs = {}
    for arm in arms:
        done = json.loads((args.run_root / arm / "DONE.json").read_text())
        config = json.loads((args.run_root / arm / "run_config.json").read_text())
        if done["steps"] != 100 or done["epochs_completed"] != 2:
            raise RuntimeError(f"Incomplete M1c arm {arm}")
        configs[arm] = config
    reference = configs["base"]
    for arm, config in configs.items():
        for key in ("initial_sha256", "maps_manifest_sha256", "train_ids", "val_ids", "calibration_sha256"):
            if config[key] != reference[key]:
                raise RuntimeError(f"Mismatched {key}: {arm}")
        for key in ("epochs", "max_steps", "learning_rate", "weight_decay", "seed"):
            if config["args"][key] != reference["args"][key]:
                raise RuntimeError(f"Mismatched {key}: {arm}")
    for path in variants.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    backend = QwenSharedBackend.from_local(device)
    backend.set_trainable_branch("transmission")
    backend.transformer.eval()
    backend.vae.eval()
    rows = []
    weights = {}
    for variant, path in variants.items():
        load_adapter(backend, path, device)
        weights[variant] = {"path": str(path), "sha256": sha256(path)}
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        for sample_id in ids:
            inp_path = args.data_root / "blended" / f"{sample_id}.png"
            gt_path = args.data_root / "transmission_layer" / f"{sample_id}.png"
            mask_path = args.data_root / "dolp_mask" / f"{sample_id}.png"
            mask = png_tensor(mask_path)[:, :1].ge(.5).float()
            with torch.no_grad():
                output = backend.forward_normalized(image_tensor(inp_path)[None].to(device), "transmission")
                output = ((output.float() + 1) * .5).clamp(0, 1)
            output_path = args.output_dir / f"{sample_id}_{variant}.png"
            to_image(output).save(output_path)
            rows.append({"id": sample_id, "variant": variant,
                         **measure(png_tensor(output_path), png_tensor(gt_path), mask)})
    for sample_id in ids:
        inp = args.data_root / "blended" / f"{sample_id}.png"
        gt = args.data_root / "transmission_layer" / f"{sample_id}.png"
        mask = png_tensor(args.data_root / "dolp_mask" / f"{sample_id}.png")[:, :1].ge(.5).float()
        rows.append({"id": sample_id, "variant": "input", **measure(png_tensor(inp), png_tensor(gt), mask)})
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = ("l1", "psnr", "ssim", "masked_l1", "masked_psnr", "outside_l1")
    means = {variant: {key: sum(row[key] for row in rows if row["variant"] == variant) / len(ids) for key in metrics}
             for variant in ("input", *variants)}
    report = {"ids": ids, "metric_domain": "saved 8-bit RGB PNG at 512x384",
              "weights": weights, "means": means, "rows": rows}
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = ("psnr", "ssim", "masked_psnr", "masked_l1", "outside_l1")
    lines = ["# M1c evaluation", "", "All metrics use saved 8-bit PNGs at 512×384. The evaluation mask is the unchanged M1b hard DoLP mask; M1c training uses soft weights only.", "",
             "| Variant | PSNR ↑ | SSIM ↑ | Mask PSNR ↑ | Mask L1 ↓ | Outside L1 ↓ |",
             "|---|---:|---:|---:|---:|---:|"]
    for variant, values in means.items():
        lines.append("| " + variant + " | " + " | ".join(f"{values[key]:.6f}" for key in columns) + " |")
    lines.extend(["", "## Matched differences", "", "Positive PSNR/SSIM and negative L1 favor the true soft map.", "",
                  "| Contrast | ΔPSNR | ΔSSIM | ΔMask PSNR | ΔMask L1 | ΔOutside L1 |",
                  "|---|---:|---:|---:|---:|---:|"])
    for comparator in ("base", "shift"):
        lines.append("| soft − " + comparator + " | " + " | ".join(f"{means['soft'][key]-means[comparator][key]:+.6f}" for key in columns) + " |")
    lines.extend(["", "Three validation images and one seed give a directional result only. Inspect the saved images for text, fine texture, and new objects.", ""])
    (args.output_dir / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(args.output_dir), "means": means}, indent=2), flush=True)


if __name__ == "__main__":
    main()
