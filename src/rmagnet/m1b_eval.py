"""Compare saved 8-bit PNG outputs for Stage 2 and M1b arms on 11/12/17."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from PIL import Image

from .m1b_train import INITIAL, PROJECT, sha256
from .qwen_backend import QwenSharedBackend
from .stage1_train import DEFAULT_DATA, image_tensor, ssim
from .stage2_eval import to_image


def load_adapter(backend: QwenSharedBackend, path: Path, device: torch.device) -> None:
    state = safetensors.torch.load_file(path, device=str(device))
    _, unexpected = backend.transformer.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA keys: {unexpected[:5]}")


def png_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0) / 255


def measure(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    err = pred - gt
    mse = err.square().mean()
    m = mask.expand_as(pred)
    other = 1 - m
    m_mse = (err.square() * m).sum() / m.sum().clamp_min(1)
    return {"l1": float(err.abs().mean()),
            "psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))),
            "ssim": float(ssim(pred, gt)),
            "masked_l1": float((err.abs() * m).sum() / m.sum().clamp_min(1)),
            "masked_psnr": float(-10 * torch.log10(m_mse.clamp_min(1e-12))),
            "outside_l1": float((err.abs() * other).sum() / other.sum().clamp_min(1))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--initial", type=Path, default=INITIAL)
    parser.add_argument("--run-root", type=Path, default=PROJECT / "runs/m1b")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "runs/m1b/evaluation")
    parser.add_argument("--arms", default="base,dolp,shuffle")
    parser.add_argument("--ids", default="11,12,17")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    ids = [value for value in args.ids.split(",") if value]
    arms = [value for value in args.arms.split(",") if value]
    variants = {"stage2": args.initial}
    for arm in arms:
        if arm not in {"base", "dolp", "shuffle"}:
            raise ValueError(arm)
        variants[arm] = args.run_root / arm / "final_transmission_lora.safetensors"
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
            with Image.open(mask_path) as loaded:
                mask = torch.from_numpy((np.asarray(loaded.convert("L"), dtype=np.uint8) >= 128).astype(np.float32)).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                inp = image_tensor(inp_path).unsqueeze(0).to(device)
                output = backend.forward_normalized(inp, "transmission")
                output = ((output.float() + 1) * 0.5).clamp(0, 1)
            output_path = args.output_dir / f"{sample_id}_{variant}.png"
            to_image(output).save(output_path)
            rows.append({"id": sample_id, "variant": variant,
                         **measure(png_tensor(output_path), png_tensor(gt_path), mask)})
    for sample_id in ids:
        inp = args.data_root / "blended" / f"{sample_id}.png"
        gt = args.data_root / "transmission_layer" / f"{sample_id}.png"
        with Image.open(args.data_root / "dolp_mask" / f"{sample_id}.png") as loaded:
            mask = torch.from_numpy((np.asarray(loaded.convert("L"), dtype=np.uint8) >= 128).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        rows.append({"id": sample_id, "variant": "input", **measure(png_tensor(inp), png_tensor(gt), mask)})
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = {variant: {key: sum(float(row[key]) for row in rows if row["variant"] == variant) / len(ids)
                       for key in ("l1", "psnr", "ssim", "masked_l1", "masked_psnr", "outside_l1")}
             for variant in ("input", *variants)}
    report = {"ids": ids, "metric_domain": "saved 8-bit RGB PNG", "seed": args.seed,
              "weights": weights, "means": means, "rows": rows}
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = ("psnr", "ssim", "masked_psnr", "masked_l1", "outside_l1")
    lines = ["# M1b evaluation", "", "All values use saved 8-bit PNGs at 512×384. DoLP masks are evaluation weights only.", "",
             "| Variant | PSNR ↑ | SSIM ↑ | Mask PSNR ↑ | Mask L1 ↓ | Outside L1 ↓ |",
             "|---|---:|---:|---:|---:|---:|"]
    for variant, value in means.items():
        lines.append("| " + variant + " | " + " | ".join(f"{value[key]:.4f}" for key in columns) + " |")
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(args.output_dir), "means": means}, indent=2), flush=True)


if __name__ == "__main__":
    main()
