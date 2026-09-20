"""Evaluate Stage 2 transmission adapters with identical stochastic latents."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .qwen_backend import QwenSharedBackend
from .stage1_train import DEFAULT_DATA, discover_ids, image_tensor, ssim


DEFAULT_RUN = Path("/share/linmingheng-local/xuke/RMagNet/runs/stage2_transmission_r128")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantize01(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().clamp(0, 1).mul(255).round().div(255)


def to_image(tensor: torch.Tensor) -> Image.Image:
    array = quantize01(tensor.detach().cpu())[0].permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).round().astype(np.uint8), mode="RGB")


def error_image(prediction: torch.Tensor, target: torch.Tensor) -> Image.Image:
    error = (quantize01(prediction) - quantize01(target)).abs().mean(1, keepdim=True)
    error = (error * 4).clamp(0, 1)
    heat = torch.cat([error, error.sqrt() * 0.55, torch.zeros_like(error)], dim=1)
    return to_image(heat)


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = quantize01(prediction)
    target = quantize01(target)
    mse = F.mse_loss(prediction, target)
    return {
        "l1": float(F.l1_loss(prediction, target)),
        "psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))),
        "ssim": float(ssim(prediction, target)),
    }


def panel(images: list[tuple[str, Image.Image]]) -> Image.Image:
    width, height = images[0][1].size
    header = 28
    canvas = Image.new("RGB", (width * len(images), height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(images):
        x = index * width
        canvas.paste(image, (x, header))
        draw.text((x + 8, 7), label, fill="black")
    return canvas


def resolve_variants(run_dir: Path, requested: list[str]) -> dict[str, Path | None]:
    allowed = {"baseline", "best", "final"}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}; choose from {sorted(allowed)}")
    variants: dict[str, Path | None] = {}
    for name in requested:
        if name == "baseline":
            variants[name] = None
        elif name == "best":
            path = run_dir / "best_transmission_lora.safetensors"
            if not path.is_file():
                raise FileNotFoundError(f"Best adapter is missing: {path}")
            variants[name] = path
        else:
            pointer = run_dir / "last_checkpoint.txt"
            if not pointer.is_file():
                raise FileNotFoundError(f"Final checkpoint pointer is missing: {pointer}")
            path = Path(pointer.read_text(encoding="utf-8").strip()) / "transmission_lora.safetensors"
            if not path.is_file():
                raise FileNotFoundError(f"Final adapter is missing: {path}")
            variants[name] = path
    return variants


def load_adapter(backend: QwenSharedBackend, path: Path, device: torch.device) -> None:
    weights = safetensors.torch.load_file(path, device=str(device))
    _, unexpected = backend.transformer.load_state_dict(weights, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected adapter keys in {path}: {unexpected[:5]}")


@torch.no_grad()
def predict(
    backend: QwenSharedBackend,
    sample_ids: list[str],
    data_root: Path,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    # Reset once per variant so every adapter sees the same per-image VAE samples.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    outputs: dict[str, torch.Tensor] = {}
    for sample_id in sample_ids:
        image = image_tensor(data_root / "blended" / f"{sample_id}.png")
        prediction = backend.forward_normalized(image.unsqueeze(0).to(device), "transmission")
        outputs[sample_id] = ((prediction.float() + 1) * 0.5).clamp(0, 1).cpu()
    return outputs


def markdown_report(
    sample_ids: list[str],
    means: dict[str, dict[str, float]],
    deltas: dict[str, dict[str, float]],
    weights: dict[str, dict[str, str | None]],
) -> str:
    lines = [
        "# Stage 2 evaluation",
        "",
        f"Samples: {', '.join(sample_ids)}",
        "",
        "Metrics are computed from saved 8-bit RGB PNG values.",
        "",
        "| Variant | L1 ↓ | PSNR ↑ | SSIM ↑ |",
        "|---|---:|---:|---:|",
    ]
    for name, values in means.items():
        lines.append(
            f"| {name} | {values['l1']:.6f} | {values['psnr']:.3f} | {values['ssim']:.4f} |"
        )
    if deltas:
        lines.extend(
            [
                "",
                "## Improvement over official WindowSeat baseline",
                "",
                "| Variant | ΔL1 ↓ | ΔPSNR ↑ | ΔSSIM ↑ |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, values in deltas.items():
            lines.append(
                f"| {name} | {values['l1']:+.6f} | {values['psnr']:+.3f} | {values['ssim']:+.4f} |"
            )
    lines.extend(["", "## Weights", ""])
    for name, record in weights.items():
        lines.append(f"- **{name}**: `{record['path']}`; SHA-256 `{record['sha256']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ids", default="11,12,17", help="comma list or 'all'")
    parser.add_argument("--variants", default="baseline,best,final")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    all_ids = discover_ids(args.data_root)
    sample_ids = all_ids if args.ids == "all" else [item for item in args.ids.split(",") if item]
    unknown_ids = sorted(set(sample_ids) - set(all_ids), key=int)
    if unknown_ids:
        raise ValueError(f"Unknown sample IDs: {unknown_ids}")
    requested_raw = [item.strip() for item in args.variants.split(",") if item.strip()]
    if len(requested_raw) != len(set(requested_raw)):
        raise ValueError(f"Duplicate variants are not allowed: {requested_raw}")
    unknown_variants = sorted(set(requested_raw) - {"baseline", "best", "final"})
    if unknown_variants:
        raise ValueError(f"Unknown variants: {unknown_variants}")
    # The official adapter must be evaluated before a trained adapter replaces
    # its weights in memory. Keep a stable order for reports and comparisons.
    requested = [name for name in ("baseline", "best", "final") if name in requested_raw]
    variants = resolve_variants(args.run_dir, requested)
    output_dir = args.output_dir or args.run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    backend = QwenSharedBackend.from_local(device)
    backend.set_trainable_branch("transmission")
    backend.transformer.eval()
    backend.vae.eval()

    predictions: dict[str, dict[str, torch.Tensor]] = {}
    weight_records: dict[str, dict[str, str | None]] = {}
    for name, path in variants.items():
        if path is not None:
            load_adapter(backend, path, device)
            weight_records[name] = {"path": str(path), "sha256": sha256(path)}
        else:
            weight_records[name] = {
                "path": "official WindowSeat adapter loaded by QwenSharedBackend",
                "sha256": None,
            }
        predictions[name] = predict(
            backend, sample_ids, args.data_root, device, args.seed
        )

    rows: list[dict[str, object]] = []
    for sample_id in sample_ids:
        image = ((image_tensor(args.data_root / "blended" / f"{sample_id}.png").unsqueeze(0) + 1) * 0.5)
        target = ((image_tensor(args.data_root / "transmission_layer" / f"{sample_id}.png").unsqueeze(0) + 1) * 0.5)
        rows.append({"id": sample_id, "variant": "input", **metrics(image, target)})
        panel_items = [("Input I45", to_image(image)), ("GT T0", to_image(target))]
        for name in requested:
            prediction = predictions[name][sample_id]
            rows.append({"id": sample_id, "variant": name, **metrics(prediction, target)})
            output = to_image(prediction)
            output.save(output_dir / f"{sample_id}_{name}.png")
            error_image(prediction, target).save(output_dir / f"{sample_id}_{name}_error_x4.png")
            panel_items.append((name, output))
        panel_items.append((f"{requested[-1]} error x4", error_image(predictions[requested[-1]][sample_id], target)))
        panel(panel_items).save(output_dir / f"{sample_id}_panel.png")

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    names = ["input", *requested]
    means = {
        name: {
            key: sum(float(row[key]) for row in rows if row["variant"] == name)
            / len(sample_ids)
            for key in ("l1", "psnr", "ssim")
        }
        for name in names
    }
    baseline = means.get("baseline")
    deltas = {}
    if baseline is not None:
        for name in requested:
            if name == "baseline":
                continue
            deltas[name] = {
                "l1": means[name]["l1"] - baseline["l1"],
                "psnr": means[name]["psnr"] - baseline["psnr"],
                "ssim": means[name]["ssim"] - baseline["ssim"],
            }
    report = {
        "ids": sample_ids,
        "metric_domain": "saved 8-bit RGB PNG values",
        "seed": args.seed,
        "weights": weight_records,
        "means": means,
        "improvement_over_baseline": deltas,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(
        markdown_report(sample_ids, means, deltas, weight_records), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
