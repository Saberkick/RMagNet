"""Evaluate Stage 1 against identity and oracle affine color baselines."""

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
from .stage1_train import DEFAULT_DATA, image_tensor, ssim


DEFAULT_RUN = Path("/share/linmingheng-local/xuke/RMagNet/runs/stage1_reflection_r8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_adapter(backend: QwenSharedBackend, path: Path, device: torch.device) -> None:
    weights = safetensors.torch.load_file(path, device=str(device))
    _, unexpected = backend.transformer.load_state_dict(weights, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected adapter keys in {path}: {unexpected[:5]}")


def quantize01(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.clamp(0, 1).mul(255).round().div(255)


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = quantize01(prediction.float())
    target = quantize01(target.float())
    mse = F.mse_loss(prediction, target)
    return {
        "l1": float(F.l1_loss(prediction, target)),
        "psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))),
        "ssim": float(ssim(prediction, target)),
    }


def oracle_affine(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-image, per-channel a*x+b fit; diagnostic only because it sees GT."""
    result = torch.empty_like(image)
    for channel in range(3):
        x = image[:, channel].reshape(-1).float()
        y = target[:, channel].reshape(-1).float()
        design = torch.stack([x, torch.ones_like(x)], dim=1)
        solution = torch.linalg.lstsq(design, y[:, None]).solution[:, 0]
        result[:, channel] = image[:, channel] * solution[0] + solution[1]
    return result.clamp(0, 1)


def delta_cosine(prediction: torch.Tensor, image: torch.Tensor, target: torch.Tensor) -> float:
    predicted_delta = (prediction - image).float().flatten()
    target_delta = (target - image).float().flatten()
    return float(F.cosine_similarity(predicted_delta[None], target_delta[None]))


def to_image(tensor: torch.Tensor) -> Image.Image:
    array = quantize01(tensor.detach().cpu())[0].permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).round().astype(np.uint8), mode="RGB")


def error_image(prediction: torch.Tensor, target: torch.Tensor) -> Image.Image:
    error = (prediction.float() - target.float()).abs().mean(dim=1, keepdim=True)
    error = (error * 4).clamp(0, 1)
    heat = torch.cat([error, error.sqrt() * 0.55, torch.zeros_like(error)], dim=1)
    return to_image(heat)


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


@torch.no_grad()
def predict_variant(
    backend: QwenSharedBackend,
    weights: Path,
    sample_ids: list[str],
    data_root: Path,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    load_adapter(backend, weights, device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    outputs = {}
    for sample_id in sample_ids:
        image = image_tensor(data_root / "blended" / f"{sample_id}.png").unsqueeze(0).to(device)
        prediction = backend.forward_normalized(image, "reflection")
        outputs[sample_id] = ((prediction.float() + 1) * 0.5).clamp(0, 1).cpu()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ids", default="11,12,17")
    parser.add_argument("--reflection-rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_dir / "acceptance"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = [item for item in args.ids.split(",") if item]

    best_path = args.run_dir / "best_reflection_lora.safetensors"
    final_dir = Path((args.run_dir / "last_checkpoint.txt").read_text().strip())
    final_path = final_dir / "reflection_lora.safetensors"
    if not best_path.is_file() or not final_path.is_file():
        raise FileNotFoundError("Best or final Stage 1 adapter is missing")

    device = torch.device("cuda")
    backend = QwenSharedBackend.from_local(device, reflection_rank=args.reflection_rank)
    backend.set_trainable_branch("reflection")
    backend.transformer.eval()
    backend.vae.eval()
    best_predictions = predict_variant(
        backend, best_path, sample_ids, args.data_root, device, args.seed
    )
    final_predictions = predict_variant(
        backend, final_path, sample_ids, args.data_root, device, args.seed
    )

    rows: list[dict[str, object]] = []
    for sample_id in sample_ids:
        image = ((image_tensor(args.data_root / "blended" / f"{sample_id}.png").unsqueeze(0) + 1) * 0.5)
        target = ((image_tensor(args.data_root / "reflection_layer" / f"{sample_id}.png").unsqueeze(0) + 1) * 0.5)
        best = best_predictions[sample_id]
        final = final_predictions[sample_id]
        affine = oracle_affine(image, target)
        variants = {
            "identity": image,
            "oracle_affine": affine,
            "best": best,
            "final": final,
        }
        for name, prediction in variants.items():
            row = {"id": sample_id, "variant": name, **metrics(prediction, target)}
            row["delta_l1_from_input"] = float(F.l1_loss(quantize01(prediction), quantize01(image)))
            row["delta_cosine_to_target_change"] = delta_cosine(
                quantize01(prediction), quantize01(image), quantize01(target)
            )
            rows.append(row)

        input_image = to_image(image)
        target_image = to_image(target)
        best_image = to_image(best)
        final_image = to_image(final)
        heat = error_image(best, target)
        best_image.save(output_dir / f"{sample_id}_best.png")
        final_image.save(output_dir / f"{sample_id}_final.png")
        heat.save(output_dir / f"{sample_id}_best_error_x4.png")
        panel(
            [
                ("I: 45 deg", input_image),
                ("Target R: 90 deg", target_image),
                ("Best R prediction", best_image),
                ("Final R prediction", final_image),
                ("Best abs error x4", heat),
            ]
        ).save(output_dir / f"{sample_id}_panel.png")

    fieldnames = list(rows[0])
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    variants = sorted({str(row["variant"]) for row in rows})
    means = {
        variant: {
            key: sum(float(row[key]) for row in rows if row["variant"] == variant)
            / len(sample_ids)
            for key in ("l1", "psnr", "ssim", "delta_l1_from_input", "delta_cosine_to_target_change")
        }
        for variant in variants
    }
    report = {
        "ids": sample_ids,
        "metric_domain": "saved 8-bit RGB PNG values",
        "best_adapter": {"path": str(best_path), "sha256": sha256(best_path)},
        "final_adapter": {"path": str(final_path), "sha256": sha256(final_path)},
        "means": means,
        "best_minus_identity": {
            "l1": means["best"]["l1"] - means["identity"]["l1"],
            "psnr": means["best"]["psnr"] - means["identity"]["psnr"],
            "ssim": means["best"]["ssim"] - means["identity"]["ssim"],
        },
        "best_minus_oracle_affine": {
            "l1": means["best"]["l1"] - means["oracle_affine"]["l1"],
            "psnr": means["best"]["psnr"] - means["oracle_affine"]["psnr"],
            "ssim": means["best"]["ssim"] - means["oracle_affine"]["ssim"],
        },
    }
    (output_dir / "acceptance.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
