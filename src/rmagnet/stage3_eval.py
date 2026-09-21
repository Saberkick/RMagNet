"""Evaluate Stage 3 fusion checkpoints and R-input ablations on saved PNGs."""

from __future__ import annotations

import argparse, csv, hashlib, json
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .conditioning import LatentFusionMixer
from .qwen_backend import QwenSharedBackend
from .stage1_train import DEFAULT_DATA, discover_ids, image_tensor, ssim

ROOT = Path("/share/linmingheng-local/xuke/RMagNet")
DEFAULT_RUN = ROOT / "runs/stage3_fusion_r8"
DEFAULT_CACHE = ROOT / "cache/stage3_candidates"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().clamp(0, 1).mul(255).round().div(255)


def to_image(tensor: torch.Tensor) -> Image.Image:
    array = (quantize(tensor.detach().cpu())[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array, "RGB")


def metric_values(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction, target = quantize(prediction), quantize(target)
    mse = F.mse_loss(prediction, target)
    return {"l1": float(F.l1_loss(prediction, target)), "psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))), "ssim": float(ssim(prediction, target))}


def error_image(prediction: torch.Tensor, target: torch.Tensor) -> Image.Image:
    error = (quantize(prediction) - quantize(target)).abs().mean(1, keepdim=True)
    error = (error * 4).clamp(0, 1)
    return to_image(torch.cat([error, error.sqrt() * 0.55, torch.zeros_like(error)], 1))


def make_panel(items: list[tuple[str, Image.Image]]) -> Image.Image:
    width, height = items[0][1].size
    canvas = Image.new("RGB", (width * len(items), height + 28), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(items):
        canvas.paste(image, (index * width, 28)); draw.text((index * width + 7, 7), label, fill="black")
    return canvas


def resolve_checkpoint(run_dir: Path, variant: str) -> tuple[Path, Path]:
    if variant == "best":
        fuse, mixer = run_dir / "best_fusion_lora.safetensors", run_dir / "best_latent_mixer.safetensors"
    elif variant == "final":
        pointer = run_dir / "last_checkpoint.txt"
        if not pointer.is_file(): raise FileNotFoundError(f"Missing {pointer}")
        checkpoint = Path(pointer.read_text().strip())
        fuse, mixer = checkpoint / "fusion_lora.safetensors", checkpoint / "latent_mixer.safetensors"
    else:
        raise ValueError(f"Unknown variant: {variant}")
    if not fuse.is_file() or not mixer.is_file(): raise FileNotFoundError(f"Missing Stage 3 weights: {fuse}, {mixer}")
    return fuse, mixer


@torch.no_grad()
def predict(backend, mixer, ids, data_root, cache_root, device, seed, ablation):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    latents, outputs = {}, {}
    for sample_id in ids:
        tensors = {"i": image_tensor(data_root / "blended" / f"{sample_id}.png"), "t": image_tensor(cache_root / "transmission" / f"{sample_id}.png"), "r": image_tensor(cache_root / "reflection" / f"{sample_id}.png")}
        latents[sample_id] = {key: backend.encode_frozen(value.unsqueeze(0).to(device)) for key, value in tensors.items()}
    shuffled = ids[1:] + ids[:1]
    for index, sample_id in enumerate(ids):
        z = latents[sample_id]
        z_r = z["r"]
        if ablation == "r_zero": z_r = torch.zeros_like(z_r)
        elif ablation == "r_shuffle": z_r = latents[shuffled[index]]["r"]
        elif ablation != "normal": raise ValueError(ablation)
        result = backend.forward_fusion_latent(mixer(z["i"], z["t"], z_r))
        outputs[sample_id] = ((result.float() + 1) * 0.5).clamp(0, 1).cpu()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ids", default="11,12,17")
    parser.add_argument("--variants", default="best,final")
    parser.add_argument("--ablations", default="normal,r_zero,r_shuffle")
    parser.add_argument("--fusion-rank", type=int, default=8)
    parser.add_argument("--mixer-width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    all_ids = discover_ids(args.data_root)
    ids = all_ids if args.ids == "all" else [x.strip() for x in args.ids.split(",") if x.strip()]
    unknown = sorted(set(ids) - set(all_ids), key=int)
    if unknown: raise ValueError(f"Unknown IDs: {unknown}")
    variants = list(dict.fromkeys(x.strip() for x in args.variants.split(",") if x.strip()))
    ablations = list(dict.fromkeys(x.strip() for x in args.ablations.split(",") if x.strip()))
    if "r_shuffle" in ablations and len(ids) < 2:
        ablations.remove("r_shuffle")
        print("skip r_shuffle: requires at least two samples", flush=True)
    output = args.output_dir or args.run_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    backend = QwenSharedBackend.from_local(device, fusion_rank=args.fusion_rank)
    backend.set_trainable_fusion(); backend.transformer.eval(); backend.vae.eval()
    mixer = LatentFusionMixer(backend.vae.config.z_dim, args.mixer_width).to(device).eval()
    rows, records, predictions = [], {}, {}
    for variant in variants:
        fuse_path, mixer_path = resolve_checkpoint(args.run_dir, variant)
        _, unexpected = backend.transformer.load_state_dict(safetensors.torch.load_file(fuse_path, device=str(device)), strict=False)
        if unexpected: raise RuntimeError(f"Unexpected Fuse keys: {unexpected[:5]}")
        mixer.load_state_dict(safetensors.torch.load_file(mixer_path, device=str(device)))
        records[variant] = {"fusion": str(fuse_path), "fusion_sha256": sha256(fuse_path), "mixer": str(mixer_path), "mixer_sha256": sha256(mixer_path)}
        for ablation in ablations:
            name = variant if ablation == "normal" else f"{variant}_{ablation}"
            predictions[name] = predict(backend, mixer, ids, args.data_root, args.cache_root, device, args.seed, ablation)
    for sample_id in ids:
        source = ((image_tensor(args.data_root / "blended" / f"{sample_id}.png").unsqueeze(0) + 1) * 0.5)
        target = ((image_tensor(args.data_root / "transmission_layer" / f"{sample_id}.png").unsqueeze(0) + 1) * 0.5)
        stage2 = ((image_tensor(args.cache_root / "transmission" / f"{sample_id}.png").unsqueeze(0) + 1) * 0.5)
        for name, value in (("input", source), ("stage2_t", stage2)):
            rows.append({"id": sample_id, "variant": name, **metric_values(value, target)})
        panels = [("Input", to_image(source)), ("GT", to_image(target)), ("Stage2 T", to_image(stage2))]
        for name, mapping in predictions.items():
            pred = mapping[sample_id]; rows.append({"id": sample_id, "variant": name, **metric_values(pred, target)})
            image = to_image(pred); image.save(output / f"{sample_id}_{name}.png")
            error_image(pred, target).save(output / f"{sample_id}_{name}_error_x4.png")
            if name in variants: panels.append((name, image))
        make_panel(panels).save(output / f"{sample_id}_panel.png")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    names = list(dict.fromkeys(str(row["variant"]) for row in rows))
    means = {name: {key: sum(float(row[key]) for row in rows if row["variant"] == name) / len(ids) for key in ("l1", "psnr", "ssim")} for name in names}
    report = {"ids": ids, "metric_domain": "saved 8-bit RGB PNG", "seed": args.seed, "weights": records, "means": means}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Stage 3 evaluation", "", f"Samples: {', '.join(ids)}", "", "Metrics use saved 8-bit RGB PNG values.", "", "| Variant | L1 ↓ | PSNR ↑ | SSIM ↑ |", "|---|---:|---:|---:|"]
    lines += [f"| {name} | {value['l1']:.6f} | {value['psnr']:.3f} | {value['ssim']:.4f} |" for name, value in means.items()]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
