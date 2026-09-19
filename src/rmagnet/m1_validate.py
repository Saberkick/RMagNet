"""Offline M1: compare Qwen backend with pinned WindowSeat and saved PNGs."""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader

from .qwen_backend import QwenSharedBackend, ROOT


INPUT = ROOT / "datasets/own3/processed_1024/blended"
REFERENCE = ROOT / "results/windowseat_own3_1024"


def _rng_state():
    return torch.get_rng_state(), torch.cuda.get_rng_state()


def _restore_rng(state):
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state(state[1])


def run(output_dir: Path, device: torch.device) -> dict:
    torch.manual_seed(2026)
    np.random.seed(2026)
    backend = QwenSharedBackend.from_local(device)
    backend.activate("transmission")
    ws = backend.upstream
    dataset = ws.TilingDataset(
        transform_graph=functools.partial(
            ws.data_transform, processing_resolution=backend.resolution
        ),
        input_folder=str(INPUT),
        gt_folder=str(INPUT),
        use_short_edge_tile=True,
        tiling_w=backend.resolution,
        tiling_h=backend.resolution,
        processing_resolution=backend.resolution,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The VAE samples latents. Compare both calls from identical RNG states.
    before_loader = _rng_state()
    first = next(iter(loader))
    before_tile = _rng_state()
    with torch.no_grad():
        original = ws.decode(
            ws.flow_step(
                ws.encode(first["input_norm"], backend.vae),
                backend.transformer,
                backend.vae,
                backend.embeddings,
            ),
            backend.vae,
        )
    _restore_rng(before_tile)
    with torch.no_grad():
        replacement = backend.forward_normalized(
            first["input_norm"], "transmission"
        )
    tile_max_abs = (original - replacement).abs().max().item()
    if tile_max_abs > 1e-5:
        raise RuntimeError(f"New backend differs from official first tile: {tile_max_abs}")

    # The new R LoRA is untrained; only check independent routing and T restore.
    _restore_rng(before_tile)
    with torch.no_grad():
        reflection = backend.forward_normalized(first["input_norm"], "reflection")
    _restore_rng(before_tile)
    with torch.no_grad():
        restored = backend.forward_normalized(first["input_norm"], "transmission")
    restored_max_abs = (replacement - restored).abs().max().item()
    branch_max_abs = (replacement - reflection).abs().max().item()
    if restored_max_abs > 1e-5 or branch_max_abs <= 1e-5:
        raise RuntimeError(
            f"Adapter switch failed: restored={restored_max_abs}, T-vs-R={branch_max_abs}"
        )
    trainable = {}
    for branch in ("transmission", "reflection"):
        backend.set_trainable_branch(branch)
        names = [name for name, p in backend.transformer.named_parameters() if p.requires_grad]
        suffix = ".default." if branch == "transmission" else ".reflection."
        if not names or any(".lora_" not in name or suffix not in name for name in names):
            raise RuntimeError(f"Wrong trainable parameter set for {branch}")
        trainable[branch] = backend.trainable_summary()["trainable"]
    backend.set_trainable_branch(None)

    # Restore the state before the DataLoader iterator to match the old runner.
    _restore_rng(before_loader)
    rows = []
    tiles: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            candidate = backend.forward_normalized(
                batch["input_norm"], "transmission"
            )[0].cpu()
            rect = [batch["tile_info"][i][0].item() for i in range(4)]
            tiles.append({"rect": rect, "pixel": candidate})
            if not batch["is_last_tile"][0].item():
                continue

            width = max(t["rect"][2] for t in tiles)
            height = max(t["rect"][3] for t in tiles)
            acc = torch.zeros(3, height, width, dtype=torch.float32)
            wsum = torch.zeros(height, width, dtype=torch.float32)
            for entry in tiles:
                x0, y0, x1, y1 = entry["rect"]
                patch = entry["pixel"].squeeze(0).float()
                h, w = patch.shape[-2:]
                if (h, w) != (y1 - y0, x1 - x0):
                    patch = ws._lanczos_resize_chw(patch, (y1 - y0, x1 - x0))
                    h, w = patch.shape[-2:]
                wx = 1 - (2 * torch.arange(w, dtype=torch.float32) / max(w - 1, 1) - 1).abs()
                wy = 1 - (2 * torch.arange(h, dtype=torch.float32) / max(h - 1, 1) - 1).abs()
                window = (wy[:, None] * wx[None, :]).clamp_min(1e-3)
                acc[:, y0:y1, x0:x1] += patch * window
                wsum[y0:y1, x0:x1] += window
            stitched = acc / wsum.clamp_min(1e-6)
            original_h = batch["meta"]["orig_res"][0][0].item()
            original_w = batch["meta"]["orig_res"][1][0].item()
            x01 = ((stitched + 1) / 2).clamp(0, 1)
            pil = torchvision.transforms.functional.to_pil_image(x01.cpu())
            resized = pil.resize((original_w, original_h), resample=Image.LANCZOS)
            prediction = torchvision.transforms.functional.to_tensor(resized).numpy()
            image_path = batch["line"][0][0]
            name = image_path.split("/")[-1][:-4]
            input_hwc = np.transpose(ws.read_rgb_file(image_path), (1, 2, 0)).astype(np.float32) / 255
            pred_hwc = np.transpose(prediction, (1, 2, 0))
            ws.visualize(
                file_prefix=name,
                input_hwc=input_hwc,
                pred_hwc=pred_hwc,
                output_dir=str(output_dir),
                save_comparison=False,
                save_alternating=False,
            )
            saved = output_dir / f"{name}_windowseat_output.png"
            prior = REFERENCE / saved.name
            with Image.open(saved) as image, Image.open(prior) as reference:
                result = np.asarray(image, dtype=np.int16)
                expected = np.asarray(reference, dtype=np.int16)
            difference = np.abs(result - expected)
            rows.append(
                {
                    "image": name,
                    "tiles": len(tiles),
                    "shape": list(result.shape),
                    "equal_saved_png": bool(np.array_equal(result, expected)),
                    "changed_values": int(np.count_nonzero(difference)),
                    "max_abs_8bit": int(difference.max()),
                    "mean_abs_8bit": float(difference.mean()),
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
            tiles.clear()
    report = {
        "upstream_first_tile_max_abs": tile_max_abs,
        "T_after_R_max_abs": restored_max_abs,
        "T_vs_untrained_R_max_abs": branch_max_abs,
        "trainable_parameters_by_branch": trainable,
        "processing_resolution": backend.resolution,
        "adapter_names": {"T": "default", "R": "reflection"},
        "images": rows,
    }
    (output_dir / "m1_parity.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/m1_parity"))
    args = parser.parse_args()
    report = run(args.output, torch.device("cuda"))
    if any(not row["equal_saved_png"] for row in report["images"]):
        raise SystemExit("Saved PNG parity differs; inspect m1_parity.json")


if __name__ == "__main__":
    main()
