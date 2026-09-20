"""Cache frozen Stage 1/2 candidates for memory-efficient Stage 3 training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import safetensors.torch
import torch
from PIL import Image

from .qwen_backend import QwenSharedBackend
from .stage1_train import DEFAULT_DATA, discover_ids, image_tensor


ROOT = Path("/share/linmingheng-local/xuke/RMagNet")
DEFAULT_CACHE = ROOT / "cache/stage3_candidates"
DEFAULT_T = ROOT / "runs/stage2_transmission_r128/best_transmission_lora.safetensors"
DEFAULT_R = ROOT / "runs/stage1_reflection_r8/best_reflection_lora.safetensors"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_png(tensor: torch.Tensor, path: Path) -> None:
    array = ((tensor[0].float().cpu().clamp(-1, 1) + 1) * 127.5).round()
    array = array.byte().permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, "RGB").save(path, compress_level=1)


def load_adapter(backend: QwenSharedBackend, path: Path) -> None:
    weights = safetensors.torch.load_file(path, device=str(next(backend.parameters()).device))
    _, unexpected = backend.transformer.load_state_dict(weights, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected adapter keys in {path}: {unexpected[:5]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--transmission-lora", type=Path, default=DEFAULT_T)
    parser.add_argument("--reflection-lora", type=Path, default=DEFAULT_R)
    parser.add_argument("--ids", default="all")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reflection-rank", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    all_ids = discover_ids(args.data_root)
    requested = all_ids if args.ids == "all" else [x.strip() for x in args.ids.split(",") if x.strip()]
    unknown = sorted(set(requested) - set(all_ids), key=int)
    if unknown:
        raise ValueError(f"Unknown IDs: {unknown}")

    backend = QwenSharedBackend.from_local(device, reflection_rank=args.reflection_rank)
    load_adapter(backend, args.transmission_lora)
    load_adapter(backend, args.reflection_lora)
    backend.set_trainable_branch(None)
    backend.transformer.eval()
    backend.vae.eval()

    for sample_id in requested:
        outputs = {
            "transmission": args.cache_root / "transmission" / f"{sample_id}.png",
            "reflection": args.cache_root / "reflection" / f"{sample_id}.png",
        }
        if not args.overwrite and all(path.is_file() for path in outputs.values()):
            print(f"skip {sample_id}", flush=True)
            continue
        image = image_tensor(args.data_root / "blended" / f"{sample_id}.png").unsqueeze(0).to(device)
        numeric_id = int(sample_id)
        torch.manual_seed(args.seed + numeric_id * 2)
        torch.cuda.manual_seed(args.seed + numeric_id * 2)
        transmission = backend.forward_normalized(image, "transmission")
        torch.manual_seed(args.seed + numeric_id * 2 + 1)
        torch.cuda.manual_seed(args.seed + numeric_id * 2 + 1)
        reflection = backend.forward_normalized(image, "reflection")
        save_png(transmission, outputs["transmission"])
        save_png(reflection, outputs["reflection"])
        print(f"cached {sample_id}", flush=True)

    manifest = {
        "data_root": str(args.data_root),
        "ids": requested,
        "seed": args.seed,
        "transmission_lora": str(args.transmission_lora),
        "transmission_sha256": sha256(args.transmission_lora),
        "reflection_lora": str(args.reflection_lora),
        "reflection_sha256": sha256(args.reflection_lora),
        "format": "8-bit RGB PNG",
    }
    args.cache_root.mkdir(parents=True, exist_ok=True)
    (args.cache_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
