"""Prepare the M2 variable-aspect reflection-removal dataset.

The source is a ZIP with four aligned files per sample.  Images are resized
without cropping or padding to roughly the C1-L20 pixel budget.  Both axes are
quantized to multiples of 16 so Qwen's VAE and 2x2 latent packing remain valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, ImageStat


VERSION = "m2-variable-aspect-v1"
ROLE_SUFFIXES = {
    "input": "",
    "reflection_90": "_90",
    "dolp": "_DoLP",
    "gt": "_GT",
}
OUTPUT_FOLDERS = {
    "input": "blended",
    "reflection_90": "reflection_90",
    "dolp": "dolp",
    "gt": "transmission_layer",
}
EXPECTED_EXTENSIONS = {
    "input": ".jpg",
    "reflection_90": ".jpg",
    "dolp": ".png",
    "gt": ".jpg",
}
FILE_PATTERN = re.compile(
    r"^(?P<id>.+?)(?P<suffix>_90|_DoLP|_GT)?(?P<ext>\.(?:jpg|png))$",
    re.IGNORECASE,
)


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    while block := stream.read(1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def role_from_suffix(suffix: str | None) -> str:
    if not suffix:
        return "input"
    return {
        "_90": "reflection_90",
        "_dolp": "dolp",
        "_gt": "gt",
    }[suffix.lower()]


def capture_group(sample_id: str) -> str:
    return sample_id.split("_", 1)[0]


def aspect_bucket(ratio: float) -> str:
    if ratio < 0.50:
        return "extreme_portrait"
    if ratio < 0.75:
        return "portrait"
    if ratio < 0.90:
        return "mild_portrait"
    if ratio < 1.10:
        return "near_square"
    if ratio < 1.50:
        return "mild_landscape"
    if ratio < 2.00:
        return "landscape"
    return "extreme_landscape"


def target_size(width: int, height: int, pixels: int, multiple: int) -> tuple[int, int]:
    """Find the closest legal size while preserving aspect ratio and avoiding upscale."""
    source_area = width * height
    desired_area = min(source_area, pixels)
    scale = min(1.0, math.sqrt(pixels / source_area))
    ideal_w = width * scale
    ideal_h = height * scale

    def candidates(value: float, maximum: int) -> list[int]:
        center = value / multiple
        values = {
            max(multiple, int(round(center + offset)) * multiple)
            for offset in (-2, -1, 0, 1, 2)
        }
        legal = sorted(value for value in values if value <= maximum)
        if not legal:
            legal = [max(multiple, (maximum // multiple) * multiple)]
        return legal

    source_ratio = width / height
    best: tuple[float, int, int] | None = None
    for candidate_w in candidates(ideal_w, width):
        for candidate_h in candidates(ideal_h, height):
            candidate_ratio = candidate_w / candidate_h
            ratio_error = abs(math.log(candidate_ratio / source_ratio))
            area_error = abs(math.log((candidate_w * candidate_h) / desired_area))
            score = 8.0 * ratio_error + area_error
            choice = (score, candidate_w, candidate_h)
            if best is None or choice < best:
                best = choice
    assert best is not None
    _, result_w, result_h = best
    return result_w, result_h


def deterministic_split(samples: list[dict], seed: int) -> dict[str, str]:
    """Assign capture groups to an approximately 80/10/10 split.

    The greedy deficit rule balances total sample count and aspect buckets while
    keeping every crop from one capture group in exactly one split.
    """
    split_names = ("train", "validation", "test")
    fractions = {"train": 0.80, "validation": 0.10, "test": 0.10}
    groups: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        groups[sample["group"]].append(sample)

    total = len(samples)
    total_buckets = Counter(sample["aspect_bucket"] for sample in samples)
    desired_total = {name: fractions[name] * total for name in split_names}
    desired_bucket = {
        name: {bucket: fractions[name] * count for bucket, count in total_buckets.items()}
        for name in split_names
    }
    current_total = Counter()
    current_bucket: dict[str, Counter] = {name: Counter() for name in split_names}

    def stable_tie(group: str) -> str:
        return hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()

    ordered = sorted(groups, key=lambda group: (-len(groups[group]), stable_tie(group)))
    assignments: dict[str, str] = {}
    for group in ordered:
        members = groups[group]
        buckets = Counter(member["aspect_bucket"] for member in members)
        scored = []
        for split_index, split in enumerate(split_names):
            total_deficit = (desired_total[split] - current_total[split]) / desired_total[split]
            bucket_terms = []
            for bucket, count in buckets.items():
                desired = max(desired_bucket[split][bucket], 0.5)
                bucket_terms.extend(
                    [(desired - current_bucket[split][bucket]) / desired] * count
                )
            bucket_deficit = sum(bucket_terms) / len(bucket_terms)
            overshoot = max(
                0.0,
                (current_total[split] + len(members) - desired_total[split])
                / desired_total[split],
            )
            score = total_deficit + 0.40 * bucket_deficit - 2.0 * overshoot
            scored.append((score, -split_index, split))
        split = max(scored)[2]
        assignments[group] = split
        current_total[split] += len(members)
        current_bucket[split].update(buckets)

    return assignments


def open_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> Image.Image:
    with zf.open(info) as stream:
        image = Image.open(stream)
        image.load()
    return ImageOps.exif_transpose(image)


def scan_archive(archive: Path, target_pixels: int, multiple: int) -> tuple[list[dict], dict]:
    members: dict[str, dict[str, zipfile.ZipInfo]] = defaultdict(dict)
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            match = FILE_PATTERN.match(name)
            if not match:
                raise ValueError(f"Unexpected file in archive: {info.filename}")
            role = role_from_suffix(match.group("suffix"))
            extension = match.group("ext").lower()
            if extension != EXPECTED_EXTENSIONS[role]:
                raise ValueError(f"Wrong extension for {info.filename}: role={role}")
            sample_id = match.group("id")
            if role in members[sample_id]:
                raise ValueError(f"Duplicate role for {sample_id}: {role}")
            members[sample_id][role] = info

        if not members:
            raise ValueError("Archive contains no samples")
        expected_roles = set(ROLE_SUFFIXES)
        samples = []
        exif_orientations = Counter()
        for sample_id in sorted(members):
            role_members = members[sample_id]
            if set(role_members) != expected_roles:
                raise ValueError(
                    f"Incomplete sample {sample_id}: got={sorted(role_members)}, "
                    f"expected={sorted(expected_roles)}"
                )
            source_records = {}
            logical_sizes = {}
            dolp_stats = None
            for role, info in role_members.items():
                with zf.open(info) as raw_stream:
                    digest = sha256_stream(raw_stream)
                with zf.open(info) as image_stream:
                    encoded = Image.open(image_stream)
                    orientation = int(encoded.getexif().get(274, 1))
                    exif_orientations[(role, orientation)] += 1
                    encoded.load()
                image = ImageOps.exif_transpose(encoded)
                expected_mode = "L" if role == "dolp" else "RGB"
                if image.mode != expected_mode:
                    raise ValueError(
                        f"Unexpected mode for {sample_id}/{role}: {image.mode}, "
                        f"expected {expected_mode}"
                    )
                logical_sizes[role] = image.size
                source_records[role] = {
                    "archive_member": info.filename,
                    "bytes": info.file_size,
                    "sha256": digest,
                    "encoded_size": list(encoded.size),
                    "logical_size": list(image.size),
                    "mode": image.mode,
                    "exif_orientation": orientation,
                }
                if role == "dolp":
                    extrema = image.getextrema()
                    dolp_stats = {
                        "min_u8": int(extrema[0]),
                        "max_u8": int(extrema[1]),
                        "mean_u8": float(ImageStat.Stat(image).mean[0]),
                    }
            if len(set(logical_sizes.values())) != 1:
                raise ValueError(f"Logical size mismatch for {sample_id}: {logical_sizes}")
            width, height = logical_sizes["input"]
            output_w, output_h = target_size(width, height, target_pixels, multiple)
            ratio = width / height
            output_ratio = output_w / output_h
            ratio_error = abs(output_ratio / ratio - 1.0)
            if ratio_error > 0.03:
                raise ValueError(
                    f"Aspect error exceeds 3% for {sample_id}: {ratio_error:.4%}"
                )
            samples.append(
                {
                    "id": sample_id,
                    "group": capture_group(sample_id),
                    "aspect_bucket": aspect_bucket(ratio),
                    "source_size": [width, height],
                    "source_aspect_ratio": ratio,
                    "target_size": [output_w, output_h],
                    "target_aspect_ratio": output_ratio,
                    "aspect_relative_error": ratio_error,
                    "source": source_records,
                    "dolp_stats": dolp_stats,
                }
            )

    scan_stats = {
        "sample_count": len(samples),
        "capture_group_count": len({sample["group"] for sample in samples}),
        "source_width": {
            "min": min(sample["source_size"][0] for sample in samples),
            "median": statistics.median(sample["source_size"][0] for sample in samples),
            "max": max(sample["source_size"][0] for sample in samples),
        },
        "source_height": {
            "min": min(sample["source_size"][1] for sample in samples),
            "median": statistics.median(sample["source_size"][1] for sample in samples),
            "max": max(sample["source_size"][1] for sample in samples),
        },
        "source_aspect_ratio": {
            "min": min(sample["source_aspect_ratio"] for sample in samples),
            "median": statistics.median(sample["source_aspect_ratio"] for sample in samples),
            "max": max(sample["source_aspect_ratio"] for sample in samples),
        },
        "target_width": {
            "min": min(sample["target_size"][0] for sample in samples),
            "median": statistics.median(sample["target_size"][0] for sample in samples),
            "max": max(sample["target_size"][0] for sample in samples),
        },
        "target_height": {
            "min": min(sample["target_size"][1] for sample in samples),
            "median": statistics.median(sample["target_size"][1] for sample in samples),
            "max": max(sample["target_size"][1] for sample in samples),
        },
        "target_pixels": {
            "min": min(sample["target_size"][0] * sample["target_size"][1] for sample in samples),
            "median": statistics.median(
                sample["target_size"][0] * sample["target_size"][1] for sample in samples
            ),
            "max": max(sample["target_size"][0] * sample["target_size"][1] for sample in samples),
        },
        "max_aspect_relative_error": max(
            sample["aspect_relative_error"] for sample in samples
        ),
        "aspect_buckets": dict(Counter(sample["aspect_bucket"] for sample in samples)),
        "exif_orientations": {
            f"{role}:{orientation}": count
            for (role, orientation), count in sorted(exif_orientations.items())
        },
    }
    return samples, scan_stats


def process_image(image: Image.Image, role: str, size: tuple[int, int]) -> Image.Image:
    mode = "L" if role == "dolp" else "RGB"
    image = image.convert(mode)
    if image.size == size:
        return image.copy()
    downscale = size[0] <= image.width and size[1] <= image.height
    if role == "dolp":
        resample = Image.Resampling.BOX if downscale else Image.Resampling.BILINEAR
    else:
        resample = Image.Resampling.LANCZOS if downscale else Image.Resampling.BICUBIC
    return image.resize(size, resample=resample)


def preview_ids(samples: list[dict]) -> list[tuple[str, str]]:
    selectors = [
        ("narrowest", min(samples, key=lambda item: item["source_aspect_ratio"])),
        ("widest", max(samples, key=lambda item: item["source_aspect_ratio"])),
        ("smallest", min(samples, key=lambda item: math.prod(item["source_size"]))),
        ("largest", max(samples, key=lambda item: math.prod(item["source_size"]))),
        (
            "near_square",
            min(samples, key=lambda item: abs(math.log(item["source_aspect_ratio"]))),
        ),
    ]
    result = []
    seen = set()
    for label, sample in selectors:
        if sample["id"] not in seen:
            result.append((label, sample["id"]))
            seen.add(sample["id"])
    return result


def build_previews(root: Path, samples_by_id: dict[str, dict]) -> list[dict]:
    preview_dir = root / "previews"
    preview_dir.mkdir()
    selected = preview_ids(list(samples_by_id.values()))
    rows = []
    cell_w, cell_h = 280, 220
    header_h = 42
    roles = ("input", "reflection_90", "dolp", "gt")
    for label, sample_id in selected:
        sample = samples_by_id[sample_id]
        canvas = Image.new("RGB", (cell_w * 4, cell_h + header_h), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 5),
            f"{label}: {sample_id}  source={sample['source_size']}  target={sample['target_size']}",
            fill="black",
        )
        for column, role in enumerate(roles):
            path = root / OUTPUT_FOLDERS[role] / f"{sample_id}.png"
            with Image.open(path) as loaded:
                image = loaded.convert("RGB")
            fitted = ImageOps.contain(image, (cell_w - 12, cell_h - 28), Image.Resampling.LANCZOS)
            x0 = column * cell_w + (cell_w - fitted.width) // 2
            y0 = header_h + 20 + (cell_h - 28 - fitted.height) // 2
            canvas.paste(fitted, (x0, y0))
            draw.text((column * cell_w + 6, header_h + 2), role, fill="black")
        filename = f"{label}_{sample_id}.png"
        canvas.save(preview_dir / filename, format="PNG", compress_level=6)
        rows.append({"label": label, "id": sample_id, "file": f"previews/{filename}"})

    overview = Image.new("RGB", (cell_w * 4, len(rows) * (cell_h + header_h)), "white")
    for index, row in enumerate(rows):
        with Image.open(root / row["file"]) as image:
            overview.paste(image, (0, index * (cell_h + header_h)))
    overview.save(preview_dir / "overview.png", format="PNG", compress_level=6)
    return rows


def prepare(args: argparse.Namespace) -> None:
    archive = args.archive.resolve()
    output = args.output.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)

    print(f"[1/5] Scanning and decoding {archive}", flush=True)
    samples, scan_stats = scan_archive(archive, args.target_pixels, args.multiple)
    assignments = deterministic_split(samples, args.seed)
    for sample in samples:
        sample["split"] = assignments[sample["group"]]

    split_counts = Counter(sample["split"] for sample in samples)
    split_groups = {
        split: len({sample["group"] for sample in samples if sample["split"] == split})
        for split in ("train", "validation", "test")
    }
    print(f"[2/5] Split counts: {dict(split_counts)}; groups: {split_groups}", flush=True)

    staging = output.with_name(f".{output.name}.building-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for folder in OUTPUT_FOLDERS.values():
        (staging / folder).mkdir()
    (staging / "splits").mkdir()

    archive_sha = sha256_path(archive)
    sample_map = {sample["id"]: sample for sample in samples}
    print("[3/5] Resizing aligned modalities without crop or padding", flush=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            for index, sample in enumerate(samples, start=1):
                target = tuple(sample["target_size"])
                processed = {}
                for role in ROLE_SUFFIXES:
                    source_record = sample["source"][role]
                    info = zf.getinfo(source_record["archive_member"])
                    image = open_member(zf, info)
                    result = process_image(image, role, target)
                    destination = staging / OUTPUT_FOLDERS[role] / f"{sample['id']}.png"
                    result.save(destination, format="PNG", compress_level=6)
                    with Image.open(destination) as check:
                        check.load()
                        expected_mode = "L" if role == "dolp" else "RGB"
                        if check.size != target or check.mode != expected_mode:
                            raise RuntimeError(
                                f"Processed verification failed for {sample['id']}/{role}: "
                                f"size={check.size}, mode={check.mode}"
                            )
                    processed[role] = {
                        "path": str(destination.relative_to(staging)),
                        "bytes": destination.stat().st_size,
                        "sha256": sha256_path(destination),
                        "mode": "L" if role == "dolp" else "RGB",
                        "size": list(target),
                    }
                sample["processed"] = processed
                if index % 20 == 0 or index == len(samples):
                    print(f"  processed {index}/{len(samples)}", flush=True)

        for split in ("train", "validation", "test"):
            ids = sorted(sample["id"] for sample in samples if sample["split"] == split)
            (staging / "splits" / f"{split}.txt").write_text(
                "\n".join(ids) + "\n", encoding="utf-8"
            )

        print("[4/5] Building five geometry previews", flush=True)
        previews = build_previews(staging, sample_map)
        output_bytes = sum(
            path.stat().st_size for path in staging.rglob("*") if path.is_file()
        )
        manifest = {
            "complete": True,
            "version": VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_archive": {
                "path": str(archive),
                "bytes": archive.stat().st_size,
                "sha256": archive_sha,
            },
            "transform": {
                "policy": "preserve-aspect-no-crop-no-padding",
                "target_pixels": args.target_pixels,
                "dimension_multiple": args.multiple,
                "allow_upscale": False,
                "rgb_downsample": "Pillow Lanczos",
                "rgb_upscale": "Pillow bicubic (normally disabled)",
                "dolp_downsample": "Pillow BOX area approximation",
                "dolp_upscale": "Pillow bilinear (normally disabled)",
                "output_format": "lossless PNG",
            },
            "split": {
                "seed": args.seed,
                "group_key": "substring before first underscore",
                "requested_fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
                "sample_counts": dict(split_counts),
                "group_counts": split_groups,
            },
            "scan_stats": scan_stats,
            "preview_samples": previews,
            "processed_bytes_before_manifest": output_bytes,
            "samples": samples,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        readme = (
            "# RMagNet M2 processed dataset\n\n"
            f"- Version: `{VERSION}`\n"
            f"- Samples: {len(samples)}\n"
            f"- Split: train={split_counts['train']}, validation={split_counts['validation']}, "
            f"test={split_counts['test']}\n"
            "- Geometry: preserve aspect ratio, no crop, no padding, dimensions divisible by 16.\n"
            "- Preview: `previews/overview.png`\n"
            "- Full provenance and SHA-256 values: `manifest.json`\n"
        )
        (staging / "README.md").write_text(readme, encoding="utf-8")
        print("[5/5] Publishing completed dataset atomically", flush=True)
        staging.rename(output)
    except BaseException:
        print(f"Preparation failed; incomplete staging kept at {staging}", flush=True)
        raise

    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "samples": len(samples),
                "split_counts": dict(split_counts),
                "target_size_range": {
                    "width": scan_stats["target_width"],
                    "height": scan_stats["target_height"],
                    "pixels": scan_stats["target_pixels"],
                },
                "max_aspect_relative_error": scan_stats["max_aspect_relative_error"],
                "preview": str(output / "previews" / "overview.png"),
            },
            indent=2,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-pixels", type=int, default=512 * 384)
    parser.add_argument("--multiple", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.target_pixels <= 0:
        parser.error("--target-pixels must be positive")
    if args.multiple <= 0:
        parser.error("--multiple must be positive")
    return args


if __name__ == "__main__":
    prepare(parse_args())
