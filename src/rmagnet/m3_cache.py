"""Build and audit the M3 offline semantic-separation cache.

The extractor is safely shardable across GPUs.  Finalization reuses the corrected
M2 Q20(GT) cache, fits train-only PCA/prototypes, writes compact soft-state and
relation targets, and never touches validation/test examples.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment

from .c1_l20_prepare import heat, preview_panel, resize_float
from .m2a_prepare import q20_feature
from .qwen_backend import QwenSharedBackend
from .stage1_train import image_tensor


ROOT = Path("/share/linmingheng-local/xuke")
PROJECT = ROOT / "RMagNet"
DEFAULT_DATA = ROOT / "datasets/rmagnet_m2_aspect"
DEFAULT_SOURCE_CACHE = PROJECT / "data_cache/m2a_q20"
DEFAULT_OUTPUT = PROJECT / "data_cache/m3_semantic_v1"
DATA_VERSION = "m2-variable-aspect-v2-corrected-labels"
SOURCE_FORMULA = "m2a-q20-variable-aspect-v1"
CACHE_VERSION = "m3-semantic-separation-v1"
BLOCK_INDEX = 19
BLOCK_NUMBER = 20
FLOW_TIMESTEP = 499
HIDDEN_SIZE = 3072
PCA_DIM = 32
PCA_SAMPLE_LIMIT = 32768
CLUSTERS = 4
SEEDS = (2026, 2027, 2028)
SCALAR_NAMES = (
    "d_i",
    "d_90",
    "residual_alignment",
    "d_i_90",
    "rgb_i",
    "rgb_90",
    "gradient_i",
    "gradient_90",
    "dolp",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(text: str, base: int = 2026) -> int:
    value = hashlib.sha256(f"{base}:{text}".encode()).digest()
    return int.from_bytes(value[:8], "little") % (2**31 - 1)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    safetensors.torch.save_file(
        {name: value.detach().contiguous().cpu() for name, value in tensors.items()},
        temporary,
        metadata=metadata,
    )
    os.replace(temporary, path)


def current_git_commit() -> str:
    return subprocess.check_output(["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True).strip()


def load_context(data_root: Path, source_cache: Path) -> tuple[dict, dict[str, dict], list[str], dict, dict[str, dict]]:
    dataset_manifest_path = data_root / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if not dataset_manifest.get("complete") or dataset_manifest.get("version") != DATA_VERSION:
        raise RuntimeError("M3 requires the corrected M2 dataset manifest")
    train_ids = (data_root / "splits/train.txt").read_text(encoding="utf-8").split()
    if len(train_ids) != 144 or len(set(train_ids)) != 144:
        raise RuntimeError(f"Expected 144 unique train IDs, got {len(train_ids)}")
    records = {record["id"]: record for record in dataset_manifest["samples"]}
    if not set(train_ids) <= set(records):
        raise RuntimeError("Train split contains IDs absent from the dataset manifest")

    cache_manifest_path = source_cache / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    identity = cache_manifest.get("identity", {})
    source = cache_manifest.get("source_dataset", {})
    if not cache_manifest.get("complete") or identity.get("formula_version") != SOURCE_FORMULA:
        raise RuntimeError("The M2a Q20 source cache is incomplete or incompatible")
    if identity.get("block_zero_based_index") != BLOCK_INDEX or identity.get("flow_timestep") != FLOW_TIMESTEP:
        raise RuntimeError("The source cache uses another Qwen layer or timestep")
    if identity.get("selected_ids") != train_ids or source.get("cached_count") != 144:
        raise RuntimeError("The source cache does not cover the exact M2 train split")
    dataset_sha = sha256(dataset_manifest_path)
    if source.get("manifest_sha256") != dataset_sha:
        raise RuntimeError("Dataset manifest SHA differs from the source cache identity")
    cache_records = {record["id"]: record for record in cache_manifest["samples"]}
    if set(cache_records) != set(train_ids):
        raise RuntimeError("Source cache IDs differ from train IDs")

    for sample_id in train_ids:
        record = records[sample_id]
        width, height = record["target_size"]
        paths = sample_paths(data_root, sample_id)
        expected_modes = {"input": "RGB", "p90": "RGB", "gt": "RGB", "dolp": "L"}
        for role, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                if image.size != (width, height) or image.mode != expected_modes[role]:
                    raise ValueError(f"Unaligned M3 source {sample_id}/{role}: {image.size}/{image.mode}")
        cached = cache_records[sample_id]
        grid_h, grid_w = cached["token_grid_hw"]
        if (grid_h, grid_w) != (height // 16, width // 16):
            raise ValueError(f"Token grid mismatch for {sample_id}")
        feature_path = source_cache / cached["cache"]["gt_feature"]
        if not feature_path.is_file() or sha256(feature_path) != cached["cache"]["gt_feature_sha256"]:
            raise RuntimeError(f"Broken Q20(GT) source cache: {sample_id}")
    return dataset_manifest, records, train_ids, cache_manifest, cache_records


def sample_paths(root: Path, sample_id: str) -> dict[str, Path]:
    return {
        "input": root / "blended" / f"{sample_id}.png",
        "p90": root / "reflection_90" / f"{sample_id}.png",
        "gt": root / "transmission_layer" / f"{sample_id}.png",
        "dolp": root / "dolp" / f"{sample_id}.png",
    }


def load_c(source_cache: Path, cache_record: dict) -> torch.Tensor:
    path = source_cache / cache_record["cache"]["gt_feature"]
    value = safetensors.torch.load_file(path)["q20_gt"]
    if value.ndim != 2 or value.shape[1] != HIDDEN_SIZE or value.dtype != torch.bfloat16:
        raise ValueError(f"Unexpected cached C tensor: {path} {value.shape}/{value.dtype}")
    return value


def extraction_record_valid(feature_path: Path, record_path: Path, sample_id: str) -> bool:
    if not feature_path.is_file() or not record_path.is_file():
        return False
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        return record.get("id") == sample_id and record.get("feature_sha256") == sha256(feature_path)
    except Exception:
        return False


def extract(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    source_cache = args.source_cache.resolve()
    output = args.output.resolve()
    _, records, train_ids, _, cache_records = load_context(data_root, source_cache)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    selected = train_ids[args.shard_index :: args.num_shards]
    feature_dir = output / "scratch/features"
    record_dir = output / "scratch/records"
    feature_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    remaining = []
    for sample_id in selected:
        feature_path = feature_dir / f"{sample_id}.safetensors"
        record_path = record_dir / f"{sample_id}.json"
        if not extraction_record_valid(feature_path, record_path, sample_id):
            remaining.append(sample_id)
    print(json.dumps({
        "status": "resume" if len(remaining) != len(selected) else "start",
        "shard": args.shard_index,
        "num_shards": args.num_shards,
        "selected": len(selected),
        "remaining": len(remaining),
        "device": args.device,
    }), flush=True)
    if not remaining:
        return

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(2026 + args.shard_index)
    torch.cuda.manual_seed(2026 + args.shard_index)
    backend = QwenSharedBackend.from_local(device)
    backend.set_trainable_branch(None)
    backend.transformer.eval()
    backend.vae.eval()
    if any(parameter.requires_grad for parameter in backend.transformer.parameters()):
        raise RuntimeError("M3 extraction requires a frozen Qwen transformer")
    try:
        for position, sample_id in enumerate(remaining, 1):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            record = records[sample_id]
            width, height = record["target_size"]
            grid_h, grid_w = height // 16, width // 16
            expected_shape = (1, grid_h * grid_w, HIDDEN_SIZE)
            paths = sample_paths(data_root, sample_id)
            c = load_c(source_cache, cache_records[sample_id]).float()

            q_i = q20_feature(backend, image_tensor(paths["input"])[None])
            if tuple(q_i.shape) != expected_shape:
                raise ValueError(f"Unexpected Q20(I) shape for {sample_id}: {q_i.shape}")
            e_i = (q_i[0].float() - c).to(torch.bfloat16)
            del q_i
            torch.cuda.empty_cache()

            q_90 = q20_feature(backend, image_tensor(paths["p90"])[None])
            if tuple(q_90.shape) != expected_shape:
                raise ValueError(f"Unexpected Q20(P90) shape for {sample_id}: {q_90.shape}")
            e_90 = (q_90[0].float() - c).to(torch.bfloat16)
            del q_90, c
            torch.cuda.synchronize(device)
            if not torch.isfinite(e_i.float()).all() or not torch.isfinite(e_90.float()).all():
                raise RuntimeError(f"Non-finite residual features for {sample_id}")

            feature_path = feature_dir / f"{sample_id}.safetensors"
            atomic_safetensors(
                feature_path,
                {"e_i": e_i, "e_90": e_90},
                {
                    "sample_id": sample_id,
                    "cache_version": CACHE_VERSION,
                    "token_grid_hw": f"{grid_h},{grid_w}",
                    "dtype": "bfloat16",
                },
            )
            peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30
            peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30
            sidecar = {
                "id": sample_id,
                "shard_index": args.shard_index,
                "image_size_wh": [width, height],
                "token_grid_hw": [grid_h, grid_w],
                "shape": [grid_h * grid_w, HIDDEN_SIZE],
                "feature": str(feature_path.relative_to(output)),
                "feature_sha256": sha256(feature_path),
                "source": {role: {"path": str(path), "sha256": sha256(path)} for role, path in paths.items()},
                "q20_gt_sha256": cache_records[sample_id]["cache"]["gt_feature_sha256"],
                "memory_gib": {"peak_allocated": peak_allocated, "peak_reserved": peak_reserved},
                "completed_at_utc": utc_now(),
            }
            atomic_json(record_dir / f"{sample_id}.json", sidecar)
            print(json.dumps({
                "sample": sample_id,
                "progress": f"{position}/{len(remaining)}",
                "shard": args.shard_index,
                "peak_allocated_gib": round(peak_allocated, 3),
                "peak_reserved_gib": round(peak_reserved, 3),
            }), flush=True)
            del e_i, e_90
    finally:
        del backend
        gc.collect()
        torch.cuda.empty_cache()


def deterministic_indices(total: int, count: int, key: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(stable_seed(key))
    if count >= total:
        return torch.arange(total)
    return torch.randperm(total, generator=generator)[:count]


def gather_pca_samples(
    train_ids: list[str],
    source_cache: Path,
    cache_records: dict[str, dict],
    scratch: Path,
    kind: str,
    limit: int,
) -> torch.Tensor:
    chunks = []
    if kind == "content":
        per_sample = math.ceil(limit / len(train_ids))
        for sample_id in train_ids:
            value = load_c(source_cache, cache_records[sample_id]).float()
            index = deterministic_indices(value.shape[0], per_sample, f"pca-content:{sample_id}")
            chunks.append(value[index])
    elif kind == "residual":
        per_modality = math.ceil(limit / (2 * len(train_ids)))
        for sample_id in train_ids:
            stored = safetensors.torch.load_file(scratch / "features" / f"{sample_id}.safetensors")
            for name in ("e_i", "e_90"):
                value = stored[name].float()
                index = deterministic_indices(value.shape[0], per_modality, f"pca-{name}:{sample_id}")
                chunks.append(value[index])
    else:
        raise ValueError(kind)
    matrix = torch.cat(chunks, 0)
    if matrix.shape[0] > limit:
        index = deterministic_indices(matrix.shape[0], limit, f"pca-final:{kind}")
        matrix = matrix[index]
    if matrix.shape[1] != HIDDEN_SIZE or not torch.isfinite(matrix).all():
        raise RuntimeError(f"Invalid {kind} PCA matrix: {matrix.shape}")
    return matrix


def fit_pca(matrix: torch.Tensor, device: torch.device, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    mean = matrix.mean(0)
    centered = matrix - mean
    work = centered.to(device)
    _, singular, components = torch.pca_lowrank(work, q=PCA_DIM, center=False, niter=4)
    components = components.cpu()
    singular = singular.cpu()
    total_variance = centered.square().sum().clamp_min(1e-12)
    explained = singular.square() / total_variance
    result = {
        "mean": mean.cpu().float(),
        "components": components.float(),
        "singular_values": singular.float(),
        "explained_variance_ratio": explained.float(),
    }
    del work
    torch.cuda.empty_cache()
    return result


def project(values: torch.Tensor, pca: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    result = (values.to(device).float() - pca["mean"].to(device)) @ pca["components"].to(device)
    return F.normalize(result, dim=-1, eps=1e-6).cpu()


def image_float(path: Path, grayscale: bool = False) -> np.ndarray:
    with Image.open(path) as image:
        mode = "L" if grayscale else "RGB"
        return np.asarray(image.convert(mode), dtype=np.float32) / 255.0


def gradient_magnitude(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.mean(2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    return np.sqrt(np.square(gx) + np.square(gy))


def token_resize(values: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    return resize_float(values.astype(np.float32), (grid_w, grid_h)).reshape(-1).astype(np.float32)


def cosine_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(first.float(), second.float(), dim=-1, eps=1e-6)


def build_raw_sample(
    sample_id: str,
    data_root: Path,
    source_cache: Path,
    cache_record: dict,
    scratch: Path,
    content_pca: dict[str, torch.Tensor],
    residual_pca: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, object]:
    c = load_c(source_cache, cache_record).float()
    stored = safetensors.torch.load_file(scratch / "features" / f"{sample_id}.safetensors")
    e_i = stored["e_i"].float()
    e_90 = stored["e_90"].float()
    q_i = c + e_i
    q_90 = c + e_90
    grid_h, grid_w = cache_record["token_grid_hw"]
    paths = sample_paths(data_root, sample_id)
    rgb_i = image_float(paths["input"])
    rgb_90 = image_float(paths["p90"])
    rgb_gt = image_float(paths["gt"])
    dolp = image_float(paths["dolp"], grayscale=True)
    d_i = cosine_distance(q_i, c)
    d_90 = cosine_distance(q_90, c)
    alignment = F.cosine_similarity(e_i, e_90, dim=-1, eps=1e-6).clamp_min(0)
    d_i_90 = cosine_distance(q_i, q_90)
    diff_i = np.abs(rgb_i - rgb_gt).mean(2)
    diff_90 = np.abs(rgb_90 - rgb_gt).mean(2)
    grad_gt = gradient_magnitude(rgb_gt)
    grad_i = np.abs(gradient_magnitude(rgb_i) - grad_gt)
    grad_90 = np.abs(gradient_magnitude(rgb_90) - grad_gt)
    scalar = torch.stack([
        d_i,
        d_90,
        alignment,
        d_i_90,
        torch.from_numpy(token_resize(diff_i, grid_h, grid_w)),
        torch.from_numpy(token_resize(diff_90, grid_h, grid_w)),
        torch.from_numpy(token_resize(grad_i, grid_h, grid_w)),
        torch.from_numpy(token_resize(grad_90, grid_h, grid_w)),
        torch.from_numpy(token_resize(dolp, grid_h, grid_w)),
    ], 1).float()
    if scalar.shape != (grid_h * grid_w, len(SCALAR_NAMES)) or not torch.isfinite(scalar).all():
        raise RuntimeError(f"Invalid scalar cues for {sample_id}: {scalar.shape}")
    raw_reflection = torch.sqrt(d_i.clamp_min(0) * d_90.clamp_min(0))
    raw_reflection = raw_reflection * alignment * (0.7 + 0.3 * scalar[:, -1])
    return {
        "id": sample_id,
        "grid": (grid_h, grid_w),
        "content": project(c, content_pca, device),
        "residual_i": project(e_i, residual_pca, device),
        "residual_90": project(e_90, residual_pca, device),
        "scalar": scalar,
        "raw_reflection": raw_reflection.float(),
    }


def robust_stats(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    low = torch.quantile(values.float(), 0.02, dim=0)
    high = torch.quantile(values.float(), 0.98, dim=0)
    high = torch.maximum(high, low + 1e-6)
    return low, high


def robust_apply(values: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return ((values.float() - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)


def sinkhorn(logits: torch.Tensor, iterations: int = 500, tolerance: float = 1e-4) -> torch.Tensor:
    """Balanced assignment in the log domain to avoid peaked-logit collapse."""
    log_q = logits.double().t()
    k, n = log_q.shape
    log_q -= torch.logsumexp(log_q.flatten(), dim=0)
    for iteration in range(iterations):
        log_q -= torch.logsumexp(log_q, dim=1, keepdim=True)
        log_q -= math.log(k)
        log_q -= torch.logsumexp(log_q, dim=0, keepdim=True)
        log_q -= math.log(n)
        if iteration % 10 == 9:
            occupancy = torch.exp(torch.logsumexp(log_q, dim=1))
            if float((occupancy - 1.0 / k).abs().max()) < tolerance:
                break
    return torch.exp(log_q + math.log(n)).t().float()


def kmeans_plus_plus(values: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=values.device).manual_seed(seed)
    first = int(torch.randint(values.shape[0], (1,), generator=generator, device=values.device))
    chosen = [first]
    distance = torch.cdist(values, values[first : first + 1]).square().squeeze(1)
    for _ in range(1, k):
        probability = distance / distance.sum().clamp_min(1e-12)
        index = int(torch.multinomial(probability, 1, generator=generator))
        chosen.append(index)
        new_distance = torch.cdist(values, values[index : index + 1]).square().squeeze(1)
        distance = torch.minimum(distance, new_distance)
    return values[torch.tensor(chosen, device=values.device)].clone()


def fit_prototypes(values: torch.Tensor, seed: int, device: torch.device) -> tuple[torch.Tensor, dict]:
    work = values.to(device)
    prototypes = kmeans_plus_plus(work, CLUSTERS, seed)
    converged = False
    final_delta = math.inf
    temperature = 0.20
    for iteration in range(50):
        distance = torch.cdist(work, prototypes).square()
        temperature = max(0.05, 0.5 * float(distance.min(1).values.median()))
        posterior = sinkhorn(-distance / temperature)
        updated = posterior.t() @ work
        updated /= posterior.sum(0).unsqueeze(1).clamp_min(1e-6)
        final_delta = float((updated - prototypes).square().mean().sqrt())
        prototypes = updated
        if final_delta < 1e-5:
            converged = True
            break
    report = {
        "seed": seed,
        "iterations": iteration + 1,
        "converged": converged,
        "final_rms_delta": final_delta,
        "temperature": temperature,
        "fit_occupancy": posterior.mean(0).cpu().tolist(),
    }
    del work, posterior
    torch.cuda.empty_cache()
    return prototypes.cpu(), report


def prototype_stability(primary: torch.Tensor, other: torch.Tensor) -> tuple[float, list[int]]:
    first = F.normalize(primary.float(), dim=1)
    second = F.normalize(other.float(), dim=1)
    similarity = (first @ second.t()).numpy()
    rows, columns = linear_sum_assignment(-similarity)
    order = columns[np.argsort(rows)].tolist()
    return float(similarity[rows, columns].mean()), order


def balanced_assign(
    values: torch.Tensor,
    prototypes: torch.Tensor,
    device: torch.device,
    temperature: float,
) -> torch.Tensor:
    chunks = []
    prototypes = prototypes.to(device)
    for start in range(0, values.shape[0], 32768):
        work = values[start : start + 32768].to(device)
        chunks.append((-torch.cdist(work, prototypes).square() / temperature).cpu())
    logits = torch.cat(chunks, 0).to(device)
    result = sinkhorn(logits).cpu()
    del logits, prototypes
    torch.cuda.empty_cache()
    return result


def smooth_posterior(posterior: torch.Tensor, content: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    grid_h, grid_w = grid
    p = posterior.reshape(grid_h, grid_w, CLUSTERS).float()
    c = F.normalize(content.float(), dim=1).reshape(grid_h, grid_w, PCA_DIM)
    for _ in range(3):
        numerator = p.clone()
        denominator = torch.ones((grid_h, grid_w, 1))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r0, r1 = max(0, -dr), min(grid_h, grid_h - dr)
            c0, c1 = max(0, -dc), min(grid_w, grid_w - dc)
            nr0, nr1 = r0 + dr, r1 + dr
            nc0, nc1 = c0 + dc, c1 + dc
            similarity = (c[r0:r1, c0:c1] * c[nr0:nr1, nc0:nc1]).sum(-1, keepdim=True)
            weight = torch.exp((similarity - 1.0) / 0.10).clamp_min(1e-4)
            numerator[r0:r1, c0:c1] += weight * p[nr0:nr1, nc0:nc1]
            denominator[r0:r1, c0:c1] += weight
        neighbour = numerator / denominator
        p = 0.5 * p + 0.5 * neighbour
        p /= p.sum(-1, keepdim=True).clamp_min(1e-6)
    return p.reshape(-1, CLUSTERS)


def build_relations(c: torch.Tensor, content: torch.Tensor, grid: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grid_h, grid_w = grid
    n = grid_h * grid_w
    pairs: set[tuple[int, int]] = set()
    offsets = ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0))
    for row in range(grid_h):
        for col in range(grid_w):
            first = row * grid_w + col
            for dr, dc in offsets:
                other_r, other_c = row + dr, col + dc
                if 0 <= other_r < grid_h and 0 <= other_c < grid_w:
                    second = other_r * grid_w + other_c
                    pairs.add((min(first, second), max(first, second)))
    projected = F.normalize(content.float(), dim=1)
    similarity = projected @ projected.t()
    similarity_top = similarity.clone()
    similarity_bottom = similarity.clone()
    similarity_top.fill_diagonal_(-2)
    similarity_bottom.fill_diagonal_(2)
    top = torch.topk(similarity_top, k=2, dim=1, largest=True).indices
    bottom = torch.topk(similarity_bottom, k=2, dim=1, largest=False).indices
    for first in range(n):
        for second in itertools.chain(top[first].tolist(), bottom[first].tolist()):
            if first != second:
                pairs.add((min(first, second), max(first, second)))
    edge_index = torch.tensor(sorted(pairs), dtype=torch.int64).t().contiguous()
    normalized_c = F.normalize(c.float(), dim=1)
    raw = (normalized_c[edge_index[0]] * normalized_c[edge_index[1]]).sum(1)
    tau = raw.median()
    target = torch.sigmoid((raw - tau) / 0.07)
    confidence = 0.1 + 0.9 * (2 * (target - 0.5).abs()).clamp(0, 1)
    return edge_index.to(torch.int32), target.to(torch.float16), confidence.to(torch.float16)


def dominant_cluster_image(posterior: torch.Tensor, grid: tuple[int, int], size: tuple[int, int]) -> Image.Image:
    palette = np.asarray([[45, 75, 190], [32, 160, 115], [235, 165, 35], [190, 55, 75]], dtype=np.uint8)
    labels = posterior.argmax(1).reshape(*grid).numpy()
    image = Image.fromarray(palette[labels], mode="RGB")
    return image.resize(size, Image.Resampling.NEAREST)


def preview(
    sample_id: str,
    paths: dict[str, Path],
    posterior: torch.Tensor,
    reflection: torch.Tensor,
    boundary: torch.Tensor,
    grid: tuple[int, int],
    output: Path,
) -> Path:
    with Image.open(paths["input"]) as image:
        inp = image.convert("RGB")
    with Image.open(paths["gt"]) as image:
        gt = image.convert("RGB")
    with Image.open(paths["p90"]) as image:
        p90 = image.convert("RGB")
    max_width = 240
    scale = min(1.0, max_width / inp.width)
    size = (max(1, round(inp.width * scale)), max(1, round(inp.height * scale)))
    inp = inp.resize(size, Image.Resampling.LANCZOS)
    gt = gt.resize(size, Image.Resampling.LANCZOS)
    p90 = p90.resize(size, Image.Resampling.LANCZOS)
    cluster = dominant_cluster_image(posterior, grid, size)
    r_image = heat(reflection.reshape(*grid).numpy(), size)
    b_image = heat(boundary.reshape(*grid).numpy(), size)
    relative = Path("previews") / f"{sample_id}_panel.png"
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    preview_panel([
        (f"{sample_id} I", inp),
        ("GT", gt),
        ("P90", p90),
        ("soft-state argmax", cluster),
        ("reflection evidence R", r_image),
        ("soft boundary B", b_image),
    ]).save(path, compress_level=6)
    return relative


def finalize(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    source_cache = args.source_cache.resolve()
    output = args.output.resolve()
    scratch = output / "scratch"
    complete_path = output / "manifest.json"
    if complete_path.is_file():
        manifest = json.loads(complete_path.read_text(encoding="utf-8"))
        if manifest.get("complete") and manifest.get("cache_version") == CACHE_VERSION:
            print(json.dumps({"status": "already_complete", "output": str(output)}))
            return
        raise RuntimeError("Refusing to overwrite an incompatible M3 manifest")
    dataset_manifest, records, train_ids, source_manifest, cache_records = load_context(data_root, source_cache)
    extraction_records = {}
    for sample_id in train_ids:
        feature_path = scratch / "features" / f"{sample_id}.safetensors"
        record_path = scratch / "records" / f"{sample_id}.json"
        if not extraction_record_valid(feature_path, record_path, sample_id):
            raise RuntimeError(f"Missing or invalid extraction scratch: {sample_id}")
        extraction_records[sample_id] = json.loads(record_path.read_text(encoding="utf-8"))

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("pca", "prototypes", "samples", "previews", "audit"):
        (output / folder).mkdir(exist_ok=True)

    content_matrix = gather_pca_samples(train_ids, source_cache, cache_records, scratch, "content", PCA_SAMPLE_LIMIT)
    content_pca = fit_pca(content_matrix, device, 2026)
    del content_matrix
    residual_matrix = gather_pca_samples(train_ids, source_cache, cache_records, scratch, "residual", PCA_SAMPLE_LIMIT)
    residual_pca = fit_pca(residual_matrix, device, 2027)
    del residual_matrix
    atomic_safetensors(output / "pca/content_pca.safetensors", content_pca, {"kind": "Q20(GT)", "dimension": str(PCA_DIM)})
    atomic_safetensors(output / "pca/residual_pca.safetensors", residual_pca, {"kind": "shared E_I/E_90", "dimension": str(PCA_DIM)})

    raw_samples: list[dict[str, object]] = []
    for position, sample_id in enumerate(train_ids, 1):
        raw_samples.append(build_raw_sample(
            sample_id, data_root, source_cache, cache_records[sample_id], scratch,
            content_pca, residual_pca, device,
        ))
        print(json.dumps({"stage": "features", "progress": f"{position}/{len(train_ids)}", "sample": sample_id}), flush=True)
    all_scalar = torch.cat([item["scalar"] for item in raw_samples], 0)
    scalar_low, scalar_high = robust_stats(all_scalar)
    all_reflection = torch.cat([item["raw_reflection"] for item in raw_samples], 0)
    reflection_low, reflection_high = robust_stats(all_reflection[:, None])

    z_parts = []
    for item in raw_samples:
        scalar_normalized = robust_apply(item["scalar"], scalar_low, scalar_high)
        item["scalar_normalized"] = scalar_normalized
        z_parts.append(torch.cat([
            0.25 * item["content"],
            item["residual_i"],
            item["residual_90"],
            scalar_normalized,
        ], 1))
    all_z = torch.cat(z_parts, 0)
    if not torch.isfinite(all_z).all() or all_z.shape[1] != PCA_DIM * 3 + len(SCALAR_NAMES):
        raise RuntimeError(f"Invalid clustering matrix: {all_z.shape}")
    fit_index = deterministic_indices(all_z.shape[0], min(PCA_SAMPLE_LIMIT, all_z.shape[0]), "prototype-fit")
    fit_values = all_z[fit_index]
    fitted = []
    fit_reports = []
    for seed in SEEDS:
        prototypes, report = fit_prototypes(fit_values, seed, device)
        fitted.append(prototypes)
        fit_reports.append(report)
    stability = []
    for seed, other in zip(SEEDS[1:], fitted[1:]):
        score, order = prototype_stability(fitted[0], other)
        stability.append({"seed": seed, "mean_matched_cosine": score, "primary_to_seed_order": order})
    minimum_stability = min(item["mean_matched_cosine"] for item in stability)
    if minimum_stability < 0.80:
        raise RuntimeError(f"Prototype seed stability failed: {minimum_stability:.6f}")
    prototypes = fitted[0]
    posterior_all = balanced_assign(all_z, prototypes, device, fit_reports[0]["temperature"])
    atomic_safetensors(
        output / "prototypes/prototypes.safetensors",
        {"prototypes": prototypes.float(), "scalar_low": scalar_low, "scalar_high": scalar_high,
         "reflection_low": reflection_low, "reflection_high": reflection_high},
        {"cache_version": CACHE_VERSION, "clusters": str(CLUSTERS)},
    )
    fit_report = {
        "seeds": list(SEEDS),
        "fits": fit_reports,
        "stability": stability,
        "minimum_stability": minimum_stability,
        "fit_sample_count": int(fit_values.shape[0]),
        "feature_dimension": int(all_z.shape[1]),
        "balanced_global_occupancy_before_smoothing": posterior_all.mean(0).tolist(),
    }
    atomic_json(output / "prototypes/fit_report.json", fit_report)

    offsets = []
    cursor = 0
    for item in raw_samples:
        count = item["content"].shape[0]
        offsets.append((cursor, cursor + count))
        cursor += count
    final_records = []
    occupancy_sum = torch.zeros(CLUSTERS)
    token_total = 0
    for position, (item, (start, end)) in enumerate(zip(raw_samples, offsets), 1):
        sample_id = item["id"]
        grid = item["grid"]
        posterior = smooth_posterior(posterior_all[start:end], item["content"], grid)
        entropy = -(posterior.clamp_min(1e-8) * posterior.clamp_min(1e-8).log()).sum(1) / math.log(CLUSTERS)
        confidence = (1.0 - entropy).clamp(0, 1)
        reflection = robust_apply(item["raw_reflection"][:, None], reflection_low, reflection_high).squeeze(1)
        r_grid = reflection.reshape(*grid)
        gradient = torch.zeros_like(r_grid)
        gradient[:, 1:] += (r_grid[:, 1:] - r_grid[:, :-1]).abs()
        gradient[1:, :] += (r_grid[1:, :] - r_grid[:-1, :]).abs()
        boundary_raw = 0.5 * entropy.reshape(*grid) + 0.5 * gradient
        b_low, b_high = robust_stats(boundary_raw.flatten()[:, None])
        boundary = robust_apply(boundary_raw.flatten()[:, None], b_low, b_high).squeeze(1)
        c = load_c(source_cache, cache_records[sample_id])
        edge_index, edge_target, edge_confidence = build_relations(c, item["content"], grid)
        sample_path = output / "samples" / f"{sample_id}.safetensors"
        atomic_safetensors(
            sample_path,
            {
                "posterior": posterior.t().to(torch.float16),
                "confidence": confidence.to(torch.float16),
                "reflection_evidence": reflection.to(torch.float16),
                "boundary": boundary.to(torch.float16),
                "edge_index": edge_index,
                "edge_target": edge_target,
                "edge_confidence": edge_confidence,
                "token_grid_hw": torch.tensor(grid, dtype=torch.int32),
            },
            {"sample_id": sample_id, "cache_version": CACHE_VERSION, "clusters": str(CLUSTERS)},
        )
        paths = sample_paths(data_root, sample_id)
        preview_rel = preview(sample_id, paths, posterior, reflection, boundary, grid, output)
        occupancy_sum += posterior.sum(0)
        token_total += posterior.shape[0]
        final_records.append({
            "id": sample_id,
            "group": records[sample_id]["group"],
            "aspect_bucket": records[sample_id]["aspect_bucket"],
            "image_size_wh": records[sample_id]["target_size"],
            "token_grid_hw": list(grid),
            "token_count": posterior.shape[0],
            "source": extraction_records[sample_id]["source"],
            "source_q20_gt_sha256": extraction_records[sample_id]["q20_gt_sha256"],
            "scratch_feature_sha256": extraction_records[sample_id]["feature_sha256"],
            "cache": {
                "sample": str(sample_path.relative_to(output)),
                "sample_sha256": sha256(sample_path),
                "preview": str(preview_rel),
                "preview_sha256": sha256(output / preview_rel),
            },
            "stats": {
                "posterior_mean": posterior.mean(0).tolist(),
                "confidence_mean": float(confidence.mean()),
                "reflection_min": float(reflection.min()),
                "reflection_mean": float(reflection.mean()),
                "reflection_max": float(reflection.max()),
                "boundary_mean": float(boundary.mean()),
                "relation_edges": edge_index.shape[1],
                "posterior_sum_max_error": float((posterior.sum(1) - 1).abs().max()),
            },
        })
        print(json.dumps({"stage": "write", "progress": f"{position}/{len(raw_samples)}", "sample": sample_id}), flush=True)
    final_occupancy = occupancy_sum / token_total
    if bool((final_occupancy < 0.05).any()) or bool((final_occupancy > 0.45).any()):
        raise RuntimeError(f"Final cluster occupancy failed: {final_occupancy.tolist()}")

    scalar_stats = {
        name: {"q02": float(scalar_low[index]), "q98": float(scalar_high[index])}
        for index, name in enumerate(SCALAR_NAMES)
    }
    manifest = {
        "schema_version": 1,
        "complete": True,
        "cache_version": CACHE_VERSION,
        "created_at_utc": utc_now(),
        "project_git_commit": current_git_commit(),
        "source_dataset": {
            "root": str(data_root),
            "version": dataset_manifest["version"],
            "manifest": str(data_root / "manifest.json"),
            "manifest_sha256": sha256(data_root / "manifest.json"),
            "split": "train",
            "sample_count": len(train_ids),
            "train_ids": train_ids,
            "validation_and_test_used_for_fit": False,
        },
        "teacher": {
            "source_cache": str(source_cache),
            "source_manifest_sha256": sha256(source_cache / "manifest.json"),
            "block_one_based": BLOCK_NUMBER,
            "block_zero_based_index": BLOCK_INDEX,
            "flow_timestep": FLOW_TIMESTEP,
            "hidden_size": HIDDEN_SIZE,
            "vae_encoding": "posterior mode; deterministic",
            "adapters": "all LoRA disabled",
            "teacher_policy": "fixed base Qwen; never rebuilt from student LoRA",
        },
        "representation": {
            "content": "C=Q20(GT)",
            "input_residual": "E_I=Q20(I)-C",
            "p90_residual": "E_90=Q20(P90)-C",
            "absolute_xy_in_features": False,
            "pca_dimension": PCA_DIM,
            "pca_fit_sample_limit": PCA_SAMPLE_LIMIT,
            "scalar_names": list(SCALAR_NAMES),
            "scalar_robust_stats": scalar_stats,
            "content_group_scale": 0.25,
            "residual_group_scale": 1.0,
        },
        "clustering": {
            "kind": "K=4 balanced Sinkhorn soft prototypes with content-guided spatial smoothing",
            "clusters": CLUSTERS,
            "seeds": list(SEEDS),
            "minimum_seed_stability": minimum_stability,
            "capacity_gate": [0.05, 0.45],
            "final_occupancy": final_occupancy.tolist(),
            "hard_labels_used_for_training": False,
        },
        "relations": {
            "target": "sigmoid((cos(C_i,C_j)-per-image-median)/0.07)",
            "local_offsets": [[0, 1], [1, 0], [1, 1], [1, -1], [0, 2], [2, 0]],
            "nonlocal_pairs_per_token": "top2 and bottom2 by 32D content projection",
            "global_n_squared_matrix_stored": False,
        },
        "reflection_evidence": "robust_unit(sqrt(d_I*d_90)*max(cos(E_I,E_90),0)*(0.7+0.3*DoLP))",
        "boundary": "robust_unit(0.5*posterior_entropy + 0.5*gradient(reflection_evidence))",
        "storage": {
            "sample_dtype": "float16 except int32 indices/grid",
            "scratch_policy": "delete only after independent cache audit passes",
        },
        "memory": {
            "extractor_batch_size": 1,
            "parallel_shards": len({record["shard_index"] for record in extraction_records.values()}),
            "max_peak_allocated_gib": max(record["memory_gib"]["peak_allocated"] for record in extraction_records.values()),
            "max_peak_reserved_gib": max(record["memory_gib"]["peak_reserved"] for record in extraction_records.values()),
        },
        "aspect_bucket_counts": dict(Counter(records[sample_id]["aspect_bucket"] for sample_id in train_ids)),
        "samples": final_records,
    }
    atomic_json(complete_path, manifest)
    atomic_json(output / "audit/cluster_occupancy.json", {
        "token_total": token_total,
        "final_occupancy": final_occupancy.tolist(),
        "minimum": float(final_occupancy.min()),
        "maximum": float(final_occupancy.max()),
    })
    print(json.dumps({
        "status": "complete",
        "samples": len(final_records),
        "tokens": token_total,
        "occupancy": final_occupancy.tolist(),
        "minimum_seed_stability": minimum_stability,
        "output": str(output),
    }, indent=2), flush=True)


def audit(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    source_cache = args.source_cache.resolve()
    output = args.output.resolve()
    _, _, train_ids, _, _ = load_context(data_root, source_cache)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("cache_version") != CACHE_VERSION:
        raise RuntimeError("M3 manifest is incomplete or incompatible")
    if manifest["source_dataset"]["train_ids"] != train_ids or len(manifest["samples"]) != 144:
        raise RuntimeError("M3 cache does not cover the exact train split")
    if manifest["source_dataset"]["validation_and_test_used_for_fit"]:
        raise RuntimeError("Validation/test leakage flag is set")
    if manifest["representation"]["absolute_xy_in_features"]:
        raise RuntimeError("Absolute coordinates must not enter M3 clustering")
    if manifest["clustering"]["minimum_seed_stability"] < 0.80:
        raise RuntimeError("Prototype stability is below the design gate")

    occupancy = torch.zeros(CLUSTERS, dtype=torch.float64)
    token_total = 0
    max_sum_error = 0.0
    max_index = -1
    min_reflection = 1.0
    max_reflection = 0.0
    max_source_hash_errors = 0
    bucket_counts = Counter()
    for record in manifest["samples"]:
        sample_id = record["id"]
        paths = sample_paths(data_root, sample_id)
        for role, path in paths.items():
            if sha256(path) != record["source"][role]["sha256"]:
                max_source_hash_errors += 1
        sample_path = output / record["cache"]["sample"]
        if sha256(sample_path) != record["cache"]["sample_sha256"]:
            raise RuntimeError(f"M3 sample hash mismatch: {sample_id}")
        stored = safetensors.torch.load_file(sample_path)
        posterior = stored["posterior"].float().t()
        confidence = stored["confidence"].float()
        reflection = stored["reflection_evidence"].float()
        boundary = stored["boundary"].float()
        edge_index = stored["edge_index"].long()
        edge_target = stored["edge_target"].float()
        edge_confidence = stored["edge_confidence"].float()
        grid = tuple(int(value) for value in stored["token_grid_hw"])
        tokens = grid[0] * grid[1]
        if posterior.shape != (tokens, CLUSTERS):
            raise ValueError(f"Posterior shape mismatch: {sample_id}/{posterior.shape}")
        tensors = (posterior, confidence, reflection, boundary, edge_target, edge_confidence)
        if not all(torch.isfinite(value).all() for value in tensors):
            raise RuntimeError(f"Non-finite compact cache: {sample_id}")
        error = float((posterior.sum(1) - 1).abs().max())
        max_sum_error = max(max_sum_error, error)
        if error > 2e-3:
            raise RuntimeError(f"Posterior sum failed after FP16 storage: {sample_id}/{error}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] != edge_target.numel():
            raise ValueError(f"Relation edge shape mismatch: {sample_id}")
        if edge_index.numel():
            max_index = max(max_index, int(edge_index.max()))
            if int(edge_index.min()) < 0 or int(edge_index.max()) >= tokens:
                raise ValueError(f"Relation edge out of range: {sample_id}")
        if not all(bool(((value >= -1e-4) & (value <= 1.0001)).all()) for value in tensors[:4]):
            raise ValueError(f"Probability/evidence range failed: {sample_id}")
        occupancy += posterior.double().sum(0)
        token_total += tokens
        min_reflection = min(min_reflection, float(reflection.min()))
        max_reflection = max(max_reflection, float(reflection.max()))
        bucket_counts[record["aspect_bucket"]] += 1
    if max_source_hash_errors:
        raise RuntimeError(f"Source hash errors: {max_source_hash_errors}")
    occupancy /= token_total
    if bool((occupancy < 0.05).any()) or bool((occupancy > 0.45).any()):
        raise RuntimeError(f"Audited cluster occupancy failed: {occupancy.tolist()}")
    result = {
        "status": "passed",
        "cache_version": CACHE_VERSION,
        "samples": len(manifest["samples"]),
        "tokens": token_total,
        "cluster_occupancy": occupancy.tolist(),
        "minimum_seed_stability": manifest["clustering"]["minimum_seed_stability"],
        "max_posterior_sum_error_after_fp16": max_sum_error,
        "max_relation_index_seen": max_index,
        "reflection_range": [min_reflection, max_reflection],
        "source_hash_errors": max_source_hash_errors,
        "aspect_bucket_counts": dict(bucket_counts),
        "scratch_present_before": (output / "scratch").exists(),
        "scratch_removed_this_run": False,
        "scratch_present_after": (output / "scratch").exists(),
        "checked_at_utc": utc_now(),
    }
    if args.cleanup_scratch:
        scratch = output / "scratch"
        resolved = scratch.resolve()
        if resolved != (output.resolve() / "scratch") or output.resolve() not in resolved.parents:
            raise RuntimeError("Unsafe scratch path")
        if scratch.exists():
            shutil.rmtree(scratch)
            result["scratch_removed_this_run"] = True
        result["scratch_present_after"] = scratch.exists()
    atomic_json(output / "audit/cache_check.json", result)
    print(json.dumps(result, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--device", default="cuda:0")
    extract_parser.add_argument("--shard-index", type=int, required=True)
    extract_parser.add_argument("--num-shards", type=int, required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--device", default="cuda:0")
    audit_parser = subparsers.add_parser("check")
    audit_parser.add_argument("--cleanup-scratch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "extract":
        extract(args)
    elif args.command == "finalize":
        finalize(args)
    elif args.command == "check":
        audit(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()

