"""Pinned local WindowSeat/Qwen backend for M1 parity and single-branch probes.

The upstream implementation supplies VAE normalization, latent packing and the
one-step flow update. No Hugging Face download is attempted by this module.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import safetensors.torch
import torch
from peft import LoraConfig

from .backend import BRANCHES, BranchBackend


ROOT = Path("/share/linmingheng-local/xuke")
REPO = ROOT / "projects/windowseat-reflection-removal"
MANIFEST = ROOT / "configs/windowseat_download_manifest.json"
ADAPTER_NAMES = {"transmission": "default", "reflection": "reflection"}


def _snapshot(repo: str, revision: str) -> Path:
    return ROOT / ".cache/huggingface/hub" / (
        "models--" + repo.replace("/", "--")
    ) / "snapshots" / revision


def load_upstream():
    module_name = "rmagnet_pinned_windowseat_inference"
    if module_name in sys.modules:
        return sys.modules[module_name]
    source = REPO / "windowseat_inference.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pinned WindowSeat source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def check_snapshots() -> tuple[Path, Path]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    commit = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != manifest["repository_commit"]:
        raise RuntimeError(f"WindowSeat commit mismatch: {commit}")
    folders = []
    for model in manifest["models"]:
        folder = _snapshot(model["repo"], model["revision"])
        for entry in model["files"]:
            file = folder / entry["path"]
            if not file.is_file() or file.stat().st_size != entry["size_bytes"]:
                raise RuntimeError(f"Missing or wrong local model file: {file}")
        folders.append(folder)
    return folders[0], folders[1]


class QwenSharedBackend(BranchBackend):
    """One NF4 Qwen DiT with distinct named LoRA adapters.

    The reflection adapter is untrained and its image has no reflection meaning.
    M1 only establishes routing, parity of the T adapter, and single-branch grads.
    """

    def __init__(self, vae, transformer, embeddings, resolution: int, upstream):
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.embeddings = embeddings
        self.resolution = resolution
        self.upstream = upstream
        self._trainable_branch: str | None = None
        self.vae.eval()
        for parameter in self.vae.parameters():
            parameter.requires_grad_(False)
        self.activate("transmission")

    @classmethod
    def from_local(cls, device: torch.device, reflection_rank: int = 8):
        base, lora = check_snapshots()
        upstream = load_upstream()
        vae = upstream.load_qwen_vae(str(base), device)
        transformer = upstream.load_qwen_transformer(str(base), device)

        config = LoraConfig.from_pretrained(str(lora), subfolder="transformer_lora")
        transformer.add_adapter(config, adapter_name=ADAPTER_NAMES["transmission"])
        state = safetensors.torch.load_file(
            str(lora / "transformer_lora/pytorch_lora_weights.safetensors")
        )
        _, unexpected = transformer.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected T LoRA weights: {unexpected[:5]}")
        embeddings = safetensors.torch.load_file(
            str(lora / "text_embeddings/state_dict.safetensors")
        )
        resolution = json.loads(
            (lora / "model_index.json").read_text(encoding="utf-8")
        )["processing_resolution"]

        # LoRA initialization consumes RNG. Restore it so the T prediction uses
        # the same stochastic VAE latent sequence as the original T-only runner.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(device)
        r_config = copy.deepcopy(config)
        r_config.r = reflection_rank
        r_config.lora_alpha = reflection_rank
        transformer.add_adapter(r_config, adapter_name=ADAPTER_NAMES["reflection"])
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        return cls(vae, transformer, embeddings, resolution, upstream)

    def activate(self, branch: str) -> None:
        if branch not in BRANCHES:
            raise ValueError(f"Unknown branch: {branch}")
        name = ADAPTER_NAMES[branch]
        self.transformer.set_adapter(name)
        for parameter_name, parameter in self.transformer.named_parameters():
            selected = (
                self._trainable_branch == branch
                and ".lora_" in parameter_name
                and f".{name}." in parameter_name
            )
            parameter.requires_grad_(selected)

    def set_trainable_branch(self, branch: str | None) -> None:
        if branch is not None and branch not in BRANCHES:
            raise ValueError(f"Unknown branch: {branch}")
        self._trainable_branch = branch
        self.activate(branch or "transmission")

    def trainable_summary(self) -> dict[str, int]:
        return {
            "total": sum(p.numel() for p in self.transformer.parameters()),
            "trainable": sum(
                p.numel() for p in self.transformer.parameters() if p.requires_grad
            ),
        }

    def forward_normalized(self, image: torch.Tensor, branch: str) -> torch.Tensor:
        """Take RGB [-1,1], return RGB [-1,1] with upstream flow conventions."""
        self.activate(branch)
        with torch.no_grad():
            latent = self.upstream.encode(image, self.vae)
        edited = self.upstream.flow_step(
            latent, self.transformer, self.vae, self.embeddings
        )
        return self.upstream.decode(edited, self.vae)

    def forward(self, image: torch.Tensor, branch: str) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected [B,3,H,W] RGB input")
        result = self.forward_normalized(image * 2 - 1, branch)
        return ((result + 1) / 2).clamp(0, 1)
