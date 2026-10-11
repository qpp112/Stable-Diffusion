"""GPU memory management: offloading strategies and peak-memory measurement.

Offload modes (weights live in host RAM and visit the GPU only when needed):

* ``none``       - everything stays on the GPU (fastest).
* ``model``      - one *component* on the GPU at a time: text encoders -> UNet -> VAE.
                   Peak weight memory = the largest component (SDXL UNet, ~5.1 GB fp16)
                   instead of the sum (~7 GB). Small slowdown (3 host->GPU copies / image).
* ``sequential`` - additionally streams the UNet *block by block* in every denoising
                   step. Peak weight memory drops to the largest block (~0.8 GB for
                   SDXL) at the cost of copying the whole UNet over PCIe every step.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List

import torch
import torch.nn as nn

OFFLOAD_MODES = ("none", "model", "sequential")


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


class Offloader:
    def __init__(self, device: torch.device, mode: str = "none"):
        if mode not in OFFLOAD_MODES:
            raise ValueError(f"offload mode must be one of {OFFLOAD_MODES}")
        self.device = device
        self.mode = mode
        self._hooks: List = []
        self._streamed: List[nn.Module] = []

    def place(self, modules: List[nn.Module], unet: nn.Module) -> None:
        """Initial placement for the chosen mode."""
        self.remove_hooks()
        if self.mode == "none":
            for m in modules:
                m.to(self.device)
            return
        for m in modules:
            m.to("cpu")
        empty_cache(self.device)
        if self.mode == "sequential":
            self._add_block_hooks(unet)

    @contextlib.contextmanager
    def use(self, *modules: nn.Module) -> Iterator[None]:
        """Make ``modules`` resident on the GPU for the duration of the block."""
        moved = []
        if self.mode != "none":
            for m in modules:
                if not getattr(m, "_sd_sequential", False):
                    m.to(self.device)
                    moved.append(m)
        try:
            yield
        finally:
            for m in moved:
                m.to("cpu")
            if moved:
                empty_cache(self.device)

    # -- block-level streaming for the UNet
    def _add_block_hooks(self, unet: nn.Module) -> None:
        blocks = [unet.time_embed, *unet.input_blocks, unet.middle_block, *unet.output_blocks, unet.out]
        if hasattr(unet, "label_emb"):
            blocks.insert(1, unet.label_emb)
        dev = self.device

        def to_gpu(module, args):       # forward pre-hook: must return None
            module.to(dev)

        def to_cpu(module, args, out):  # forward hook: must return None
            module.to("cpu")

        for b in blocks:
            self._hooks.append(b.register_forward_pre_hook(to_gpu))
            self._hooks.append(b.register_forward_hook(to_cpu))
        unet._sd_sequential = True
        self._streamed.append(unet)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for m in self._streamed:
            m._sd_sequential = False
        self._streamed.clear()


@dataclass
class StageStats:
    seconds: float = 0.0
    peak_bytes: int = 0


@dataclass
class MemoryReport:
    stages: Dict[str, StageStats] = field(default_factory=dict)

    @property
    def peak_gb(self) -> float:
        return max((s.peak_bytes for s in self.stages.values()), default=0) / 2**30

    def summary(self) -> str:
        parts = [f"{k}: {v.seconds:.1f}s / {v.peak_bytes / 2**30:.2f} GB" for k, v in self.stages.items()]
        return " | ".join(parts) + f" || peak {self.peak_gb:.2f} GB"


@contextlib.contextmanager
def track(report: MemoryReport, name: str, device: torch.device) -> Iterator[None]:
    """Record wall time and peak allocated CUDA memory of a pipeline stage."""
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if cuda:
            torch.cuda.synchronize(device)
        stats = StageStats(seconds=time.perf_counter() - t0,
                           peak_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0)
        report.stages[name] = stats
