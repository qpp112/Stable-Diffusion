"""Multi-head scaled dot-product attention with swappable backends.

Attention is the memory bottleneck at high resolution. For a 1024x1024 image the SD 1.5
UNet's first attention level sees a 128x128 latent = 16,384 tokens, so a single
materialised attention matrix is

    batch(2, CFG) x heads(8) x 16384 x 16384 x 2 bytes (fp16)  ~= 8.6 GB.

Backends (selected globally with :func:`set_attention_backend`):

* ``naive``  - textbook softmax(QK^T / sqrt(d)) V. Materialises the full matrix.
* ``sliced`` - processes ``slice_size`` query tokens at a time, so peak memory is
               O(slice_size x M) instead of O(N x M). Same math, a bit slower.
* ``sdpa``   - ``torch.nn.functional.scaled_dot_product_attention``, which dispatches
               to FlashAttention / memory-efficient kernels that never materialise
               the matrix. Default.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

BACKENDS = ("sdpa", "sliced", "naive")


@dataclass
class AttentionConfig:
    backend: str = "sdpa"
    slice_size: int = 1024


_CFG = AttentionConfig()


def set_attention_backend(backend: str = "sdpa", slice_size: int | None = None) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    _CFG.backend = backend
    if slice_size is not None:
        _CFG.slice_size = int(slice_size)


def get_attention_config() -> AttentionConfig:
    return _CFG


def _naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    w = torch.matmul(q * scale, k.transpose(-1, -2))   # (B, H, N, M)  <- the big one
    return torch.matmul(w.softmax(dim=-1), v)


def _sliced(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, size: int) -> torch.Tensor:
    out = torch.empty_like(q)
    for i in range(0, q.shape[2], size):
        out[:, :, i:i + size] = _naive(q[:, :, i:i + size], k, v)
    return out


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
    """q: (B, N, H*D), k/v: (B, M, H*D)  ->  (B, N, H*D)."""
    b, n, c = q.shape
    d = c // heads
    # reshape (not view): inputs may be non-contiguous, e.g. the VAE's transposed feature maps
    q = q.reshape(b, n, heads, d).transpose(1, 2)
    k = k.reshape(b, -1, heads, d).transpose(1, 2)
    v = v.reshape(b, -1, heads, d).transpose(1, 2)
    if _CFG.backend == "sdpa":
        out = F.scaled_dot_product_attention(q, k, v)
    elif _CFG.backend == "sliced":
        out = _sliced(q, k, v, _CFG.slice_size)
    else:
        out = _naive(q, k, v)
    return out.transpose(1, 2).reshape(b, n, c)
