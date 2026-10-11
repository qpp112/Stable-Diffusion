"""KL-regularised autoencoder ("VAE") from Latent Diffusion (Rombach et al., 2022).

Maps 3xHxW images in [-1, 1] to 4x(H/8)x(W/8) latents. Parameter names follow the original
``ldm`` / ``sgm`` code (``encoder.down.0.block.0.conv1``...) so the ``first_stage_model.*``
weights of an SD 1.5 / SDXL single-file checkpoint load directly.

Memory: decoding is the single most memory-hungry step at 1024x1024 - the last decoder
stage holds several 128x1024x1024 fp32 activations (~0.5 GB each) and the mid-block
self-attention runs over 128x128 = 16,384 tokens. :meth:`AutoencoderKL.decode_tiled`
decodes overlapping latent tiles and feather-blends them, bounding peak memory by
the tile size instead of the image size.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import attention


@dataclass(frozen=True)
class VAEConfig:
    in_channels: int = 3
    out_channels: int = 3
    ch: int = 128
    ch_mult: tuple = (1, 2, 4, 4)
    num_res_blocks: int = 2
    z_channels: int = 4
    scaling_factor: float = 0.18215   # SD 1.5; SDXL uses 0.13025

    @property
    def downsample_factor(self) -> int:
        return 2 ** (len(self.ch_mult) - 1)


SD15_VAE = VAEConfig()
SDXL_VAE = VAEConfig(scaling_factor=0.13025)


def Normalize(c: int) -> nn.GroupNorm:  # noqa: N802 - name kept from the original code
    return nn.GroupNorm(32, c, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.norm1 = Normalize(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = Normalize(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        if cin != cout:
            self.nin_shortcut = nn.Conv2d(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if hasattr(self, "nin_shortcut"):
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    """Single-head self-attention over all spatial positions (1x1 convs for q, k, v)."""

    def __init__(self, c: int):
        super().__init__()
        self.norm = Normalize(c)
        self.q = nn.Conv2d(c, c, 1)
        self.k = nn.Conv2d(c, c, 1)
        self.v = nn.Conv2d(c, c, 1)
        self.proj_out = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x)
        tok = lambda t: t.reshape(b, c, h * w).transpose(1, 2)  # noqa: E731  (B, HW, C)
        out = attention(tok(self.q(y)), tok(self.k(y)), tok(self.v(y)), heads=1)
        return x + self.proj_out(out.transpose(1, 2).reshape(b, c, h, w))


class Downsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1)))  # asymmetric padding, as in the original


class Upsample(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class _Mid(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.block_1 = ResnetBlock(c, c)
        self.attn_1 = AttnBlock(c)
        self.block_2 = ResnetBlock(c, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block_2(self.attn_1(self.block_1(x)))


class _Level(nn.Module):
    """One resolution level: ``block`` (ModuleList) + optional ``downsample``/``upsample``."""


class Encoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.conv_in = nn.Conv2d(cfg.in_channels, cfg.ch, 3, padding=1)
        in_mult = (1,) + tuple(cfg.ch_mult)
        self.down = nn.ModuleList()
        for i, mult in enumerate(cfg.ch_mult):
            level = _Level()
            cin, cout = cfg.ch * in_mult[i], cfg.ch * mult
            level.block = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                level.block.append(ResnetBlock(cin, cout))
                cin = cout
            if i != len(cfg.ch_mult) - 1:
                level.downsample = Downsample(cout)
            self.down.append(level)
        c = cfg.ch * cfg.ch_mult[-1]
        self.mid = _Mid(c)
        self.norm_out = Normalize(c)
        self.conv_out = nn.Conv2d(c, 2 * cfg.z_channels, 3, padding=1)   # mean and log-variance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for level in self.down:
            for blk in level.block:
                h = blk(h)
            if hasattr(level, "downsample"):
                h = level.downsample(h)
        h = self.mid(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        c = cfg.ch * cfg.ch_mult[-1]
        self.conv_in = nn.Conv2d(cfg.z_channels, c, 3, padding=1)
        self.mid = _Mid(c)
        levels = []
        for i in reversed(range(len(cfg.ch_mult))):
            level = _Level()
            cout = cfg.ch * cfg.ch_mult[i]
            level.block = nn.ModuleList()
            for _ in range(cfg.num_res_blocks + 1):
                level.block.append(ResnetBlock(c, cout))
                c = cout
            if i != 0:
                level.upsample = Upsample(c)
            levels.insert(0, level)          # up[0] is the highest resolution, as in ldm
        self.up = nn.ModuleList(levels)
        self.norm_out = Normalize(c)
        self.conv_out = nn.Conv2d(c, cfg.out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.mid(self.conv_in(z))
        for level in reversed(self.up):
            for blk in level.block:
                h = blk(h)
            if hasattr(level, "upsample"):
                h = level.upsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))


def _feather(n: int, ramp: int, start: bool, end: bool, device) -> torch.Tensor:
    """1-D blending weights: linear ramps on sides that overlap a neighbouring tile."""
    w = torch.ones(n, device=device)
    ramp = min(ramp, n // 2)
    if ramp > 0:
        r = torch.arange(1, ramp + 1, device=device, dtype=torch.float32) / (ramp + 1)
        if start:
            w[:ramp] = r
        if end:
            w[n - ramp:] = r.flip(0)
    return w


class AutoencoderKL(nn.Module):
    def __init__(self, cfg: VAEConfig = SD15_VAE):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.quant_conv = nn.Conv2d(2 * cfg.z_channels, 2 * cfg.z_channels, 1)
        self.post_quant_conv = nn.Conv2d(cfg.z_channels, cfg.z_channels, 1)

    def encode(self, x: torch.Tensor, sample: bool = False, generator=None) -> torch.Tensor:
        """Image in [-1, 1] -> *unscaled* latent (multiply by ``cfg.scaling_factor`` for the UNet)."""
        mean, logvar = self.quant_conv(self.encoder(x)).chunk(2, dim=1)
        if not sample:
            return mean
        std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
        return mean + std * torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """*Unscaled* latent -> image in roughly [-1, 1]."""
        return self.decoder(self.post_quant_conv(z))

    @torch.no_grad()
    def decode_tiled(self, z: torch.Tensor, tile: int = 64, overlap: int = 16) -> torch.Tensor:
        """Decode overlapping ``tile x tile`` latent tiles and feather-blend them.

        GroupNorm statistics are computed per tile, so tiles differ slightly at their
        borders; the linear cross-fade over ``overlap`` latent pixels hides the seams.
        """
        b, _, h, w = z.shape
        if h <= tile and w <= tile:
            return self.decode(z)
        f = self.cfg.downsample_factor
        stride = tile - overlap
        if stride <= 0:
            raise ValueError("overlap must be smaller than tile")

        def starts(n):
            s = list(range(0, max(n - tile, 0) + 1, stride))
            if s[-1] + tile < n:
                s.append(n - tile)
            return s

        out = torch.zeros(b, self.cfg.out_channels, h * f, w * f, device=z.device, dtype=torch.float32)
        acc = torch.zeros(1, 1, h * f, w * f, device=z.device, dtype=torch.float32)
        for y0 in starts(h):
            for x0 in starts(w):
                dec = self.decode(z[:, :, y0:y0 + tile, x0:x0 + tile]).float()
                th, tw = dec.shape[-2:]
                wy = _feather(th, overlap * f, y0 > 0, y0 + tile < h, z.device)
                wx = _feather(tw, overlap * f, x0 > 0, x0 + tile < w, z.device)
                wgt = (wy[:, None] * wx[None, :])[None, None]
                ys, xs = slice(y0 * f, y0 * f + th), slice(x0 * f, x0 * f + tw)
                out[:, :, ys, xs] += dec * wgt
                acc[:, :, ys, xs] += wgt
                del dec
        return (out / acc).to(z.dtype)
