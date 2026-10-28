"""Conditional UNet (epsilon-prediction) used by SD 1.5 and SDXL.

Structure (Rombach et al., 2022; Podell et al., 2023):

    input_blocks:  conv_in, then per resolution level [ResBlock (+ SpatialTransformer)] x 2
                   followed by a strided-conv Downsample (except at the last level)
    middle_block:  ResBlock, SpatialTransformer, ResBlock
    output_blocks: mirror of input_blocks; each block concatenates the matching skip
                   activation, and the last block of a level ends with an Upsample

The time step enters every ResBlock through a learned projection of a sinusoidal
embedding; text enters every SpatialTransformer through cross-attention. SDXL adds a
"vector" condition y (pooled text embedding + size/crop micro-conditioning) that is
embedded and added to the time embedding.

Module/parameter names match the original ``ldm``/``sgm`` ``openaimodel.UNetModel`` so the
``model.diffusion_model.*`` checkpoint weights load with ``load_state_dict(strict=True)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import attention


@dataclass(frozen=True)
class UNetConfig:
    in_channels: int = 4
    out_channels: int = 4
    model_channels: int = 320
    channel_mult: Tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int, ...] = (4, 2, 1)   # downsampling factors that get attention
    transformer_depth: Tuple[int, ...] = (1, 1, 1, 1)    # per level
    transformer_depth_middle: int = 1
    num_heads: int = 8                                     # used if num_head_channels is None
    num_head_channels: Optional[int] = None
    context_dim: int = 768
    use_linear_in_transformer: bool = False
    adm_in_channels: Optional[int] = None                  # SDXL vector-conditioning width


SD15_UNET = UNetConfig()
SDXL_UNET = UNetConfig(
    channel_mult=(1, 2, 4),
    attention_resolutions=(4, 2),
    transformer_depth=(1, 2, 10),
    transformer_depth_middle=10,
    num_head_channels=64,
    context_dim=2048,
    use_linear_in_transformer=True,
    adm_in_channels=2816,   # 1280 (pooled bigG text) + 6 x 256 (size / crop embeddings)
)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding [cos(t w_i), sin(t w_i)], w_i = max_period^(-i / (dim/2))."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ----------------------------------------------------------------------------- blocks
class ResBlock(nn.Module):
    def __init__(self, channels: int, emb_channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out = out_channels or channels
        self.in_layers = nn.Sequential(nn.GroupNorm(32, channels), nn.SiLU(),
                                       nn.Conv2d(channels, out, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, out))
        self.out_layers = nn.Sequential(nn.GroupNorm(32, out), nn.SiLU(), nn.Dropout(0.0),
                                        nn.Conv2d(out, out, 3, padding=1))
        self.skip_connection = nn.Identity() if out == channels else nn.Conv2d(channels, out, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        h = h + self.emb_layers(emb)[:, :, None, None]
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: Optional[int], heads: int, dim_head: int):
        super().__init__()
        inner = heads * dim_head
        context_dim = context_dim or query_dim
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, query_dim), nn.Dropout(0.0))

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx = x if context is None else context
        return self.to_out(attention(self.to_q(x), self.to_k(ctx), self.to_v(ctx), self.heads))


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(GEGLU(dim, dim * mult), nn.Dropout(0.0), nn.Linear(dim * mult, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    """Pre-LN block: self-attention -> cross-attention to text -> GEGLU feed-forward."""

    def __init__(self, dim: int, heads: int, dim_head: int, context_dim: int):
        super().__init__()
        self.attn1 = CrossAttention(dim, None, heads, dim_head)
        self.ff = FeedForward(dim)
        self.attn2 = CrossAttention(dim, context_dim, heads, dim_head)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        return x + self.ff(self.norm3(x))


class SpatialTransformer(nn.Module):
    """Flattens a feature map to tokens, applies ``depth`` transformer blocks, reshapes back."""

    def __init__(self, channels: int, heads: int, dim_head: int, depth: int, context_dim: int,
                 use_linear: bool):
        super().__init__()
        inner = heads * dim_head
        self.use_linear = use_linear
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.proj_in = nn.Linear(channels, inner) if use_linear else nn.Conv2d(channels, inner, 1)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner, heads, dim_head, context_dim) for _ in range(depth)])
        self.proj_out = nn.Linear(inner, channels) if use_linear else nn.Conv2d(inner, channels, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, -1)      # (B, HW, C) tokens
        if self.use_linear:
            x = self.proj_in(x)
        for blk in self.transformer_blocks:
            x = blk(x, context)
        if self.use_linear:
            x = self.proj_out(x)
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class TimestepEmbedSequential(nn.Sequential):
    """Sequential container that routes the time embedding / text context to the layers that need it."""

    def forward(self, x, emb, context=None):  # type: ignore[override]
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


# ------------------------------------------------------------------------------ model
class UNetModel(nn.Module):
    def __init__(self, cfg: UNetConfig = SD15_UNET):
        super().__init__()
        self.cfg = cfg
        mc = cfg.model_channels
        ted = mc * 4  # time-embedding width

        self.time_embed = nn.Sequential(nn.Linear(mc, ted), nn.SiLU(), nn.Linear(ted, ted))
        if cfg.adm_in_channels is not None:
            self.label_emb = nn.Sequential(nn.Sequential(
                nn.Linear(cfg.adm_in_channels, ted), nn.SiLU(), nn.Linear(ted, ted)))

        def transformer(ch: int, depth: int) -> SpatialTransformer:
            if cfg.num_head_channels is not None:
                heads, dim_head = ch // cfg.num_head_channels, cfg.num_head_channels
            else:
                heads, dim_head = cfg.num_heads, ch // cfg.num_heads
            return SpatialTransformer(ch, heads, dim_head, depth, cfg.context_dim, cfg.use_linear_in_transformer)

        # ---- encoder half
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(cfg.in_channels, mc, 3, padding=1))])
        skip_channels = [mc]
        ch, ds = mc, 1
        for level, mult in enumerate(cfg.channel_mult):
            for _ in range(cfg.num_res_blocks):
                layers: list = [ResBlock(ch, ted, mult * mc)]
                ch = mult * mc
                if ds in cfg.attention_resolutions:
                    layers.append(transformer(ch, cfg.transformer_depth[level]))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                skip_channels.append(ch)
            if level != len(cfg.channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch)))
                skip_channels.append(ch)
                ds *= 2

        # ---- bottleneck
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, ted), transformer(ch, cfg.transformer_depth_middle), ResBlock(ch, ted))

        # ---- decoder half
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(cfg.channel_mult))[::-1]:
            for i in range(cfg.num_res_blocks + 1):
                layers = [ResBlock(ch + skip_channels.pop(), ted, mc * mult)]
                ch = mc * mult
                if ds in cfg.attention_resolutions:
                    layers.append(transformer(ch, cfg.transformer_depth[level]))
                if level and i == cfg.num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(nn.GroupNorm(32, ch), nn.SiLU(), nn.Conv2d(mc, cfg.out_channels, 3, padding=1))

    @property
    def dtype(self) -> torch.dtype:
        return self.time_embed[0].weight.dtype

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor,
                y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, 4, h, w) noisy latent, t: (B,) timesteps, context: (B, 77, context_dim),
        y: (B, adm_in_channels) SDXL vector condition. Returns the predicted noise."""
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        emb = self.time_embed(timestep_embedding(t, self.cfg.model_channels).to(x.dtype))
        if self.cfg.adm_in_channels is not None:
            if y is None:
                raise ValueError("this UNet needs the vector condition y")
            emb = emb + self.label_emb(y.to(x.dtype))

        skips = []
        h = x
        for blk in self.input_blocks:
            h = blk(h, emb, context)
            skips.append(h)
        h = self.middle_block(h, emb, context)
        for blk in self.output_blocks:
            h = blk(torch.cat([h, skips.pop()], dim=1), emb, context)
        return self.out(h)
