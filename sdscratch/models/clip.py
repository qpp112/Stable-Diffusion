"""CLIP text encoder (causal Transformer), written from scratch.

Parameter names mirror Hugging Face's ``CLIPTextModel`` / ``CLIPTextModelWithProjection``
so original checkpoints load with a plain ``load_state_dict`` (the OpenCLIP ViT-bigG
encoder used by SDXL is converted to this layout in :mod:`sdscratch.loader`).

Used configurations:
  * SD 1.5          - OpenAI CLIP ViT-L/14 text tower, last layer (after final LayerNorm).
  * SDXL encoder 1  - same ViT-L/14, *penultimate* layer, no final LayerNorm.
  * SDXL encoder 2  - OpenCLIP ViT-bigG/14, penultimate layer + projected pooled
                      embedding (final LayerNorm -> EOS token -> text_projection).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CLIPTextConfig:
    vocab_size: int = 49408
    max_len: int = 77
    dim: int = 768
    layers: int = 12
    heads: int = 12
    mlp_dim: int = 3072
    act: str = "quick_gelu"          # OpenAI CLIP; OpenCLIP bigG uses exact "gelu"
    proj_dim: Optional[int] = None   # text_projection output size (None = no projection)
    pad_id: int = 49407              # token used to pad prompts to max_len
    eps: float = 1e-5


CLIP_VIT_L = CLIPTextConfig()
CLIP_VIT_BIGG = CLIPTextConfig(dim=1280, layers=32, heads=20, mlp_dim=5120, act="gelu",
                               proj_dim=1280, pad_id=0)


@dataclass
class CLIPTextOutput:
    last_hidden_state: torch.Tensor   # (B, L, D) after final LayerNorm
    penultimate: torch.Tensor         # (B, L, D) output of layer N-1, no final LayerNorm
    pooled: torch.Tensor              # (B, D or proj_dim) EOS token of last_hidden_state (+ projection)


class CLIPEmbeddings(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.position_embedding = nn.Embedding(cfg.max_len, cfg.dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(tokens) + self.position_embedding.weight[: tokens.shape[1]]


class CLIPAttention(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.heads = cfg.heads
        self.q_proj = nn.Linear(cfg.dim, cfg.dim)
        self.k_proj = nn.Linear(cfg.dim, cfg.dim)
        self.v_proj = nn.Linear(cfg.dim, cfg.dim)
        self.out_proj = nn.Linear(cfg.dim, cfg.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        split = lambda t: t.view(b, n, self.heads, c // self.heads).transpose(1, 2)  # noqa: E731
        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        # 77 tokens: memory is negligible, so always use the fused kernel with a causal mask.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).reshape(b, n, c))


class CLIPMLP(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.mlp_dim)
        self.fc2 = nn.Linear(cfg.mlp_dim, cfg.dim)
        self.act = cfg.act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = x * torch.sigmoid(1.702 * x) if self.act == "quick_gelu" else F.gelu(x)
        return self.fc2(x)


class CLIPEncoderLayer(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.dim, eps=cfg.eps)
        self.self_attn = CLIPAttention(cfg)
        self.layer_norm2 = nn.LayerNorm(cfg.dim, eps=cfg.eps)
        self.mlp = CLIPMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pre-LN residual block
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class CLIPEncoder(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.layers = nn.ModuleList([CLIPEncoderLayer(cfg) for _ in range(cfg.layers)])


class CLIPTextTransformer(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.embeddings = CLIPEmbeddings(cfg)
        self.encoder = CLIPEncoder(cfg)
        self.final_layer_norm = nn.LayerNorm(cfg.dim, eps=cfg.eps)


class CLIPTextModel(nn.Module):
    def __init__(self, cfg: CLIPTextConfig = CLIP_VIT_L):
        super().__init__()
        self.cfg = cfg
        self.text_model = CLIPTextTransformer(cfg)
        if cfg.proj_dim is not None:
            self.text_projection = nn.Linear(cfg.dim, cfg.proj_dim, bias=False)

    def forward(self, tokens: torch.Tensor) -> CLIPTextOutput:
        tm = self.text_model
        x = tm.embeddings(tokens)
        layers = tm.encoder.layers
        penultimate = x
        for i, layer in enumerate(layers):
            if i == len(layers) - 1:
                penultimate = x
            x = layer(x)
        last = tm.final_layer_norm(x)
        # The end-of-text token has the largest id (49407), so argmax finds the first EOS.
        pooled = last[torch.arange(last.shape[0], device=last.device), tokens.argmax(dim=-1)]
        if self.cfg.proj_dim is not None:
            pooled = self.text_projection(pooled)
        return CLIPTextOutput(last_hidden_state=last, penultimate=penultimate, pooled=pooled)
