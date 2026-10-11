"""Model specifications: architecture configs + where to fetch the official weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .models.clip import CLIP_VIT_BIGG, CLIP_VIT_L, CLIPTextConfig
from .models.unet import SD15_UNET, SDXL_UNET, UNetConfig
from .models.vae import SD15_VAE, SDXL_VAE, VAEConfig


@dataclass(frozen=True)
class ModelSpec:
    name: str
    unet: UNetConfig
    vae: VAEConfig
    text_encoders: Tuple[CLIPTextConfig, ...]
    # checkpoint key prefixes of each text encoder inside the single-file checkpoint
    text_encoder_prefixes: Tuple[str, ...]
    text_encoder_formats: Tuple[str, ...]          # "hf" or "openclip"
    repo_id: str
    filename: str
    default_size: int
    default_guidance: float
    prediction_type: str = "epsilon"
    tokenizer_files: Tuple[str, str] = ("tokenizer/vocab.json", "tokenizer/merges.txt")

    @property
    def is_sdxl(self) -> bool:
        return self.unet.adm_in_channels is not None

    @property
    def latent_multiple(self) -> int:
        """Image side lengths must be divisible by this (VAE /8, then UNet downsamplings)."""
        return self.vae.downsample_factor * 2 ** (len(self.unet.channel_mult) - 1)


SD15 = ModelSpec(
    name="sd15",
    unet=SD15_UNET,
    vae=SD15_VAE,
    text_encoders=(CLIP_VIT_L,),
    text_encoder_prefixes=("cond_stage_model.transformer.",),
    text_encoder_formats=("hf",),
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    filename="v1-5-pruned-emaonly.safetensors",
    default_size=512,
    default_guidance=7.5,
)

SDXL = ModelSpec(
    name="sdxl",
    unet=SDXL_UNET,
    vae=SDXL_VAE,
    text_encoders=(CLIP_VIT_L, CLIP_VIT_BIGG),
    text_encoder_prefixes=("conditioner.embedders.0.transformer.", "conditioner.embedders.1.model."),
    text_encoder_formats=("hf", "openclip"),
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    filename="sd_xl_base_1.0.safetensors",
    default_size=1024,
    default_guidance=6.0,
)

SPECS = {"sd15": SD15, "sdxl": SDXL}
