"""Load official single-file SD 1.5 / SDXL checkpoints into the from-scratch modules.

* Tensors are read one at a time with ``safetensors.safe_open`` and cast immediately,
  so peak host RAM is ~ the size of the largest component in the target dtype.
* Modules are constructed on the ``meta`` device (no memory, no random init) and the
  checkpoint tensors are *assigned* as parameters (``load_state_dict(assign=True)``).
  Building the SDXL UNet in fp32 first would otherwise need 10 GB of RAM.
* Loading is ``strict``: any missing / unexpected key is an error, so a successful load
  proves the hand-written architecture matches the checkpoint exactly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .configs import SPECS, ModelSpec
from .models.clip import CLIPTextModel
from .models.unet import UNetModel
from .models.vae import AutoencoderKL

StateDict = Dict[str, torch.Tensor]


# ------------------------------------------------------------------------ download
def download(model: str = "sdxl", cache_dir: Optional[str] = None) -> Tuple[str, str, str]:
    """Fetch (checkpoint, vocab.json, merges.txt) from the Hugging Face Hub (cached)."""
    from huggingface_hub import hf_hub_download

    spec = SPECS[model]
    ckpt = hf_hub_download(spec.repo_id, spec.filename, cache_dir=cache_dir)
    vocab = hf_hub_download(spec.repo_id, spec.tokenizer_files[0], cache_dir=cache_dir)
    merges = hf_hub_download(spec.repo_id, spec.tokenizer_files[1], cache_dir=cache_dir)
    return ckpt, vocab, merges


# ---------------------------------------------------------------------------- read
def checkpoint_keys(path: str) -> List[str]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        return list(f.keys())


def detect_model_type(keys: List[str]) -> str:
    if any(k.startswith("conditioner.embedders.1.") for k in keys):
        return "sdxl"
    if any(k.startswith("cond_stage_model.transformer.") for k in keys):
        return "sd15"
    raise ValueError("unrecognised checkpoint: expected an SD 1.5 or SDXL single-file .safetensors")


def read_prefix(path: str, prefix: str, dtype: torch.dtype) -> StateDict:
    """All tensors whose key starts with ``prefix`` (prefix stripped), floats cast to ``dtype``."""
    if not path.endswith(".safetensors"):
        raise ValueError("only .safetensors checkpoints are supported (pickled .ckpt files can run code)")
    from safetensors import safe_open

    out: StateDict = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.startswith(prefix):
                t = f.get_tensor(k)
                out[k[len(prefix):]] = t.to(dtype) if t.is_floating_point() else t
    if not out:
        raise KeyError(f"no tensors with prefix {prefix!r} in {path}")
    return out


# ------------------------------------------------------------------------- convert
def convert_vae(sd: StateDict) -> StateDict:
    """Some re-saved checkpoints store the VAE mid-attention q/k/v/proj_out as Linear (2-D)
    weights; our AttnBlock uses 1x1 convs (4-D). Training-only discriminator/loss weights are dropped."""
    sd = {k: v for k, v in sd.items() if not k.startswith("loss.")}
    for k in list(sd):
        if ".attn_1." in k and k.endswith(".weight") and sd[k].ndim == 2:
            sd[k] = sd[k][:, :, None, None]
    return sd


def convert_hf_clip(sd: StateDict) -> StateDict:
    sd = {k: v for k, v in sd.items() if not k.endswith("position_ids")}  # non-parameter buffer
    if not any(k.startswith("text_model.") for k in sd):                   # very old HF layout
        sd = {("text_model." + k if not k.startswith("text_projection") else k): v for k, v in sd.items()}
    return sd


def convert_openclip(sd: StateDict) -> StateDict:
    """OpenCLIP text tower (``resblocks``, fused ``in_proj``) -> HF ``CLIPTextModel`` layout."""
    out: StateDict = {
        "text_model.embeddings.token_embedding.weight": sd.pop("token_embedding.weight"),
        "text_model.embeddings.position_embedding.weight": sd.pop("positional_embedding"),
        "text_model.final_layer_norm.weight": sd.pop("ln_final.weight"),
        "text_model.final_layer_norm.bias": sd.pop("ln_final.bias"),
        # OpenCLIP computes  pooled @ P  with P: (width, embed_dim); nn.Linear computes x @ W^T.
        "text_projection.weight": sd.pop("text_projection").t().contiguous(),
    }
    sd.pop("logit_scale", None)
    n_layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("transformer.resblocks."))
    rename = {"ln_1": "layer_norm1", "ln_2": "layer_norm2", "mlp.c_fc": "mlp.fc1",
              "mlp.c_proj": "mlp.fc2", "attn.out_proj": "self_attn.out_proj"}
    for i in range(n_layers):
        src, dst = f"transformer.resblocks.{i}.", f"text_model.encoder.layers.{i}."
        for kind in ("weight", "bias"):
            q, k, v = sd.pop(f"{src}attn.in_proj_{kind}").chunk(3, dim=0)
            out[f"{dst}self_attn.q_proj.{kind}"] = q.clone()
            out[f"{dst}self_attn.k_proj.{kind}"] = k.clone()
            out[f"{dst}self_attn.v_proj.{kind}"] = v.clone()
            for a, b in rename.items():
                out[f"{dst}{b}.{kind}"] = sd.pop(f"{src}{a}.{kind}")
    if sd:
        raise KeyError(f"unconverted OpenCLIP keys: {sorted(sd)[:10]} ...")
    return out


# --------------------------------------------------------------------------- build
def build_module(ctor: Callable[[], nn.Module], sd: StateDict, name: str) -> nn.Module:
    """Construct on the meta device and assign checkpoint tensors (strict)."""
    with torch.device("meta"):
        module = ctor()
    try:
        module.load_state_dict(sd, strict=True, assign=True)
    except RuntimeError as e:  # make architecture mismatches readable
        expected, got = set(module.state_dict()), set(sd)
        missing, unexpected = sorted(expected - got), sorted(got - expected)
        raise RuntimeError(
            f"[{name}] checkpoint does not match the architecture.\n"
            f"  missing ({len(missing)}): {missing[:8]}\n  unexpected ({len(unexpected)}): {unexpected[:8]}\n"
            f"  original error: {str(e)[:500]}") from None
    return module.eval().requires_grad_(False)


@dataclass
class Components:
    spec: ModelSpec
    unet: UNetModel
    vae: AutoencoderKL
    text_encoders: List[CLIPTextModel]


def load_components(path: str, model: Optional[str] = None, dtype: torch.dtype = torch.float16,
                    vae_dtype: Optional[torch.dtype] = None) -> Components:
    """Load UNet, VAE and text encoder(s) onto the CPU in the requested dtypes."""
    model = model or detect_model_type(checkpoint_keys(path))
    spec = SPECS[model]
    vae_dtype = vae_dtype or dtype

    unet = build_module(lambda: UNetModel(spec.unet), read_prefix(path, "model.diffusion_model.", dtype), "unet")
    vae = build_module(lambda: AutoencoderKL(spec.vae),
                       convert_vae(read_prefix(path, "first_stage_model.", vae_dtype)), "vae")
    encoders = []
    for i, (cfg, prefix, fmt) in enumerate(zip(spec.text_encoders, spec.text_encoder_prefixes,
                                               spec.text_encoder_formats)):
        sd = read_prefix(path, prefix, dtype)
        sd = convert_openclip(sd) if fmt == "openclip" else convert_hf_clip(sd)
        encoders.append(build_module(lambda cfg=cfg: CLIPTextModel(cfg), sd, f"text_encoder_{i}"))
    return Components(spec, unet, vae, encoders)


def resolve_checkpoint(model: str, checkpoint: Optional[str], cache_dir: Optional[str]) -> Tuple[str, str, str]:
    """Use a local checkpoint if given (tokenizer files still come from the Hub cache)."""
    if checkpoint is None:
        return download(model, cache_dir)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(checkpoint)
    from huggingface_hub import hf_hub_download

    spec = SPECS[model]
    vocab = hf_hub_download(spec.repo_id, spec.tokenizer_files[0], cache_dir=cache_dir)
    merges = hf_hub_download(spec.repo_id, spec.tokenizer_files[1], cache_dir=cache_dir)
    return checkpoint, vocab, merges
