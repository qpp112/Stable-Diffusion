"""Text-to-image pipeline: prompt -> CLIP -> iterative denoising in latent space -> VAE decode.

    pipe = StableDiffusion.from_pretrained("sdxl", memory=MemoryOptions(offload="model"))
    image = pipe("a watercolor fox in a snowy forest", steps=25, sampler="dpm++2m", seed=0)[0]

Classifier-free guidance (Ho & Salimans, 2022):
    eps = eps(x, t, empty) + w * (eps(x, t, prompt) - eps(x, t, empty))
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence, Union

import torch

from .configs import ModelSpec
from .loader import Components, load_components, resolve_checkpoint
from .memory import MemoryReport, Offloader, track
from .models.attention import set_attention_backend
from .models.unet import timestep_embedding
from .samplers import NoiseSchedule, get_sampler
from .tokenizer import CLIPTokenizer


@dataclass
class MemoryOptions:
    offload: str = "none"            # "none" | "model" | "sequential"   (see memory.py)
    attention: str = "sdpa"          # "sdpa" | "sliced" | "naive"        (see models/attention.py)
    attention_slice_size: int = 1024
    vae_tiling: Optional[bool] = None  # None = automatic (on above 768x768 output)
    vae_tile: int = 64               # latent pixels per tile (= 512 image pixels)
    vae_overlap: int = 16
    cfg_mode: str = "batch"          # "batch": cond+uncond in one UNet call; "sequential": two calls
    channels_last: bool = False


PRESETS = {
    # roughly ordered from fastest to most memory-frugal
    "fast": MemoryOptions(),
    "balanced": MemoryOptions(offload="model", vae_tiling=True),
    "low": MemoryOptions(offload="model", vae_tiling=True, cfg_mode="sequential"),
    "minimal": MemoryOptions(offload="sequential", vae_tiling=True, cfg_mode="sequential"),
}


@dataclass
class Conditioning:
    context: torch.Tensor                 # (B, 77, context_dim) cross-attention tokens
    y: Optional[torch.Tensor] = None      # (B, 2816) SDXL vector condition


@dataclass
class GenerationResult:
    images: list
    latents: torch.Tensor
    report: MemoryReport = field(default_factory=MemoryReport)

    def __getitem__(self, i):
        return self.images[i]


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_pil(images: torch.Tensor) -> list:
    from PIL import Image

    arr = ((images.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return [Image.fromarray(a) for a in arr.permute(0, 2, 3, 1).cpu().numpy()]


class StableDiffusion:
    def __init__(self, components: Components, tokenizer: CLIPTokenizer, device: Union[str, torch.device],
                 dtype: torch.dtype, memory: Optional[MemoryOptions] = None):
        self.spec: ModelSpec = components.spec
        self.unet = components.unet
        self.vae = components.vae
        self.text_encoders = components.text_encoders
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dtype = dtype
        self.schedule = NoiseSchedule()   # SD 1.5 and SDXL: scaled_linear, 0.00085 -> 0.012, T = 1000
        self.offloader = Offloader(self.device)
        self.memory = MemoryOptions()
        self.set_memory_options(memory or MemoryOptions())

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, model: str = "sdxl", checkpoint: Optional[str] = None,
                        device: Optional[str] = None, dtype: Optional[torch.dtype] = None,
                        vae_dtype: Optional[torch.dtype] = None, memory: Optional[MemoryOptions] = None,
                        cache_dir: Optional[str] = None) -> "StableDiffusion":
        dev = torch.device(device or default_device())
        if dtype is None:
            dtype = torch.float16 if dev.type in ("cuda", "mps") else torch.float32
        if vae_dtype is None:
            # The SDXL VAE overflows in fp16 (NaNs -> black images); fp32 + tiling is cheap.
            vae_dtype = torch.float32 if (model == "sdxl" and dtype == torch.float16) else dtype
        ckpt, vocab, merges = resolve_checkpoint(model, checkpoint, cache_dir)
        comps = load_components(ckpt, model, dtype=dtype, vae_dtype=vae_dtype)
        return cls(comps, CLIPTokenizer(vocab, merges), dev, dtype, memory)

    def set_memory_options(self, memory: Optional[MemoryOptions] = None, **overrides) -> None:
        memory = replace(memory or self.memory, **overrides)
        self.memory = memory
        set_attention_backend(memory.attention, memory.attention_slice_size)
        self.offloader.mode = memory.offload
        self.unet.to(memory_format=torch.channels_last if memory.channels_last else torch.contiguous_format)
        self.offloader.place([*self.text_encoders, self.unet, self.vae], self.unet)

    # ---------------------------------------------------------------- prompting
    def _tokens(self, prompts: Sequence[str], pad_id: int) -> torch.Tensor:
        return torch.tensor(self.tokenizer(prompts, pad_id=pad_id), dtype=torch.long, device=self.device)

    def encode_prompt(self, prompts: Sequence[str]) -> Conditioning:
        if not self.spec.is_sdxl:  # SD 1.5: last layer of CLIP ViT-L
            te = self.text_encoders[0]
            return Conditioning(te(self._tokens(prompts, te.cfg.pad_id)).last_hidden_state)
        # SDXL: penultimate layers of ViT-L (768) and ViT-bigG (1280), concatenated -> 2048;
        # the pooled, projected bigG embedding becomes part of the vector condition.
        te_l, te_g = self.text_encoders
        out_l = te_l(self._tokens(prompts, te_l.cfg.pad_id))
        out_g = te_g(self._tokens(prompts, te_g.cfg.pad_id))
        return Conditioning(torch.cat([out_l.penultimate, out_g.penultimate], dim=-1), out_g.pooled)

    def _size_condition(self, height: int, width: int, batch: int) -> torch.Tensor:
        """SDXL micro-conditioning: Fourier features of (original size, crop top-left, target size)."""
        vals = torch.tensor([height, width, 0, 0, height, width], dtype=torch.float32, device=self.device)
        return timestep_embedding(vals, 256).reshape(1, -1).expand(batch, -1)   # (B, 6 * 256)

    # --------------------------------------------------------------- denoising
    def _eps_fn(self, cond: Conditioning, uncond: Optional[Conditioning], guidance: float) -> Callable:
        unet, dtype, mem = self.unet, self.dtype, self.memory
        if uncond is not None and mem.cfg_mode == "batch":
            ctx2 = torch.cat([uncond.context, cond.context])
            y2 = None if cond.y is None else torch.cat([uncond.y, cond.y])

        def eps_fn(x: torch.Tensor, t: float) -> torch.Tensor:
            xin = x.to(dtype)
            if mem.channels_last:
                xin = xin.contiguous(memory_format=torch.channels_last)
            b = x.shape[0]
            if uncond is None:
                return unet(xin, torch.full((b,), float(t), device=x.device), cond.context, cond.y).float()
            if mem.cfg_mode == "batch":
                out = unet(torch.cat([xin, xin]), torch.full((2 * b,), float(t), device=x.device), ctx2, y2)
                e_u, e_c = out.float().chunk(2)
            else:
                tt = torch.full((b,), float(t), device=x.device)
                e_u = unet(xin, tt, uncond.context, uncond.y).float()
                e_c = unet(xin, tt, cond.context, cond.y).float()
            return e_u + guidance * (e_c - e_u)

        return eps_fn

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae_dtype = self.vae.post_quant_conv.weight.dtype
        z = (latents / self.spec.vae.scaling_factor).to(vae_dtype)
        tiling = self.memory.vae_tiling
        if tiling is None:
            tiling = z.shape[-2] * z.shape[-1] > 96 * 96
        if tiling:
            return self.vae.decode_tiled(z, self.memory.vae_tile, self.memory.vae_overlap).float()
        return self.vae.decode(z).float()

    @torch.no_grad()  # (not inference_mode: offload hooks move parameters inside the call)
    def __call__(self, prompt: Union[str, Sequence[str]], negative_prompt: Union[str, Sequence[str]] = "",
                 height: Optional[int] = None, width: Optional[int] = None, steps: int = 25,
                 guidance: Optional[float] = None, sampler: str = "dpm++2m", spacing: str = "karras",
                 seed: int = 0, eta: float = 0.0, latents: Optional[torch.Tensor] = None,
                 progress: bool = True, callback: Optional[Callable] = None,
                 decode: bool = True) -> GenerationResult:
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        b = len(prompts)
        negs = [negative_prompt] * b if isinstance(negative_prompt, str) else list(negative_prompt)
        h = height or self.spec.default_size
        w = width or self.spec.default_size
        m = self.spec.latent_multiple
        if h % m or w % m:
            raise ValueError(f"height and width must be multiples of {m} for {self.spec.name}")
        guidance = self.spec.default_guidance if guidance is None else guidance
        dev = self.device
        report = MemoryReport()

        # 1) text conditioning
        with track(report, "text_encoders", dev), self.offloader.use(*self.text_encoders):
            cond = self.encode_prompt(prompts)
            uncond = None
            if guidance != 1.0:
                if self.spec.is_sdxl and all(n == "" for n in negs):
                    # SDXL was trained with zeroed embeddings for dropped prompts
                    uncond = Conditioning(torch.zeros_like(cond.context), torch.zeros_like(cond.y))
                else:
                    uncond = self.encode_prompt(negs)
        if self.spec.is_sdxl:
            size = self._size_condition(h, w, b).to(cond.y.dtype)
            cond.y = torch.cat([cond.y, size], dim=-1)
            if uncond is not None:
                uncond.y = torch.cat([uncond.y, size], dim=-1)

        # 2) denoising loop (latents kept in fp32; UNet runs in self.dtype)
        gen = torch.Generator("cpu").manual_seed(seed)   # CPU RNG -> same image on any device
        f = self.spec.vae.downsample_factor
        shape = (b, self.spec.unet.in_channels, h // f, w // f)
        x = latents.to(dev, torch.float32) if latents is not None else \
            torch.randn(shape, generator=gen, dtype=torch.float32).to(dev)
        kwargs = {"spacing": spacing}
        if sampler == "ddim":
            kwargs["eta"] = eta
        smp = get_sampler(sampler, self.schedule, steps, **kwargs)
        noise_fn = lambda z: torch.randn(z.shape, generator=gen, dtype=torch.float32).to(z.device)  # noqa: E731

        bar = None
        if progress:
            try:
                from tqdm.auto import tqdm
                bar = tqdm(total=smp.nfe, desc=f"{smp.name} x{smp.nfe}", leave=False)
            except ImportError:
                pass

        def cb(i, t, xt, x0):
            if bar is not None:
                bar.update(1)
            if callback is not None:
                callback(i, t, xt, x0)

        with track(report, "unet", dev), self.offloader.use(self.unet):
            x = smp.sample(self._eps_fn(cond, uncond, guidance), x, noise_fn=noise_fn, callback=cb)
        if bar is not None:
            bar.close()

        # 3) decode
        images = []
        if decode:
            with track(report, "vae", dev), self.offloader.use(self.vae):
                images = to_pil(self.decode_latents(x))
        return GenerationResult(images=images, latents=x, report=report)
