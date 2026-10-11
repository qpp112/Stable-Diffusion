"""Generate images from the command line.

Examples
--------
  # SDXL, 1024x1024, fits in ~8 GB with model offload + tiled VAE
  python scripts/generate.py --model sdxl --prompt "a watercolor fox in a snowy forest" --preset balanced

  # SD 1.5, compare samplers quickly
  python scripts/generate.py --model sd15 --sampler ddim --steps 50 --prompt "a lighthouse at dusk"
"""

import argparse
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from sdscratch.pipeline import PRESETS, StableDiffusion  # noqa: E402
from sdscratch.samplers import SAMPLERS  # noqa: E402

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["sd15", "sdxl"], default="sdxl")
    p.add_argument("--checkpoint", help="local .safetensors (default: download from the Hub)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default="")
    p.add_argument("--height", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float)
    p.add_argument("--sampler", choices=sorted(SAMPLERS), default="dpm++2m")
    p.add_argument("--spacing", choices=["trailing", "leading", "linspace", "karras", "logsnr"], default="karras")
    p.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity (0 = deterministic)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num", type=int, default=1, help="number of images (consecutive seeds)")
    p.add_argument("--preset", choices=sorted(PRESETS), default="fast", help="memory preset")
    p.add_argument("--offload", choices=["none", "model", "sequential"])
    p.add_argument("--attention", choices=["sdpa", "sliced", "naive"])
    p.add_argument("--vae-tiling", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--dtype", choices=sorted(DTYPES))
    p.add_argument("--device")
    p.add_argument("--out", default="outputs")
    a = p.parse_args()

    overrides = {k: v for k, v in dict(offload=a.offload, attention=a.attention).items() if v}
    if a.vae_tiling != "auto":
        overrides["vae_tiling"] = a.vae_tiling == "on"
    mem = replace(PRESETS[a.preset], **overrides)   # decided before loading, so weights never overflow the GPU
    pipe = StableDiffusion.from_pretrained(a.model, checkpoint=a.checkpoint, device=a.device,
                                           dtype=DTYPES[a.dtype] if a.dtype else None, memory=mem)

    os.makedirs(a.out, exist_ok=True)
    for i in range(a.num):
        seed = a.seed + i
        res = pipe(a.prompt, negative_prompt=a.negative, height=a.height, width=a.width, steps=a.steps,
                   guidance=a.guidance, sampler=a.sampler, spacing=a.spacing, eta=a.eta, seed=seed)
        path = os.path.join(a.out, f"{a.model}_{a.sampler}_{a.steps}steps_seed{seed}.png")
        res.images[0].save(path)
        print(f"saved {path}   [{res.report.summary()}]")


if __name__ == "__main__":
    main()
