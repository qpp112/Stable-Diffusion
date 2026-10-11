"""Numerical parity check against Hugging Face diffusers (the reference implementation).

Loads the *same* single-file checkpoint into diffusers and into the from-scratch modules,
feeds identical inputs to every component and reports the max abs / relative difference:
tokenizer ids, text encoder(s), UNet noise prediction (incl. SDXL size conditioning),
VAE encoder and decoder.

  pip install diffusers transformers accelerate
  python scripts/parity_check.py --model sdxl            # fp16 on GPU
  python scripts/parity_check.py --model sd15 --fp32     # tighter tolerances
"""

import argparse
import gc
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from sdscratch.loader import load_components, resolve_checkpoint  # noqa: E402
from sdscratch.models.unet import timestep_embedding  # noqa: E402
from sdscratch.tokenizer import CLIPTokenizer  # noqa: E402

PROMPT = "a photograph of an astronaut riding a horse on mars, highly detailed"


def synthetic_image(size, device, dtype):
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    img = torch.stack([torch.sin(3 * xx), torch.cos(2 * yy), xx * yy])[None]
    return img.to(device, dtype)


@torch.no_grad()
def reference(model, ckpt, dtype, dev, size):
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

    sdxl = model == "sdxl"
    pipe = (StableDiffusionXLPipeline if sdxl else StableDiffusionPipeline).from_single_file(ckpt, torch_dtype=dtype)
    r = {}
    tok = lambda t: t(PROMPT, padding="max_length", max_length=77, truncation=True,  # noqa: E731
                      return_tensors="pt").input_ids
    r["ids_1"] = tok(pipe.tokenizer)
    te = pipe.text_encoder.to(dev)
    o = te(r["ids_1"].to(dev), output_hidden_states=True)
    r["te_1"] = (o.hidden_states[-2] if sdxl else o.last_hidden_state).float().cpu()
    te.to("cpu")
    if sdxl:
        r["ids_2"] = tok(pipe.tokenizer_2)
        te2 = pipe.text_encoder_2.to(dev)
        o2 = te2(r["ids_2"].to(dev), output_hidden_states=True)
        r["te_2"], r["pooled"] = o2.hidden_states[-2].float().cpu(), o2.text_embeds.float().cpu()
        te2.to("cpu")
    ctx = torch.cat([r["te_1"], r["te_2"]], -1) if sdxl else r["te_1"]
    g = torch.Generator().manual_seed(0)
    r["x"] = torch.randn(1, 4, size // 8, size // 8, generator=g)
    unet = pipe.unet.to(dev)
    extra = {}
    if sdxl:
        extra["added_cond_kwargs"] = {"text_embeds": r["pooled"].to(dev, dtype),
                                      "time_ids": torch.tensor([[size, size, 0, 0, size, size]], device=dev,
                                                               dtype=dtype)}
    r["unet"] = unet(r["x"].to(dev, dtype), torch.tensor([500], device=dev), encoder_hidden_states=ctx.to(dev, dtype),
                     **extra).sample.float().cpu()
    unet.to("cpu")
    vae = pipe.vae.to(dev, torch.float32)
    img = synthetic_image(size, dev, torch.float32)
    r["vae_enc"] = vae.encode(img).latent_dist.mean.float().cpu()
    r["vae_dec"] = vae.decode(r["vae_enc"].to(dev)).sample.float().cpu()
    del pipe, unet, vae, te
    return r


@torch.no_grad()
def ours(model, ckpt, vocab, merges, dtype, dev, size, r):
    sdxl = model == "sdxl"
    c = load_components(ckpt, model, dtype=dtype, vae_dtype=torch.float32)
    tok = CLIPTokenizer(vocab, merges)
    o = {"ids_1": torch.tensor(tok([PROMPT], pad_id=c.text_encoders[0].cfg.pad_id))}
    te = c.text_encoders[0].to(dev)
    out = te(o["ids_1"].to(dev))
    o["te_1"] = (out.penultimate if sdxl else out.last_hidden_state).float().cpu()
    te.to("cpu")
    if sdxl:
        o["ids_2"] = torch.tensor(tok([PROMPT], pad_id=c.text_encoders[1].cfg.pad_id))
        te2 = c.text_encoders[1].to(dev)
        out2 = te2(o["ids_2"].to(dev))
        o["te_2"], o["pooled"] = out2.penultimate.float().cpu(), out2.pooled.float().cpu()
        te2.to("cpu")
    # Feed the *reference* conditioning so UNet differences are not mixed with encoder differences.
    ctx = torch.cat([r["te_1"], r["te_2"]], -1) if sdxl else r["te_1"]
    y = None
    if sdxl:
        size_emb = timestep_embedding(torch.tensor([size, size, 0, 0, size, size], dtype=torch.float32), 256)
        y = torch.cat([r["pooled"], size_emb.reshape(1, -1)], -1).to(dev, dtype)
    unet = c.unet.to(dev)
    o["unet"] = unet(r["x"].to(dev, dtype), torch.tensor([500.0], device=dev), ctx.to(dev, dtype), y).float().cpu()
    unet.to("cpu")
    vae = c.vae.to(dev)
    o["vae_enc"] = vae.encode(synthetic_image(size, dev, torch.float32)).float().cpu()
    o["vae_dec"] = vae.decode(r["vae_enc"].to(dev)).float().cpu()
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["sd15", "sdxl"], default="sdxl")
    ap.add_argument("--checkpoint")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if (a.fp32 or dev.type == "cpu") else torch.float16
    ckpt, vocab, merges = resolve_checkpoint(a.model, a.checkpoint, None)

    r = reference(a.model, ckpt, dtype, dev, a.size)
    gc.collect()
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    o = ours(a.model, ckpt, vocab, merges, dtype, dev, a.size, r)

    tol = 1e-3 if dtype == torch.float32 else 2e-2
    print(f"\n{a.model.upper()} parity vs diffusers ({dtype}, {dev}); relative = max|ours-ref| / max|ref|\n")
    print(f"{'component':12s} {'max abs diff':>14s} {'relative':>10s}  result")
    ok_all = True
    for k in r:
        if k == "x":
            continue
        if k.startswith("ids"):
            same = torch.equal(r[k], o[k])
            ok_all &= same
            print(f"{k:12s} {'':>14s} {'':>10s}  {'identical' if same else 'DIFFERENT'}")
            continue
        d = (o[k] - r[k]).abs().max().item()
        rel = d / max(r[k].abs().max().item(), 1e-12)
        ok = rel < tol
        ok_all &= ok
        print(f"{k:12s} {d:14.3e} {rel:10.2e}  {'PASS' if ok else 'FAIL'}")
    print("\nALL PASS" if ok_all else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()
