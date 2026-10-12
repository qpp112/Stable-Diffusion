"""Peak-VRAM / speed ladder for 1024x1024 (or any size) generation.

Each configuration adds one optimisation on top of the previous one and runs in a fresh
subprocess (clean CUDA allocator, survives OOM). Results go to <out>.json / .md / .png.

  python scripts/benchmark_memory.py --model sdxl --size 1024 --steps 20
  python scripts/benchmark_memory.py --model sdxl --vram-limit-gb 8    # emulate an 8 GB card

``--vram-limit-gb`` caps this process's CUDA allocator with
``torch.cuda.set_per_process_memory_fraction`` so a big Colab/HPC GPU behaves like a
consumer card: anything that needs more raises OOM.
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LADDER = [
    ("fp32, naive attention (baseline)", dict(dtype="fp32", attention="naive", vae_tiling=False)),
    ("+ fp16 weights", dict(dtype="fp16", attention="naive", vae_tiling=False)),
    ("+ SDPA attention", dict(dtype="fp16", attention="sdpa", vae_tiling=False)),
    ("+ tiled VAE decode", dict(dtype="fp16", attention="sdpa", vae_tiling=True)),
    ("+ model offload", dict(dtype="fp16", attention="sdpa", vae_tiling=True, offload="model")),
    ("+ sequential CFG", dict(dtype="fp16", attention="sdpa", vae_tiling=True, offload="model",
                              cfg_mode="sequential")),
    ("+ block-wise UNet offload", dict(dtype="fp16", attention="sdpa", vae_tiling=True, offload="sequential",
                                       cfg_mode="sequential")),
]
EXTRA = [("fp16, sliced attention (no SDPA)", dict(dtype="fp16", attention="sliced", vae_tiling=True))]


def worker(cfg: dict) -> dict:
    import torch

    from sdscratch.pipeline import MemoryOptions, StableDiffusion

    dev = torch.device("cuda", torch.cuda.current_device())   # set_per_process_memory_fraction needs an index
    if cfg.get("vram_limit_gb"):
        total = torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, cfg["vram_limit_gb"] * 2**30 / total), dev)
    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[cfg["dtype"]]
    mem = MemoryOptions(offload=cfg.get("offload", "none"), attention=cfg["attention"],
                        vae_tiling=cfg["vae_tiling"], cfg_mode=cfg.get("cfg_mode", "batch"))
    result = {"gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__}
    try:
        pipe = StableDiffusion.from_pretrained(cfg["model"], checkpoint=cfg.get("checkpoint"), device="cuda",
                                               dtype=dtype, memory=mem)
        kw = dict(height=cfg["size"], width=cfg["size"], sampler="dpm++2m", seed=0, progress=False)
        pipe("warm-up", steps=2, **kw)                                    # cuDNN autotune, allocator warm-up
        res = pipe(cfg["prompt"], steps=cfg["steps"], **kw)
        os.makedirs(os.path.dirname(cfg["image"]), exist_ok=True)
        res.images[0].save(cfg["image"])
        result.update(status="ok", peak_gb=res.report.peak_gb,
                      stages={k: {"seconds": v.seconds, "peak_gb": v.peak_bytes / 2**30}
                              for k, v in res.report.stages.items()},
                      unet_s_per_step=res.report.stages["unet"].seconds / cfg["steps"],
                      total_s=sum(v.seconds for v in res.report.stages.values()))
    except torch.cuda.OutOfMemoryError as e:
        result.update(status="OOM", error=str(e).splitlines()[0][:200])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["sd15", "sdxl"], default="sdxl")
    ap.add_argument("--checkpoint")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--prompt", default="a photograph of a red fox in a snowy birch forest, golden hour, 85mm")
    ap.add_argument("--vram-limit-gb", type=float)
    ap.add_argument("--skip-baseline", action="store_true", help="skip fp32 (needs ~14 GB host RAM for SDXL)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.worker:
        print("RESULT " + json.dumps(worker(json.loads(a.worker))), flush=True)
        return

    tag = f"{a.model}_{a.size}" + (f"_limit{a.vram_limit_gb:g}gb" if a.vram_limit_gb else "")
    out = a.out or os.path.join(ROOT, "results", f"memory_{tag}")
    rows = []
    configs = (LADDER[1:] if a.skip_baseline else LADDER) + EXTRA
    for name, opts in configs:
        cfg = dict(opts, model=a.model, checkpoint=a.checkpoint, size=a.size, steps=a.steps, prompt=a.prompt,
                   vram_limit_gb=a.vram_limit_gb, image=os.path.join(out + "_images", f"{len(rows)}.png"))
        print(f"\n=== {name}", flush=True)
        proc = subprocess.run([sys.executable, __file__, "--worker", json.dumps(cfg)], capture_output=True, text=True)
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
        res = json.loads(line[-1][7:]) if line else {"status": "crashed", "error": proc.stderr[-500:]}
        res["config"] = name
        rows.append(res)
        print({k: res.get(k) for k in ("status", "peak_gb", "unet_s_per_step", "total_s")}, flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".json", "w") as f:
        json.dump(rows, f, indent=1)
    gpu = next((r.get("gpu") for r in rows if r.get("gpu")), "?")
    lines = [f"**{a.model.upper()} {a.size}x{a.size}, {a.steps} DPM-Solver++(2M) steps, CFG on, GPU: {gpu}"
             + (f", allocator capped at {a.vram_limit_gb:g} GB" if a.vram_limit_gb else "") + "**", "",
             "| Configuration | Peak VRAM (GiB) | UNet s/step | Total s | Status |", "|---|---:|---:|---:|---|"]
    for r in rows:
        ok = r["status"] == "ok"
        lines.append(f"| {r['config']} | {r['peak_gb']:.2f} |" if ok else f"| {r['config']} | - |")
        lines[-1] += (f" {r['unet_s_per_step']:.2f} | {r['total_s']:.1f} | ok |" if ok else f" - | - | {r['status']} |")
    with open(out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.2))
        names = [r["config"] for r in rows][::-1]
        vals = [r.get("peak_gb", 0) if r["status"] == "ok" else 0 for r in rows][::-1]
        bars = ax.barh(names, vals, color=["#4e79a7" if v else "#e15759" for v in vals])
        for b, r in zip(bars, rows[::-1]):
            ax.text(b.get_width() + 0.1, b.get_y() + b.get_height() / 2,
                    f"{r['peak_gb']:.1f} GiB" if r["status"] == "ok" else r["status"], va="center", fontsize=9)
        ax.set_xlim(0, max(max(vals), a.vram_limit_gb or 0, 1) * 1.18)
        if a.vram_limit_gb:
            ax.axvline(a.vram_limit_gb, c="k", ls="--", lw=1)
        ax.set_xlabel("peak allocated VRAM (GiB)")
        ax.set_title(f"{a.model.upper()} {a.size}x{a.size} on {gpu}")
        fig.tight_layout()
        fig.savefig(out + ".png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
