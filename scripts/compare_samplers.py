"""Sampler comparison on the real model: image grid + convergence of each ODE solver.

For one prompt/seed, every deterministic sampler is run at several NFEs and compared (latent
RMSE) to a high-NFE reference solution of the same guided probability-flow ODE from the same
initial noise. DDPM is stochastic, so it only appears in the image grid.

  python scripts/compare_samplers.py --model sd15 --nfe 5 10 20 50
  python scripts/compare_samplers.py --model sdxl --nfe 10 20 30 --preset balanced
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from sdscratch.pipeline import PRESETS, StableDiffusion  # noqa: E402

ODE = ["ddim", "dpm2m", "dpm++2m", "dpm++3m"]
LABELS = {"ddpm": "DDPM", "ddim": "DDIM", "dpm2m": "DPM-Solver-2M", "dpm++2m": "DPM-Solver++(2M)",
          "dpm++3m": "DPM-Solver++(3M)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["sd15", "sdxl"], default="sd15")
    ap.add_argument("--checkpoint")
    ap.add_argument("--prompt", default="a cozy reading nook with a large window, rainy evening, warm lamp light, "
                                        "detailed illustration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int)
    ap.add_argument("--guidance", type=float)
    ap.add_argument("--nfe", type=int, nargs="+", default=[5, 10, 20, 50])
    ap.add_argument("--samplers", nargs="+", default=["ddpm"] + ODE)
    ap.add_argument("--spacing", default="karras")
    ap.add_argument("--reference-nfe", type=int, default=500)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="fast")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()

    pipe = StableDiffusion.from_pretrained(a.model, checkpoint=a.checkpoint, memory=PRESETS[a.preset])
    kw = dict(height=a.size, width=a.size, guidance=a.guidance, seed=a.seed, spacing=a.spacing)
    print(f"reference: dpm++2m with {a.reference_nfe} steps")
    ref = pipe(a.prompt, steps=a.reference_nfe, sampler="dpm++2m", **kw)
    tag = f"{a.model}_seed{a.seed}"
    os.makedirs(a.out, exist_ok=True)
    ref.images[0].save(os.path.join(a.out, f"samplers_{tag}_reference.png"))

    rmse, grid = {}, {}
    for name in a.samplers:
        rmse[name] = []
        for n in a.nfe:
            res = pipe(a.prompt, steps=n, sampler=name, **kw)
            grid[(name, n)] = res.images[0]
            if name in ODE:
                rmse[name].append(float(((res.latents - ref.latents) ** 2).mean().sqrt()))
        print(name, ["%.3f" % e for e in rmse[name]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, cols = len(a.samplers), len(a.nfe)
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows), squeeze=False)
    for i, name in enumerate(a.samplers):
        for j, n in enumerate(a.nfe):
            ax = axes[i][j]
            ax.imshow(grid[(name, n)].resize((256, 256)))
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"{n} NFE")
            if j == 0:
                ax.set_ylabel(LABELS.get(name, name))
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"samplers_{tag}_grid.jpg"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name in a.samplers:
        if rmse[name]:
            ax.loglog(a.nfe, rmse[name], "o-", label=LABELS.get(name, name))
    ax.set_xlabel("NFE")
    ax.set_ylabel(f"latent RMSE vs. {a.reference_nfe}-step solution")
    ax.set_title(f"{a.model.upper()}, '{a.spacing}' timesteps")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"samplers_{tag}_convergence.png"), dpi=150)
    with open(os.path.join(a.out, f"samplers_{tag}.json"), "w") as f:
        json.dump({"args": vars(a), "rmse": rmse}, f, indent=1)
    print("saved results to", a.out)


if __name__ == "__main__":
    with torch.no_grad():
        main()
