"""Sampler study on a 2-D Gaussian mixture with an *exact* score (NumPy + SciPy, CPU, ~1 min).

Because the optimal noise predictor of a Gaussian mixture is known in closed form, every
error measured here is pure *solver* error - no network approximation is involved. The
noise schedule and timestep grids are exactly the ones Stable Diffusion uses.

Outputs
  assets/toy_samples.png      samples from each sampler at 10 NFE vs. the data
  assets/toy_convergence.png  error vs. NFE w.r.t. the exact probability-flow ODE solution:
                              (a) solvers on one timestep grid, (b) one solver on four grids
  assets/toy_guidance.png     solver error vs. classifier-free guidance scale
  results/toy_gmm.json        all numbers
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdscratch.samplers import NoiseSchedule, get_sampler  # noqa: E402
from sdscratch.toy import GaussianMixture, sliced_wasserstein  # noqa: E402

ODE_SAMPLERS = ["ddim", "dpm2m", "dpm++2m", "dpm3m", "dpm++3m"]
LABELS = {"ddpm": "DDPM (ancestral)", "ddim": "DDIM", "dpm2m": "DPM-Solver-2M", "dpm++2m": "DPM-Solver++(2M)",
          "dpm3m": "DPM-Solver-3M", "dpm++3m": "DPM-Solver++(3M)"}
COLORS = {"ddpm": "#8c8c8c", "ddim": "#e15759", "dpm2m": "#f28e2b", "dpm++2m": "#4e79a7",
          "dpm3m": "#edc948", "dpm++3m": "#59a14f"}


def exact_ode(eps_as, x_T, alpha_T, sigma_T):
    """Integrate the probability-flow ODE  dy/ds = eps(alpha y; alpha, sigma)  in the variable
    s = sigma/alpha (y = x/alpha) from s_T down to 0 with an 8th-order adaptive solver."""
    s_T = sigma_T / alpha_T

    def f(s, y):
        a = 1.0 / math.sqrt(1.0 + s * s)
        return eps_as(a * y.reshape(-1, 2), a, s * a).ravel()

    sol = solve_ivp(f, (s_T, 0.0), (x_T / alpha_T).ravel(), method="DOP853", rtol=1e-10, atol=1e-10)
    assert sol.success, sol.message
    return sol.y[:, -1].reshape(-1, 2)   # at s = 0, alpha = 1 so x = y


def run(sampler, n, eps_fn, x_T, rng, spacing):
    smp = get_sampler(sampler, SCH, n, spacing=spacing)
    return smp.sample(eps_fn, x_T.copy(), noise_fn=lambda z: rng.standard_normal(z.shape))


SCH = NoiseSchedule()
SPACINGS = ["trailing", "linspace", "karras", "logsnr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="number of samples")
    ap.add_argument("--spacing", default="karras", choices=SPACINGS, help="grid for the solver comparison")
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    os.makedirs(args.assets, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    gmm = GaussianMixture.ring(n_modes=8, radius=1.5, std=0.08)
    data = gmm.sample(args.n, rng)
    eps = gmm.eps_fn(SCH)
    T = SCH.num_train_timesteps - 1          # every grid used here starts at t = 999
    x_T = rng.standard_normal((args.n, 2))
    ref = exact_ode(lambda x, a, s: -s * gmm.score(x, a, s), x_T, SCH.alpha(T), SCH.sigma(T))

    def rmse(x0, target):
        return float(np.sqrt(((x0 - target) ** 2).sum(1).mean()))

    nfes = [5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100]
    out = {"spacing": args.spacing, "nfe": nfes, "ode_rmse": {}, "spacing_study": {}, "swd": {},
           "swd_floor": sliced_wasserstein(data, gmm.sample(args.n, np.random.default_rng(1))),
           "swd_exact_ode": sliced_wasserstein(data, ref)}

    # 1) solvers on one grid
    for name in ["ddpm"] + ODE_SAMPLERS:
        out["swd"][name], out["ode_rmse"][name] = [], []
        for n in nfes:
            x0 = run(name, n, eps, x_T, np.random.default_rng(123), args.spacing)
            out["swd"][name].append(sliced_wasserstein(data, x0))
            if name != "ddpm":
                out["ode_rmse"][name].append(rmse(x0, ref))
        if name != "ddpm":
            print(f"{name:8s} ode-rmse @10 NFE = {out['ode_rmse'][name][3]:.2e}   @20 = {out['ode_rmse'][name][6]:.2e}")

    # 2) one solver on every grid
    for sp in SPACINGS:
        out["spacing_study"][sp] = [rmse(run("dpm++2m", n, eps, x_T, rng, sp), ref) for n in nfes]
        print(f"dpm++2m / {sp:9s} ode-rmse @10 NFE = {out['spacing_study'][sp][3]:.2e}")

    # ---------------------------------------------------------------- figure: samples @ 10 NFE
    show = ["ddpm", "ddim", "dpm2m", "dpm++2m", "dpm++3m"]
    fig, axes = plt.subplots(1, len(show) + 1, figsize=(2.7 * (len(show) + 1), 3.0))
    m = 1000
    axes[0].scatter(*data[:m].T, s=2, c="k", alpha=0.5)
    axes[0].set_title("data", fontsize=10)
    for ax, name in zip(axes[1:], show):
        x0 = run(name, 10, eps, x_T, np.random.default_rng(123), args.spacing)
        ax.scatter(*x0[:m].T, s=2, c=COLORS[name], alpha=0.6)
        ax.set_title(f"{LABELS[name]}, 10 NFE", fontsize=9)
    for ax in axes:
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(args.assets, "toy_samples.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------- figure: convergence
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for name in ODE_SAMPLERS:
        a1.loglog(nfes, out["ode_rmse"][name], "o-", ms=3.5, c=COLORS[name], label=LABELS[name])
    a1.set_title(f"(a) solvers, '{args.spacing}' timesteps")
    for sp, c in zip(SPACINGS, ["#b07aa1", "#76b7b2", "#4e79a7", "#9c755f"]):
        a2.loglog(nfes, out["spacing_study"][sp], "o-", ms=3.5, c=c, label=sp)
    a2.set_title("(b) DPM-Solver++(2M), timestep spacing")
    for ax in (a1, a2):
        ax.set_xlabel("NFE (network evaluations)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    a1.set_ylabel("RMSE to exact probability-flow ODE solution")
    fig.suptitle("8-mode Gaussian mixture with exact score, Stable Diffusion noise schedule")
    fig.tight_layout()
    fig.savefig(os.path.join(args.assets, "toy_convergence.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------- guidance study
    cond = gmm.subset([0, 2, 4, 6])           # a "class" = every other mode
    ws = [1.0, 2.0, 3.0, 5.0, 7.5, 10.0]
    g_nfes = [10, 20, 50]
    g_solvers = ["ddim", "dpm2m", "dpm++2m", "dpm++3m"]
    out["guidance"] = {"w": ws, "nfe": g_nfes, "rmse": {}}
    for w in ws:
        def geps_as(x, a, s, w=w):
            su, sc = gmm.score(x, a, s), cond.score(x, a, s)
            return -s * (su + w * (sc - su))

        def geps(x, t, w=w):
            return geps_as(x, SCH.alpha(t), SCH.sigma(t))

        gref = exact_ode(geps_as, x_T, SCH.alpha(T), SCH.sigma(T))
        for name in g_solvers:
            out["guidance"]["rmse"][f"{name}@w={w:g}"] = [
                rmse(run(name, n, geps, x_T, rng, args.spacing), gref) for n in g_nfes]
    fig, axes = plt.subplots(1, len(g_nfes), figsize=(4 * len(g_nfes), 3.6), sharey=True)
    for j, ax in enumerate(axes):
        for name in g_solvers:
            ax.semilogy(ws, [out["guidance"]["rmse"][f"{name}@w={w:g}"][j] for w in ws], "o-",
                        ms=3.5, c=COLORS[name], label=LABELS[name])
        ax.set_title(f"{g_nfes[j]} NFE")
        ax.set_xlabel("guidance scale w")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("RMSE to exact guided PF-ODE")
    axes[0].legend(fontsize=8)
    fig.suptitle("Classifier-free guidance makes the ODE stiff: solver error vs. guidance scale")
    fig.tight_layout()
    fig.savefig(os.path.join(args.assets, "toy_guidance.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(args.results, "toy_gmm.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.join(args.results, "toy_gmm.json"))


if __name__ == "__main__":
    main()
