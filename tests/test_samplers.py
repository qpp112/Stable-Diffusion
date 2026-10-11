"""Sampler correctness tests. NumPy only - no torch or GPU needed.

Key idea: for Gaussian data x0 ~ N(mu, s^2 I) the probability-flow ODE has the closed form
    x_t = alpha_t * mu + sqrt(v_t / v_T) * (x_T - alpha_T * mu),   v_t = alpha_t^2 s^2 + sigma_t^2,
and the optimal noise predictor is linear: eps*(x, t) = sigma_t (x - alpha_t mu) / v_t.
So we can measure each ODE solver's error exactly and check its order of convergence.

Run with `pytest` or simply `python tests/test_samplers.py`.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdscratch.samplers import DPMSolverMultistep, NoiseSchedule, get_sampler  # noqa: E402
from sdscratch.toy import GaussianMixture  # noqa: E402

MU, S = np.array([0.7, -1.2]), 0.5
SCH = NoiseSchedule()


def gaussian_eps(x, t):
    a, s = SCH.alpha(t), SCH.sigma(t)
    return s * (x - a * MU) / (a**2 * S**2 + s**2)


def exact_ode(x_T, t_from, t_to):
    v = lambda t: SCH.alpha(t) ** 2 * S**2 + SCH.sigma(t) ** 2  # noqa: E731
    return SCH.alpha(t_to) * MU + math.sqrt(v(t_to) / v(t_from)) * (x_T - SCH.alpha(t_from) * MU)


def solver_error(name, n, x_T, spacing="logsnr"):
    """Error of the state at the last evaluated timestep (isolates ODE-solver error).

    ``logsnr`` spacing makes the steps uniform in lambda, so the error shrinks as h^p and
    the observed order p is clean. (With SD's integer grids the lambda-steps near t=0
    hardly shrink as n grows, which hides the asymptotic order.)
    """
    smp = get_sampler(name, SCH, n, spacing=spacing)
    seen = {}
    smp.sample(gaussian_eps, x_T, callback=lambda i, t, x, x0: seen.update(t=t, x=x))
    ref = exact_ode(x_T, smp.timesteps[0], seen["t"])
    return float(np.abs(seen["x"] - ref).max())


def local_error(order, algorithm, h, x_T):
    """One multistep update from exact history at lambda = -2h, -h, 0 to lambda = +h."""
    smp = DPMSolverMultistep(SCH, 10, order=order, algorithm=algorithm, lower_order_final=False)
    t_of = lambda lam: float(SCH.sigma_to_t(np.exp(-lam)))  # noqa: E731
    ts = [t_of(-2 * h), t_of(-h), t_of(0.0), t_of(h)]
    xs = [exact_ode(x_T, 999, t) for t in ts]
    out = None
    for i in range(3):
        out = smp.step(xs[i], gaussian_eps(xs[i], ts[i]), ts[i], ts[i + 1], i, None)
    return float(np.abs(out - xs[3]).max())


def test_timesteps():
    for spacing in ("trailing", "leading", "linspace"):
        for n in (1, 5, 20, 50, 1000):
            ts = SCH.timesteps(n, spacing)
            assert len(ts) == n, (spacing, n, len(ts))
            assert all(a > b for a, b in zip(ts, ts[1:])), "must be strictly descending"
            assert 0 <= ts[-1] and ts[0] <= 999
    assert SCH.timesteps(10, "trailing")[0] == 999
    assert SCH.timesteps(1000, "trailing") == list(range(999, -1, -1))


def test_first_order_solvers_agree():
    """DDIM (eta=0) == DPM-Solver-1 == DPM-Solver++-1 (all are the same exponential integrator)."""
    rng = np.random.default_rng(0)
    x_T = rng.standard_normal((64, 2))
    outs = [get_sampler(k, SCH, 25).sample(gaussian_eps, x_T.copy()) for k in ("ddim", "dpm1", "dpm++1")]
    for o in outs[1:]:
        assert np.allclose(o, outs[0], atol=1e-10)


def test_local_truncation_error_orders():
    """A k-th order multistep update with exact history has local error O(h^(k+1))."""
    rng = np.random.default_rng(1)
    x_T = rng.standard_normal((64, 2))
    for algorithm in ("dpmsolver", "dpmsolver++"):
        for order in (1, 2, 3):
            errs = [local_error(order, algorithm, h, x_T) for h in (0.1, 0.05, 0.025)]
            p = math.log2(errs[1] / errs[2])
            assert abs(p - (order + 1)) < 0.3, f"{algorithm} order {order}: errors {errs}, local order {p:.2f}"


def test_global_convergence_orders():
    rng = np.random.default_rng(1)
    x_T = rng.standard_normal((256, 2))
    expected = {"ddim": 1.0, "dpm1": 1.0, "dpm2m": 2.0, "dpm++2m": 2.0}
    for name, p in expected.items():
        errs = [solver_error(name, n, x_T) for n in (40, 80, 160)]
        orders = [math.log2(errs[i] / errs[i + 1]) for i in range(2)]
        assert abs(orders[-1] - p) < 0.2, f"{name}: errors {errs}, observed orders {orders}"
    # 3M is only *asymptotically* 2nd order globally: its first step has no history and is
    # 1st order, and that single O(h^2) local error eventually dominates. In the practical
    # NFE range it is still clearly the most accurate (next test).


def test_higher_order_is_more_accurate():
    rng = np.random.default_rng(2)
    x_T = rng.standard_normal((256, 2))
    for n in (20, 40):  # (at n=10, h ~ 0.7 is too large for 3M to beat 2M with DPM-Solver++)
        for alg in ("dpm", "dpm++"):
            e = {k: solver_error(k, n, x_T) for k in ("ddim", f"{alg}2m", f"{alg}3m")}
            assert e[f"{alg}3m"] < e[f"{alg}2m"] < e["ddim"], (n, e)


def test_ddpm_matches_data_distribution():
    """Full 1000-step ancestral DDPM with the exact eps should reproduce N(mu, s^2 I)."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((20000, 2))
    out = get_sampler("ddpm", SCH, 1000).sample(gaussian_eps, x, noise_fn=lambda z: rng.standard_normal(z.shape))
    assert np.allclose(out.mean(0), MU, atol=0.02), out.mean(0)
    assert np.allclose(out.std(0), S, atol=0.02), out.std(0)


def test_stochastic_needs_noise_fn():
    try:
        get_sampler("ddpm", SCH, 10).sample(gaussian_eps, np.zeros((1, 2)))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_gmm_exact_eps_matches_finite_difference_score():
    gmm = GaussianMixture.ring()
    rng = np.random.default_rng(4)
    x = rng.standard_normal((16, 2))
    a, s = 0.8, 0.6
    g = gmm.score(x, a, s)
    h = 1e-5
    fd = np.stack([(gmm.log_prob(x + h * e, a, s) - gmm.log_prob(x - h * e, a, s)) / (2 * h)
                   for e in np.eye(2)], 1)
    assert np.allclose(g, fd, atol=1e-5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
