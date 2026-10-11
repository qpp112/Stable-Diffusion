"""DDPM, DDIM and DPM-Solver / DPM-Solver++ samplers.

All samplers share one interface::

    sampler = get_sampler("dpm++2m", NoiseSchedule(), num_steps=20)
    x0 = sampler.sample(eps_fn, x_T, noise_fn=...)

* ``eps_fn(x, t) -> eps`` is the (classifier-free-guided) noise prediction at
  integer timestep ``t``.
* ``noise_fn(x) -> z`` returns standard normal noise shaped like ``x``; only the
  stochastic samplers (DDPM, DDIM with eta > 0) call it.

The update rules only use ``+``, ``-`` and multiplication by Python floats, so the
same code runs on ``torch.Tensor`` (real model) and ``numpy.ndarray`` (the toy
experiments and unit tests). Every sampler uses exactly one model evaluation (NFE)
per step.

Notation: alpha_t = sqrt(abar_t), sigma_t = sqrt(1 - abar_t),
lambda_t = log(alpha_t / sigma_t). A step goes from ``t`` to ``s`` (s < t, i.e.
less noise); the last step always targets the clean endpoint ``s = -1``.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from .schedule import CLEAN, NoiseSchedule, eps_to_x0

EpsFn = Callable[[object, int], object]
NoiseFn = Callable[[object], object]
StepCallback = Callable[[int, int, object, object], None]  # (step index, t, x_s, x0_pred)


class Sampler:
    name = "base"
    stochastic = False

    def __init__(self, schedule: NoiseSchedule, num_steps: int, spacing: str = "trailing"):
        self.schedule = schedule
        self.num_steps = num_steps
        self.timesteps = schedule.timesteps(num_steps, spacing)

    @property
    def nfe(self) -> int:
        return len(self.timesteps)

    def reset(self) -> None:
        """Clear any multistep history (called at the start of ``sample``)."""

    def step(self, x, eps, t: int, s: int, i: int, noise_fn: Optional[NoiseFn]):
        raise NotImplementedError

    def sample(self, eps_fn: EpsFn, x, noise_fn: Optional[NoiseFn] = None,
               callback: Optional[StepCallback] = None):
        if self.stochastic and noise_fn is None:
            raise ValueError(f"{self.name} is stochastic and needs a noise_fn")
        self.reset()
        ts = list(self.timesteps) + [CLEAN]
        for i in range(len(ts) - 1):
            t, s = ts[i], ts[i + 1]
            eps = eps_fn(x, t)
            if callback is not None:
                sch = self.schedule
                callback(i, t, x, eps_to_x0(x, eps, sch.alpha(t), sch.sigma(t)))
            x = self.step(x, eps, t, s, i, noise_fn)
        return x

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, steps={self.num_steps})"


# --------------------------------------------------------------------------- DDPM
class DDPMSampler(Sampler):
    """Ancestral sampling from the DDPM posterior q(x_s | x_t, x0_pred) (Ho et al., 2020).

    With all 1000 timesteps this is exactly Algorithm 2 of the DDPM paper. With fewer
    steps it uses the posterior between the two (non-adjacent) timesteps, which is the
    same as DDIM with eta = 1.
    """

    name = "ddpm"
    stochastic = True

    def step(self, x, eps, t, s, i, noise_fn):
        sch = self.schedule
        ab_t, ab_s = sch.abar(t), sch.abar(s)
        x0 = eps_to_x0(x, eps, math.sqrt(ab_t), math.sqrt(1.0 - ab_t))
        if s == CLEAN:
            return x0
        a_ts = ab_t / ab_s                      # alpha_{t|s}
        b_ts = 1.0 - a_ts                       # beta_{t|s}
        c_x0 = math.sqrt(ab_s) * b_ts / (1.0 - ab_t)
        c_xt = math.sqrt(a_ts) * (1.0 - ab_s) / (1.0 - ab_t)
        var = (1.0 - ab_s) / (1.0 - ab_t) * b_ts
        return x0 * c_x0 + x * c_xt + noise_fn(x) * math.sqrt(var)


# --------------------------------------------------------------------------- DDIM
class DDIMSampler(Sampler):
    """DDIM (Song et al., 2021). eta = 0 is deterministic (first-order ODE solver);
    eta = 1 recovers DDPM's posterior variance."""

    name = "ddim"

    def __init__(self, schedule, num_steps, spacing="trailing", eta: float = 0.0):
        super().__init__(schedule, num_steps, spacing)
        self.eta = float(eta)
        self.stochastic = self.eta > 0
        if self.eta > 0:
            self.name = f"ddim(eta={self.eta:g})"

    def step(self, x, eps, t, s, i, noise_fn):
        sch = self.schedule
        ab_t, ab_s = sch.abar(t), sch.abar(s)
        x0 = eps_to_x0(x, eps, math.sqrt(ab_t), math.sqrt(1.0 - ab_t))
        if s == CLEAN:
            return x0
        sig = self.eta * math.sqrt((1.0 - ab_s) / (1.0 - ab_t) * (1.0 - ab_t / ab_s))
        out = x0 * math.sqrt(ab_s) + eps * math.sqrt(max(1.0 - ab_s - sig**2, 0.0))
        if sig > 0:
            out = out + noise_fn(x) * sig
        return out


# ------------------------------------------------------------------- DPM-Solver
class DPMSolverMultistep(Sampler):
    """Multistep DPM-Solver (Lu et al., 2022a) and DPM-Solver++ (Lu et al., 2022b).

    Both integrate the probability-flow ODE in the half-log-SNR variable lambda with
    exponential integrators; the linear part is solved exactly and the network term is
    approximated by a Taylor expansion built from the previous ``order - 1`` model
    outputs (so the cost stays 1 NFE per step).

    * ``algorithm="dpmsolver++"`` expands the *data* prediction x0(x, lambda). Its
      updates are convex-combination-like in x0, which keeps it stable under large
      classifier-free guidance. This is the default in most SD UIs ("DPM++ 2M").
    * ``algorithm="dpmsolver"`` expands the *noise* prediction eps(x, lambda).

    The first step(s) use lower order because there is no history yet; the final step
    to sigma = 0 is always first order (``lambda = +inf`` there, so a higher-order
    finite difference is undefined) and ``lower_order_final`` also caps the
    second-to-last step at order 2 for short schedules, as in the reference code.
    """

    def __init__(self, schedule, num_steps, spacing="trailing", order: int = 2,
                 algorithm: str = "dpmsolver++", lower_order_final: bool = True):
        super().__init__(schedule, num_steps, spacing)
        if order not in (1, 2, 3):
            raise ValueError("order must be 1, 2 or 3")
        if algorithm not in ("dpmsolver", "dpmsolver++"):
            raise ValueError("algorithm must be 'dpmsolver' or 'dpmsolver++'")
        self.order = order
        self.algorithm = algorithm
        self.lower_order_final = lower_order_final
        prefix = "dpm++" if algorithm == "dpmsolver++" else "dpm"
        self.name = f"{prefix}{order}m" if order > 1 else f"{prefix}1"
        self.reset()

    def reset(self):
        self._outs: list = []      # last few model outputs (x0 for ++, eps otherwise)
        self._lams: list[float] = []

    def step(self, x, eps, t, s, i, noise_fn):
        sch = self.schedule
        a_t, sg_t = sch.alpha(t), sch.sigma(t)
        x0 = eps_to_x0(x, eps, a_t, sg_t)
        if s == CLEAN:
            # Both parameterisations give exactly x0_pred for a first-order step to sigma=0.
            return x0

        m = x0 if self.algorithm == "dpmsolver++" else eps
        self._outs = (self._outs + [m])[-3:]
        self._lams = (self._lams + [sch.lam(t)])[-3:]

        n = len(self.timesteps)
        order = min(self.order, len(self._outs))
        if self.lower_order_final and n < 15 and i == n - 2:
            order = min(order, 2)

        a_s, sg_s = sch.alpha(s), sch.sigma(s)
        lam_t, lam_s = self._lams[-1], sch.lam(s)
        h = lam_s - lam_t

        # Taylor-expand m(lambda) around lambda_t using backward differences of the stored
        # outputs (variable step sizes h0 = r0*h, h1 = r1*h). With
        #   m(lambda_t + u) ~ m0 + u m' + u^2/2 m'',  D1 ~ h m',  D2 ~ h^2 m''
        # the exponential integrals give the phi_k coefficients below
        # (see docs/samplers.md for the derivation).
        m0 = self._outs[-1]
        D1 = D2 = None
        if order >= 2:
            r0 = (lam_t - self._lams[-2]) / h
            D1_0 = (m0 - self._outs[-2]) * (1.0 / r0)
            D1 = D1_0
        if order == 3:
            r1 = (self._lams[-2] - self._lams[-3]) / h
            D1_1 = (self._outs[-2] - self._outs[-3]) * (1.0 / r1)
            D1 = D1_0 + (D1_0 - D1_1) * (r0 / (r0 + r1))      # ~ h   * dm/dlambda
            D2 = (D1_0 - D1_1) * (2.0 / (r0 + r1))           # ~ h^2 * d2m/dlambda2

        if self.algorithm == "dpmsolver++":
            phi = math.expm1(-h)                      # e^{-h} - 1  (negative)
            out = x * (sg_s / sg_t) - m0 * (a_s * phi)
            if order == 2:
                out = out - D1 * (0.5 * a_s * phi)
            elif order == 3:
                out = out + D1 * (a_s * (phi / h + 1.0)) - D2 * (a_s * ((phi + h) / h**2 - 0.5))
        else:
            phi = math.expm1(h)                       # e^{h} - 1  (positive)
            out = x * (a_s / a_t) - m0 * (sg_s * phi)
            if order == 2:
                out = out - D1 * (0.5 * sg_s * phi)
            elif order == 3:
                out = out - D1 * (sg_s * (phi / h - 1.0)) - D2 * (sg_s * ((phi - h) / h**2 - 0.5))
        return out


# --------------------------------------------------------------------- registry
SAMPLERS = {
    "ddpm": lambda sch, n, **kw: DDPMSampler(sch, n, **kw),
    "ddim": lambda sch, n, **kw: DDIMSampler(sch, n, **kw),
    "dpm1": lambda sch, n, **kw: DPMSolverMultistep(sch, n, order=1, algorithm="dpmsolver", **kw),
    "dpm2m": lambda sch, n, **kw: DPMSolverMultistep(sch, n, order=2, algorithm="dpmsolver", **kw),
    "dpm3m": lambda sch, n, **kw: DPMSolverMultistep(sch, n, order=3, algorithm="dpmsolver", **kw),
    "dpm++1": lambda sch, n, **kw: DPMSolverMultistep(sch, n, order=1, **kw),
    "dpm++2m": lambda sch, n, **kw: DPMSolverMultistep(sch, n, order=2, **kw),
    "dpm++3m": lambda sch, n, **kw: DPMSolverMultistep(sch, n, order=3, **kw),
}


def get_sampler(name: str, schedule: NoiseSchedule, num_steps: int, **kwargs) -> Sampler:
    key = name.lower()
    if key not in SAMPLERS:
        raise ValueError(f"unknown sampler {name!r}; choose from {sorted(SAMPLERS)}")
    return SAMPLERS[key](schedule, num_steps, **kwargs)
