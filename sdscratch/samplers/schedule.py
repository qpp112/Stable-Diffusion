"""Discrete variance-preserving (VP) noise schedule used by Stable Diffusion.

Forward process (Ho et al., 2020):

    q(x_t | x_0) = N( sqrt(abar_t) * x_0,  (1 - abar_t) * I ),   t = 0 .. T-1

We write alpha_t = sqrt(abar_t), sigma_t = sqrt(1 - abar_t) and the half-log-SNR
lambda_t = log(alpha_t / sigma_t), which is the natural time variable for DPM-Solver.

The special timestep ``t = -1`` denotes the clean data (abar = 1, sigma = 0,
lambda = +inf). Every sampler in this package ends by stepping to t = -1.

Everything here is plain NumPy / Python floats so the samplers work on any array
type that supports ``+``, ``-`` and ``*`` with Python scalars (torch tensors, numpy
arrays, ...).
"""

from __future__ import annotations

import math

import numpy as np

CLEAN = -1  # sentinel timestep for "x_0"


class NoiseSchedule:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        kind: str = "scaled_linear",
    ):
        n = num_train_timesteps
        if kind == "scaled_linear":  # SD 1.x / SDXL
            betas = np.linspace(beta_start**0.5, beta_end**0.5, n, dtype=np.float64) ** 2
        elif kind == "linear":  # original DDPM
            betas = np.linspace(beta_start, beta_end, n, dtype=np.float64)
        elif kind == "cosine":  # Nichol & Dhariwal, 2021
            s = 0.008
            f = lambda u: math.cos((u + s) / (1 + s) * math.pi / 2) ** 2  # noqa: E731
            betas = np.array([min(1 - f((i + 1) / n) / f(i / n), 0.999) for i in range(n)])
        else:
            raise ValueError(f"unknown schedule kind {kind!r}")
        self.num_train_timesteps = n
        self.kind = kind
        self.betas = betas
        self.alphas_cumprod = np.cumprod(1.0 - betas)
        ab = self.alphas_cumprod
        self._ls = 0.5 * (np.log1p(-ab) - np.log(ab))
        self._t_grid = np.arange(n, dtype=np.float64)

    # ------------------------------------------------------------------ scalars
    # Timesteps may be fractional (Karras / log-SNR spacing). Between integer steps we
    # interpolate log(sigma_t / alpha_t) linearly in t, as k-diffusion does; the UNet's
    # sinusoidal timestep embedding accepts fractional t as well.
    def abar(self, t: float) -> float:
        if t < 0:
            return 1.0
        if float(t).is_integer():
            return float(self.alphas_cumprod[int(t)])
        log_s = float(np.interp(t, self._t_grid, self._ls))
        return 1.0 / (1.0 + math.exp(2.0 * log_s))

    def alpha(self, t: float) -> float:
        return math.sqrt(self.abar(t))

    def sigma(self, t: float) -> float:
        return math.sqrt(max(1.0 - self.abar(t), 0.0))

    def lam(self, t: float) -> float:
        """Half log-SNR lambda_t = log(alpha_t / sigma_t); +inf at the clean endpoint."""
        if t < 0:
            return math.inf
        ab = self.abar(t)
        return 0.5 * (math.log(ab) - math.log1p(-ab))

    @property
    def _log_sigmas(self) -> np.ndarray:
        """log of sigma_t / alpha_t (the "EDM sigma") at each integer timestep; increasing in t."""
        return self._ls

    def sigma_to_t(self, edm_sigma) -> np.ndarray:
        """Inverse of t -> sigma_t/alpha_t, interpolated (fractional timesteps)."""
        return np.interp(np.log(edm_sigma), self._ls, self._t_grid)

    # --------------------------------------------------------------- timesteps
    def timesteps(self, num_steps: int, spacing: str = "trailing") -> list:
        """Descending list of ``num_steps`` timesteps to evaluate the model at.

        Integer spacings (the model only saw integer t during training):
          * ``trailing`` - starts exactly at T-1 (Lin et al., 2023: "Common Diffusion
            Noise Schedules and Sample Steps are Flawed"). Default.
          * ``leading``  - classic SD/DDIM spacing ``[0, k, 2k, ...] + 1``.
          * ``linspace`` - evenly spaced on [0, T-1], rounded.
        Fractional spacings (uniform in a noise-level variable rather than in t):
          * ``karras``   - Karras et al. (2022) rho=7 sigma ramp ("DPM++ 2M Karras").
          * ``logsnr``   - uniform in lambda = log(alpha/sigma), the variable DPM-Solver
            integrates in; gives clean asymptotic convergence orders.
        """
        n_train = self.num_train_timesteps
        if not 1 <= num_steps <= n_train:
            raise ValueError(f"num_steps must be in [1, {n_train}], got {num_steps}")
        if spacing in ("karras", "logsnr"):
            ls = self._log_sigmas
            s_min, s_max = math.exp(ls[0]), math.exp(ls[-1])
            ramp = np.linspace(0.0, 1.0, num_steps)
            if spacing == "karras":
                rho = 7.0
                sig = (s_max ** (1 / rho) + ramp * (s_min ** (1 / rho) - s_max ** (1 / rho))) ** rho
            else:
                sig = np.exp(ls[-1] + ramp * (ls[0] - ls[-1]))
            ts = np.clip(self.sigma_to_t(sig), 0.0, n_train - 1)
            return [float(t) for t in ts]
        if spacing == "trailing":
            ts = np.round(np.arange(n_train, 0, -n_train / num_steps)).astype(np.int64) - 1
        elif spacing == "leading":
            k = n_train // num_steps
            offset = 1 if (num_steps - 1) * k + 1 <= n_train - 1 else 0  # SD's steps_offset=1
            ts = (np.arange(num_steps) * k + offset)[::-1]
        elif spacing == "linspace":
            ts = np.round(np.linspace(0, n_train - 1, num_steps))[::-1].astype(np.int64)
        else:
            raise ValueError(f"unknown spacing {spacing!r}")
        ts = np.clip(ts, 0, n_train - 1)
        out: list = []
        for t in ts.tolist():  # deduplicate, keep descending order
            if not out or t < out[-1]:
                out.append(int(t))
        return out


def eps_to_x0(x, eps, alpha: float, sigma: float):
    """x_0 prediction from an epsilon prediction:  x0 = (x_t - sigma_t * eps) / alpha_t."""
    return (x - sigma * eps) * (1.0 / alpha)


def to_eps(model_out, x, alpha: float, sigma: float, prediction_type: str = "epsilon"):
    """Convert a network output to an epsilon prediction (VP: alpha^2 + sigma^2 = 1)."""
    if prediction_type == "epsilon":
        return model_out
    if prediction_type == "v_prediction":  # v = alpha * eps - sigma * x0  =>  eps = sigma * x + alpha * v
        return x * sigma + model_out * alpha
    if prediction_type == "sample":  # model predicts x0
        return (x - model_out * alpha) * (1.0 / sigma)
    raise ValueError(f"unknown prediction_type {prediction_type!r}")
