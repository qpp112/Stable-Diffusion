"""Analytic toy diffusion: a 2D Gaussian mixture whose noisy score is known exactly.

For data x0 ~ sum_k w_k N(mu_k, s_k^2 I), the marginal at noise level t is again a
mixture,  p_t(x) = sum_k w_k N(alpha_t mu_k, (alpha_t^2 s_k^2 + sigma_t^2) I),
so the optimal noise predictor  eps*(x, t) = -sigma_t * grad log p_t(x)  is available in
closed form. Plugging eps* into a sampler isolates *solver* error from *network* error,
which is what the sampler benchmarks in ``scripts/toy_gmm.py`` measure.

Pure NumPy - no torch required.
"""

from __future__ import annotations

import numpy as np

from .samplers.schedule import NoiseSchedule


class GaussianMixture:
    def __init__(self, means, stds, weights=None):
        self.means = np.asarray(means, dtype=np.float64)          # (K, d)
        k, self.dim = self.means.shape
        self.stds = np.broadcast_to(np.asarray(stds, dtype=np.float64), (k,)).copy()
        w = np.ones(k) if weights is None else np.asarray(weights, dtype=np.float64)
        self.weights = w / w.sum()

    @classmethod
    def ring(cls, n_modes=8, radius=1.5, std=0.08):
        ang = 2 * np.pi * np.arange(n_modes) / n_modes
        return cls(np.stack([radius * np.cos(ang), radius * np.sin(ang)], 1), std)

    def subset(self, idx):
        idx = np.asarray(idx)
        return GaussianMixture(self.means[idx], self.stds[idx], self.weights[idx])

    def sample(self, n, rng):
        comp = rng.choice(len(self.weights), size=n, p=self.weights)
        return self.means[comp] + self.stds[comp, None] * rng.standard_normal((n, self.dim))

    def _log_resp_and_var(self, x, alpha, sigma):
        var = alpha**2 * self.stds**2 + sigma**2                     # (K,)
        diff = x[:, None, :] - alpha * self.means[None]               # (B, K, d)
        logp = (np.log(self.weights) - 0.5 * self.dim * np.log(2 * np.pi * var)
                - 0.5 * (diff**2).sum(-1) / var)                      # (B, K)
        return logp, var, diff

    def log_prob(self, x, alpha=1.0, sigma=0.0):
        logp, _, _ = self._log_resp_and_var(x, alpha, sigma)
        m = logp.max(1, keepdims=True)
        return (m + np.log(np.exp(logp - m).sum(1, keepdims=True)))[:, 0]

    def score(self, x, alpha, sigma):
        logp, var, diff = self._log_resp_and_var(x, alpha, sigma)
        logp -= logp.max(1, keepdims=True)
        r = np.exp(logp)
        r /= r.sum(1, keepdims=True)                                  # responsibilities
        return -(r[..., None] * diff / var[None, :, None]).sum(1)

    def eps_fn(self, schedule: NoiseSchedule):
        """Exact epsilon-prediction  eps*(x, t) = -sigma_t * score_t(x)."""
        def eps(x, t):
            a, s = schedule.alpha(t), schedule.sigma(t)
            return -s * self.score(x, a, s)
        return eps


def guided_eps_fn(uncond: GaussianMixture, cond: GaussianMixture, schedule: NoiseSchedule, w: float):
    """Classifier-free guidance with exact scores: eps_u + w * (eps_c - eps_u)."""
    eu, ec = uncond.eps_fn(schedule), cond.eps_fn(schedule)

    def eps(x, t):
        u = eu(x, t)
        return u + w * (ec(x, t) - u)
    return eps


def sliced_wasserstein(a, b, n_proj=256, rng=None):
    """Sliced 2-Wasserstein distance between two equally sized point clouds."""
    rng = np.random.default_rng(0) if rng is None else rng
    d = a.shape[1]
    proj = rng.standard_normal((d, n_proj))
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)
    pa, pb = np.sort(a @ proj, 0), np.sort(b @ proj, 0)
    return float(np.sqrt(((pa - pb) ** 2).mean()))
