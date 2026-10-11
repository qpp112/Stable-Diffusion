# Samplers: the math behind `sdscratch/samplers`

Notation used throughout the code (`schedule.py`):

| symbol | meaning | code |
|---|---|---|
| $\bar\alpha_t$ | cumulative signal fraction, $\prod_{i\le t}(1-\beta_i)$ | `abar(t)` |
| $\alpha_t=\sqrt{\bar\alpha_t}$, $\sigma_t=\sqrt{1-\bar\alpha_t}$ | signal / noise scale | `alpha(t)`, `sigma(t)` |
| $\lambda_t=\log(\alpha_t/\sigma_t)$ | half log-SNR (increases as noise decreases) | `lam(t)` |
| $\hat\epsilon(x_t,t)$ | network's noise prediction | `eps_fn(x, t)` |
| $\hat x_0 = (x_t-\sigma_t\hat\epsilon)/\alpha_t$ | implied clean-image prediction | `eps_to_x0` |

Forward process: $x_t=\alpha_t x_0+\sigma_t\epsilon$. Stable Diffusion uses the *scaled-linear*
schedule $\beta_t = \big(\sqrt{0.00085}+\tfrac{t}{999}(\sqrt{0.012}-\sqrt{0.00085})\big)^2$, $T=1000$.
Every sampler steps from $t$ to $s<t$ and finishes with a jump to the clean endpoint
($\bar\alpha=1$, $\lambda=\infty$), whose output is $\hat x_0$.

## DDPM (Ho et al., 2020)

Sample from the Gaussian posterior $q(x_s\mid x_t, x_0=\hat x_0)$. With
$\alpha_{t|s}=\bar\alpha_t/\bar\alpha_s$ and $\beta_{t|s}=1-\alpha_{t|s}$:

$$
x_s=\underbrace{\frac{\sqrt{\bar\alpha_s}\,\beta_{t|s}}{1-\bar\alpha_t}\hat x_0+\frac{\sqrt{\alpha_{t|s}}(1-\bar\alpha_s)}{1-\bar\alpha_t}x_t}_{\text{posterior mean}}
+\sqrt{\tfrac{1-\bar\alpha_s}{1-\bar\alpha_t}\beta_{t|s}}\;z .
$$

With $s=t-1$ this is exactly Algorithm 2 of the paper; for skipped steps it is DDIM with $\eta=1$.

## DDIM (Song et al., 2021)

$$
x_s=\sqrt{\bar\alpha_s}\,\hat x_0+\sqrt{1-\bar\alpha_s-\sigma_\eta^2}\;\hat\epsilon+\sigma_\eta z,\qquad
\sigma_\eta=\eta\sqrt{\tfrac{1-\bar\alpha_s}{1-\bar\alpha_t}}\sqrt{1-\tfrac{\bar\alpha_t}{\bar\alpha_s}} .
$$

$\eta=0$ is deterministic and equals a first-order (Euler) step of the probability-flow ODE
written in $y=x/\alpha$, $\tilde\sigma=\sigma/\alpha$: $\;dy/d\tilde\sigma=\hat\epsilon$.

## DPM-Solver and DPM-Solver++ (Lu et al., 2022a,b)

Changing variables to $\lambda$ makes the linear part of the probability-flow ODE exactly solvable
(exponential integrator). With $h=\lambda_s-\lambda_t$:

$$
\textbf{noise form:}\quad x_s=\frac{\alpha_s}{\alpha_t}x_t-\alpha_s\!\int_{\lambda_t}^{\lambda_s}\! e^{-\lambda}\hat\epsilon\,d\lambda
\qquad
\textbf{data form (++):}\quad x_s=\frac{\sigma_s}{\sigma_t}x_t+\sigma_s\!\int_{\lambda_t}^{\lambda_s}\! e^{\lambda}\hat x_0\,d\lambda .
$$

Only the model output $m(\lambda)$ ($\hat\epsilon$ or $\hat x_0$) is approximated, by a Taylor
expansion $m(\lambda_t+u)\approx m_0+u\,m'+\tfrac{u^2}{2}m''$. The integrals of $e^{\pm u}u^k$ give the
$\varphi$-functions used in the code. For DPM-Solver++ ($\varphi_1=e^{-h}-1$):

$$
x_s=\frac{\sigma_s}{\sigma_t}x_t-\alpha_s\varphi_1\,m_0
+\alpha_s\Big(\frac{\varphi_1}{h}+1\Big)D_1
-\alpha_s\Big(\frac{\varphi_1+h}{h^2}-\frac12\Big)D_2,
\qquad D_1\approx h\,m',\; D_2\approx h^2 m'' .
$$

**Multistep** (1 network call per step): the derivatives come from the stored outputs of the previous
steps. With previous step sizes $h_0=r_0h$, $h_1=r_1h$:

$$
D_1^{(0)}=\frac{m_0-m_1}{r_0},\quad D_1^{(1)}=\frac{m_1-m_2}{r_1},\quad
D_1=D_1^{(0)}+\frac{r_0}{r_0+r_1}\big(D_1^{(0)}-D_1^{(1)}\big),\quad
D_2=\frac{2}{r_0+r_1}\big(D_1^{(0)}-D_1^{(1)}\big).
$$

Expanding $m_1,m_2$ around $\lambda_t$ confirms $D_1=h\,m'+O(h^3)$ and $D_2=h^2m''+O(h^3)$. Note the
factor **2** in $D_2$: with $D_2$ defined this way the coefficient is the familiar
$\varphi_3=(\varphi_1+h)/h^2-\tfrac12$; if you define $D_2$ without it (some implementations do), the
coefficient must be doubled instead. `tests/test_samplers.py::test_local_truncation_error_orders` checks
this numerically: a $k$-th order step from exact history has local error $O(h^{k+1})$ for $k=1,2,3$,
in both the noise and data forms. (For 2M the code uses the cheaper "midpoint" coefficient
$-\tfrac12\alpha_s\varphi_1D_1$, which agrees with $\alpha_s(\varphi_1/h+1)D_1$ to the order that matters.)

Practical details implemented in `DPMSolverMultistep`:
- the first step(s) fall back to lower order (no history yet) - this single 1st-order step is why
  3M is only *asymptotically* 2nd-order globally, as the tests show;
- the final step to $\sigma=0$ ($\lambda=\infty$, so $r_0=0$) is always first order and returns $\hat x_0$;
- for short schedules (<15 steps) the second-to-last step is capped at order 2.

## Timestep grids

The solver error is controlled by the step sizes **in $\lambda$**, not in $t$. On SD's schedule
$\lambda$ changes slowly near $t=999$ and very fast near $t=0$, so grids uniform in $t$
(`trailing`, `leading`, `linspace`) leave a few huge $\lambda$-steps at the low-noise end that barely
shrink as NFE grows. `karras` (Karras et al., 2022, $\rho=7$) and `logsnr` (uniform in $\lambda$)
spread the error evenly; the toy study (`assets/toy_convergence.png`, panel b) shows 3-16x lower error
at 10 NFE. These grids need fractional timesteps: $\log(\sigma/\alpha)$ is interpolated linearly between
integer steps and the UNet's sinusoidal embedding is evaluated at the fractional $t$ (as k-diffusion
does).

## Classifier-free guidance (Ho & Salimans, 2022)

$\hat\epsilon_w=\hat\epsilon(x,t,\varnothing)+w\,\big(\hat\epsilon(x,t,c)-\hat\epsilon(x,t,\varnothing)\big)$.
For $w>1$ this is no longer the score of any diffusion process and the guided ODE becomes stiff.
`assets/toy_guidance.png` shows the consequence with exact scores: for $w\gtrsim4$ and $\le$20 NFE,
first-order DDIM is *more* accurate than the 2nd/3rd-order multistep solvers, whose polynomial
extrapolation amplifies the rapidly changing guided prediction.

## References

- Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
- Song, Meng, Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021.
- Lu et al. *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps.* NeurIPS 2022.
- Lu et al. *DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models.* 2022.
- Karras, Aittala, Aila, Laine. *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS 2022.
- Lin, Liu, Li, Yang. *Common Diffusion Noise Schedules and Sample Steps are Flawed.* WACV 2024.
- Ho & Salimans. *Classifier-Free Diffusion Guidance.* 2022.
