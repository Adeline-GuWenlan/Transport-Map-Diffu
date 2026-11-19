"""Samplers. Algorithm 2 (arXiv 2502.02483 v3, p. 4) for the distributional arms, and the same Eq. (4) bridge with a deterministic
x_hat(t, x_t) for the standard arm (DDIM/DDPM family; churn 1 = the "SDE sampler" of App. I.2, see App. G.1).

Grid: uniform t_k = k/N (Algorithm 2: t_0 = 0, t_N = 1). The denoiser is called at clamp(t_{k+1}, safety, 1 - safety) (safety interval of
App. K.2); the bridge uses the true grid times. The last step goes to s = 0, where Sigma = 0 and mu = x_hat, so the output is the last
prediction. One NFE = one denoiser call; N steps = N NFE for every arm (docs/plan.md).

Caveat for N > 1/safety (e.g. NFE 1000): grid times below `safety` are clamped up to `safety`, so the denoiser is asked for the posterior at
the wrong noise level. With an exact-posterior oracle this biases the output (tests/test_samplers.py::test_low_clamp_bias_at_fine_grids).
`clamp_low=False` passes the true t below `safety` instead (the network then extrapolates below its training range). For N <= 1/safety the
two options are identical except at t_N = 1. Evaluate NFE 1000 both ways, identically for every arm.
"""
from __future__ import annotations

import torch

from .schedule import bridge_coeffs, sampling_grid


@torch.no_grad()
def denoise(model, t_scalar: float, x, generator=None):
    """One denoiser call on a batch x [n, d] at time t_scalar. Draws a fresh xi for the stochastic model."""
    t = float(t_scalar)                     # shared by all rows: the model evaluates its time path once
    if model.stochastic:
        xi = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
        return model(t, x, xi)
    return model(t, x)


@torch.no_grad()
def sample(model, n: int, n_steps: int, churn: float = 1.0, safety: float = 0.01, generator=None, dim: int = 2,
           dtype=torch.float32, return_nfe: bool = False, chunk: int | None = None, clamp_low: bool = True):
    """Generate n samples with n_steps denoiser calls. Uses `generator` for X_1, xi and Z (reproducible)."""
    model.eval()
    grid = sampling_grid(n_steps)
    x = torch.randn((n, dim), generator=generator, dtype=dtype)
    nfe = 0
    for k in range(n_steps - 1, -1, -1):
        s, t = grid[k], grid[k + 1]
        t_in = min(max(t, safety) if clamp_low else t, 1.0 - safety)
        if chunk is None:
            x0hat = denoise(model, t_in, x, generator)
        else:
            x0hat = torch.cat([denoise(model, t_in, x[i:i + chunk], generator) for i in range(0, n, chunk)])
        nfe += 1
        c_xt, c_x0, std = bridge_coeffs(s, t, churn)
        z = torch.randn(x.shape, generator=generator, dtype=dtype)
        x = c_xt * x + c_x0 * x0hat + std * z
    return (x, nfe) if return_nfe else x


@torch.no_grad()
def posterior_draws(model, t_scalar: float, xt, n_draws: int, generator=None):
    """n_draws independent predictions x_hat(t, x_t, xi) for each row of xt [n, d] -> [n, n_draws, d].
    For the standard model all draws are identical (deterministic predictor)."""
    model.eval()
    n, d = xt.shape
    t = float(t_scalar)
    if model.stochastic:
        xi = torch.randn((n, n_draws, d), generator=generator, dtype=xt.dtype)
        return model(t, xt, xi)
    return model(t, xt)[:, None, :].expand(n, n_draws, d)
