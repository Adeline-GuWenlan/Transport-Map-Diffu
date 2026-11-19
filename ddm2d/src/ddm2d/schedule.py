"""Noise schedule, forward process, Gaussian bridge (Eq. 4) and sigmoid weighting.

Paper: arXiv 2502.02483 v3, Eqs. (2)-(4) and App. G. We use r_{i,j}(s,t) = (alpha_t/alpha_s)^i (sigma_s/sigma_t)^j, the App. G form; the
main-text form with (sigma_s^2/sigma_t^2)^j does not preserve the marginals (docs/audit.md section 1.2).
"""
from __future__ import annotations

import torch


def alpha_sigma(t):
    """Flow-matching schedule, Eq. (3): alpha_t = 1 - t, sigma_t = t."""
    return 1.0 - t, t


def _bc(v, ref):
    """Broadcast a per-sample scalar tensor (shape [n] or []) against ref of shape [n, ..., d]."""
    if not torch.is_tensor(v):
        v = torch.as_tensor(v, dtype=ref.dtype, device=ref.device)
    while v.ndim < ref.ndim:
        v = v.unsqueeze(-1)
    return v


def forward_sample(x0, t, noise):
    """x_t = alpha_t x_0 + sigma_t z, Eq. (2). t has shape [n] or []."""
    a, s = alpha_sigma(t)
    return _bc(a, x0) * x0 + _bc(s, x0) * noise


def bridge_coeffs(s, t, churn):
    """Coefficients (c_xt, c_x0, std) of p(x_s | x_0, x_t) = N(c_xt x_t + c_x0 x_0, std^2 I), Eq. (4), for 0 <= s < t <= 1.

    s, t: python floats or tensors (broadcastable). churn: epsilon in [0, 1] (0 = DDIM, 1 = DDPM).
    """
    a_s, sg_s = alpha_sigma(s)
    a_t, sg_t = alpha_sigma(t)
    e2 = churn ** 2
    ra = a_t / a_s                     # alpha_t / alpha_s (alpha_s > 0 since s < 1)
    rs = sg_s / sg_t                   # sigma_s / sigma_t (sigma_t > 0 since t > 0)
    r01 = rs
    r11 = ra * rs
    r12 = ra * rs ** 2
    r22 = ra ** 2 * rs ** 2
    c_xt = e2 * r12 + (1.0 - e2) * r01
    c_x0 = a_s * (1.0 - e2 * r22 - (1.0 - e2) * r11)
    inner = e2 * r11 + (1.0 - e2)
    var = sg_s ** 2 * (1.0 - inner ** 2)
    if torch.is_tensor(var):
        var = var.clamp(min=0.0)
        std = var.sqrt()
    else:
        std = max(var, 0.0) ** 0.5
    return c_xt, c_x0, std


def bridge_mean_std(s, t, x0, xt, churn):
    """mu_{s,t}(x0, xt) and Sigma_{s,t}^{1/2} (a scalar std; Sigma = std^2 I), Eq. (4)."""
    c_xt, c_x0, std = bridge_coeffs(s, t, churn)
    mu = c_xt * xt + c_x0 * x0
    return mu, std


def sigmoid_weight(t, bias: float = 0.0):
    """w_t = (1 + exp[b - log(alpha_t^2 / sigma_t^2)])^{-1} = sigmoid(logSNR_t - b) (Sec. 4.2; Kingma et al. 2021)."""
    a, s = alpha_sigma(t)
    log_snr = 2.0 * (torch.log(a) - torch.log(s))
    return torch.sigmoid(log_snr - bias)


def sample_train_times(n, safety, generator=None, device=None, dtype=torch.float32):
    """t ~ U[safety, 1 - safety] (Alg. 1 restricted to the safety interval [eta, 1 - eta], App. K.2)."""
    u = torch.rand(n, generator=generator, device=device, dtype=dtype)
    return safety + (1.0 - 2.0 * safety) * u


def sampling_grid(n_steps: int):
    """Uniform grid t_k = k / N, k = 0..N (Algorithm 2: t_0 = 0 < ... < t_N = 1)."""
    return [k / n_steps for k in range(n_steps + 1)]
