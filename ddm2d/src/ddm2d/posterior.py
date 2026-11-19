"""Exact posterior p(x_0 | x_t) of the GMM under the flow-matching forward process (App. I.1 of arXiv 2502.02483 v3, p. 33):
  p(x0 | xt) = sum_k w_k N(x0; nu_k, Sigma),  Sigma = (alpha_t^2/sigma_t^2 + 1/s^2)^{-1} I,
  nu_k = Sigma (alpha_t/sigma_t^2 x_t + mu_k / s^2),  w_k propto pi_k N(x_t; alpha_t mu_k, (alpha_t^2 s^2 + sigma_t^2) I).
"""
from __future__ import annotations

import numpy as np

from .data import GMM_MEANS, GMM_STD, GMM_WEIGHTS


def posterior_params(xt: np.ndarray, t: float, means=GMM_MEANS, s=GMM_STD, pis=GMM_WEIGHTS):
    """xt: [n, d]. Returns weights [n, K], component means [n, K, d], scalar component variance."""
    a, sg = 1.0 - t, t
    xt = np.asarray(xt, dtype=np.float64)
    var = 1.0 / (a ** 2 / sg ** 2 + 1.0 / s ** 2)
    nu = var * (a / sg ** 2 * xt[:, None, :] + means[None, :, :] / s ** 2)
    marg_var = a ** 2 * s ** 2 + sg ** 2
    logw = np.log(pis)[None, :] - ((xt[:, None, :] - a * means[None, :, :]) ** 2).sum(-1) / (2 * marg_var)
    logw -= logw.max(1, keepdims=True)
    w = np.exp(logw)
    w /= w.sum(1, keepdims=True)
    return w, nu, var


def sample_posterior(xt: np.ndarray, t: float, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """n_draws exact posterior samples per row of xt -> [n, n_draws, d]."""
    w, nu, var = posterior_params(xt, t)
    n, K, d = nu.shape
    u = rng.random((n, n_draws))
    comp = (u[:, :, None] > np.cumsum(w, 1)[:, None, :]).sum(-1).clip(max=K - 1)   # inverse-CDF component choice
    mean = np.take_along_axis(nu, comp[:, :, None].repeat(d, 2), axis=1)
    return mean + np.sqrt(var) * rng.standard_normal((n, n_draws, d))


def posterior_mean_cov(xt: np.ndarray, t: float):
    """Exact posterior mean [n, d] and per-coordinate variance [n, d]."""
    w, nu, var = posterior_params(xt, t)
    mean = (w[:, :, None] * nu).sum(1)
    second = (w[:, :, None] * (nu ** 2 + var)).sum(1)
    return mean, second - mean ** 2
