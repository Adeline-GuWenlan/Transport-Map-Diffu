"""Exact posterior p(x_0 | x_t) for an isotropic GMM with per-component weights pi_k, means mu_k and stds s_k (ddm2dx, E2).

Forward process x_t = alpha x_0 + sigma z, alpha = 1 - t, sigma = t (flow matching, as in schedule.py). Derivation:
  p(x_0 | x_t) ∝ sum_k pi_k N(x_0; mu_k, s_k^2 I) N(x_t; alpha x_0, sigma^2 I).
For component k the product of the two Gaussians in x_0 is Gaussian with
  precision 1/v_k = alpha^2/sigma^2 + 1/s_k^2,   mean m_k = v_k (alpha x_t / sigma^2 + mu_k / s_k^2),
and its integral over x_0 is the marginal N(x_t; alpha mu_k, (alpha^2 s_k^2 + sigma^2) I). Hence
  p(x_0 | x_t) = sum_k r_k(x_t) N(x_0; m_k, v_k I),   r_k(x_t) ∝ pi_k N(x_t; alpha mu_k, (alpha^2 s_k^2 + sigma^2) I).
With equal s_k this reduces to posterior.py (App. I.1 of arXiv 2502.02483 v3); tests/test_ext_posterior_gmm.py checks that and checks
the formula against self-normalised importance sampling. Note the normalising constant of N(x_t; ., (alpha^2 s_k^2 + sigma^2) I) depends on
k when the s_k differ, so it is kept (posterior.py can drop it because its s is shared).
"""
from __future__ import annotations

import numpy as np

from .data import GMM_SPECS


def spec(name: str):
    """(means [K, d], stds [K], weights [K]) of a named GMM dataset ("gmm", "gmm2w", "gmm9w")."""
    return GMM_SPECS[name]


def posterior_params(xt: np.ndarray, t: float, means, stds, weights):
    """xt: [n, d]. Returns responsibilities r [n, K], component means m [n, K, d], component variances v [K]."""
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    a, sg = 1.0 - t, t
    xt = np.asarray(xt, dtype=np.float64)
    d = xt.shape[1]
    s2 = stds ** 2
    v = 1.0 / (a ** 2 / sg ** 2 + 1.0 / s2)                                              # [K]
    m = v[None, :, None] * (a / sg ** 2 * xt[:, None, :] + means[None, :, :] / s2[None, :, None])
    marg = a ** 2 * s2 + sg ** 2                                                         # [K]
    logr = (np.log(weights)[None, :] - 0.5 * d * np.log(2 * np.pi * marg)[None, :]
            - ((xt[:, None, :] - a * means[None, :, :]) ** 2).sum(-1) / (2 * marg[None, :]))
    logr -= logr.max(1, keepdims=True)
    r = np.exp(logr)
    r /= r.sum(1, keepdims=True)
    return r, m, v


def responsibilities(xt: np.ndarray, t: float, means, stds, weights) -> np.ndarray:
    return posterior_params(xt, t, means, stds, weights)[0]


def sample_posterior(xt: np.ndarray, t: float, n_draws: int, rng: np.random.Generator, means, stds, weights,
                     return_comp: bool = False):
    """n_draws exact posterior draws per row of xt -> [n, n_draws, d] (and the component index [n, n_draws] if return_comp)."""
    r, m, v = posterior_params(xt, t, means, stds, weights)
    n, K, d = m.shape
    u = rng.random((n, n_draws))
    comp = (u[:, :, None] > np.cumsum(r, 1)[:, None, :]).sum(-1).clip(max=K - 1)
    mean = np.take_along_axis(m, comp[:, :, None].repeat(d, 2), axis=1)
    x = mean + np.sqrt(v[comp])[:, :, None] * rng.standard_normal((n, n_draws, d))
    return (x, comp) if return_comp else x


def posterior_mean_cov(xt: np.ndarray, t: float, means, stds, weights):
    """Exact posterior mean [n, d] and per-coordinate variance [n, d]."""
    r, m, v = posterior_params(xt, t, means, stds, weights)
    mean = (r[:, :, None] * m).sum(1)
    second = (r[:, :, None] * (m ** 2 + v[None, :, None])).sum(1)
    return mean, second - mean ** 2


def assign_components(x: np.ndarray, means, stds, weights) -> np.ndarray:
    """Component with the largest pi_k N(x; mu_k, s_k^2 I) (the t -> 0 responsibility), for mode-weight evaluation."""
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[1]
    logp = (np.log(weights)[None, :] - d * np.log(stds)[None, :]
            - ((x[:, None, :] - means[None, :, :]) ** 2).sum(-1) / (2 * stds[None, :] ** 2))
    return logp.argmax(1)
