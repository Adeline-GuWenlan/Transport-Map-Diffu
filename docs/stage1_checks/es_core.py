"""Energy-score / energy-distance estimators used by the stage-1 numerical checks.

Sign convention (matches docs/energy_score_loss.tex): negatively oriented energy score,
    S_beta(P, y) = E_P ||X - y||^beta - 1/2 E_P ||X - X'||^beta      (smaller is better),
expected score S_beta(P, Q) = E_{X~P, Y~Q} S_beta(P, Y), and energy distance
    D_beta(P, Q) = 2 E||X - Y||^beta - E||X - X'||^beta - E||Y - Y'||^beta,
so that S_beta(P, Q) - S_beta(Q, Q) = D_beta(P, Q) / 2.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

# Categorical palette (dataviz skill reference palette, light mode, fixed order).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#a3a29c"


# ----------------------------------------------------------------------------- numpy
def _as2d(a):
    a = np.asarray(a, dtype=np.float64)
    return a[:, None] if a.ndim == 1 else a


def _offdiag_mean(K):
    m = K.shape[0]
    return (K.sum() - np.trace(K)) / (m * (m - 1))


def energy_distance_u(x, y, beta=1.0):
    """Unbiased U-statistic for D_beta(P, Q); x ~ P (m, d), y ~ Q (n, d). Needs m, n >= 2."""
    x, y = _as2d(x), _as2d(y)
    cross = (cdist(x, y) ** beta).mean()
    return 2 * cross - _offdiag_mean(cdist(x, x) ** beta) - _offdiag_mean(cdist(y, y) ** beta)


def energy_distance_v(x, y, beta=1.0):
    """Biased V-statistic (keeps the zero diagonal); for comparison only."""
    x, y = _as2d(x), _as2d(y)
    return 2 * (cdist(x, y) ** beta).mean() - (cdist(x, x) ** beta).mean() - (cdist(y, y) ** beta).mean()


def es_loss_u(x, y, beta=1.0):
    """Unbiased estimator of the expected (negatively oriented) energy score S_beta(P, Q).
    x: m >= 2 samples of the model, y: n >= 1 data points."""
    x, y = _as2d(x), _as2d(y)
    return (cdist(x, y) ** beta).mean() - 0.5 * _offdiag_mean(cdist(x, x) ** beta)


# ----------------------------------------------------------------------------- torch
def torch_pow_dist(a, b, beta=1.0):
    """||a_i - b_j||^beta for a (n, d), b (m, d), with gradient 0 (a valid subgradient for
    beta = 1, the true gradient for beta > 1) wherever a_i == b_j. Avoids the NaN that
    sqrt / pow produce at 0 in backward ("double-where" trick)."""
    import torch
    diff = a[:, None, :] - b[None, :, :]
    sq = (diff * diff).sum(-1)
    pos = sq > 0
    sq_safe = torch.where(pos, sq, torch.ones_like(sq))
    return torch.where(pos, sq_safe ** (beta / 2), torch.zeros_like(sq))


def torch_es_loss_grouped(x, gx, y, gy, beta=1.0, interaction_weight=0.5, pool_interaction=False):
    """Unbiased minibatch estimator of the class-conditional expected energy score
        sum_c pi_c S_beta(P_theta(.|c), Q_c),
    batch = groups g; group g has n_g data points y (ids gy == g) and m_g >= 2 generated
    samples x (ids gx == g), all with the same class. Group weight n_g / N.
      * confinement:  mean over (i in g, j in g) of ||x_j - y_i||^beta
      * interaction: -interaction_weight * mean over (j != k, both in g) of ||x_j - x_k||^beta
    pool_interaction=True reproduces a common bug (interaction over all generated pairs,
    regardless of class) and is used only to demonstrate that bug.
    """
    import torch
    N = y.shape[0]
    Dxy = torch_pow_dist(x, y, beta)                    # (M, N)
    Dxx = torch_pow_dist(x, x, beta)                    # (M, M)
    same_xy = (gx[:, None] == gy[None, :]).to(x.dtype)
    eye = torch.eye(x.shape[0], dtype=x.dtype)
    loss = x.new_zeros(())
    for g in torch.unique(gy):
        mx = (gx == g)
        my = (gy == g)
        n_g, m_g = int(my.sum()), int(mx.sum())
        assert m_g >= 2, "each group needs m_g >= 2 generated samples"
        conf = Dxy[mx][:, my].mean()
        if pool_interaction:
            inter = None
        else:
            sub = Dxx[mx][:, mx]
            inter = (sub.sum() - torch.diagonal(sub).sum()) / (m_g * (m_g - 1))
        if inter is not None:
            loss = loss + (n_g / N) * (conf - interaction_weight * inter)
        else:
            loss = loss + (n_g / N) * conf
    if pool_interaction:
        M = x.shape[0]
        inter_all = (Dxx * (1 - eye)).sum() / (M * (M - 1))
        loss = loss - interaction_weight * inter_all
    return loss


# ----------------------------------------------------------------------------- closed forms
def folded_normal_mean(delta, s):
    """E|W| for W ~ N(delta, s^2)."""
    from scipy.stats import norm
    delta = np.asarray(delta, dtype=float)
    return s * np.sqrt(2 / np.pi) * np.exp(-delta ** 2 / (2 * s ** 2)) + delta * (1 - 2 * norm.cdf(-delta / s))


def energy_distance_gauss1d(mu1, s1, mu2, s2):
    """Closed-form D_1 between N(mu1, s1^2) and N(mu2, s2^2)."""
    exy = folded_normal_mean(mu1 - mu2, np.sqrt(s1 ** 2 + s2 ** 2))
    return 2 * exy - 2 * s1 / np.sqrt(np.pi) - 2 * s2 / np.sqrt(np.pi)


def fourier_constant(d, beta):
    """C(d, beta) with int_{R^d} (1 - cos<t,x>) / ||t||^{d+beta} dt = C(d, beta) ||x||^beta."""
    from scipy.special import gamma
    return 2 * np.pi ** (d / 2) * gamma(1 - beta / 2) / (beta * 2 ** beta * gamma((d + beta) / 2))


# ----------------------------------------------------------------------------- 2-D target
def eight_gaussians(n, rng, radius=2.0, std=0.1, return_labels=False):
    k = rng.integers(0, 8, size=n)
    ang = 2 * np.pi * k / 8
    centers = radius * np.stack([np.cos(ang), np.sin(ang)], 1)
    y = centers + std * rng.standard_normal((n, 2))
    return (y, k) if return_labels else y
