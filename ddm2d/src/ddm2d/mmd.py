"""RBF-kernel squared MMD (Eqs. 9b, 10 of arXiv 2502.02483 v3), chunked, accumulated in float64.

Conventions (docs/plan.md, "MMD estimator"):
  mmd2_u : unbiased U-statistic, mean of k(x_i, x_j) over i != j, same for y, minus 2 mean k(x_i, y_j)   (primary)
  mmd2_b : biased V-statistic, diagonals included
  d_rho  : the paper's D_rho(p, q) with rho = -k (Eq. 10) = mmd2 / 2 (kernel score carries a factor 1/2)
"""
from __future__ import annotations

import numpy as np
import torch


def _kernel_sum(a, b, sigma, chunk):
    """sum_{i,j} exp(-||a_i - b_j||^2 / (2 sigma^2)), float64."""
    total = torch.zeros((), dtype=torch.float64)
    for i in range(0, a.shape[0], chunk):
        ai = a[i:i + chunk]
        for j in range(0, b.shape[0], chunk):
            bj = b[j:j + chunk]
            d2 = ((ai[:, None, :] - bj[None, :, :]) ** 2).sum(-1)
            total = total + torch.exp(-d2 / (2.0 * sigma ** 2)).sum()
    return total


def rbf_mmd2(x, y, sigma: float = 1.0, chunk: int = 2048) -> dict:
    """Returns dict with mmd2_u, mmd2_b, d_rho_u, d_rho_b and the three kernel means. x: [n, d], y: [m, d]."""
    x = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(y), dtype=torch.float64)
    n, m = x.shape[0], y.shape[0]
    sxx = _kernel_sum(x, x, sigma, chunk)
    syy = _kernel_sum(y, y, sigma, chunk)
    sxy = _kernel_sum(x, y, sigma, chunk)
    # k(x, x) = 1 exactly on the diagonal
    kxx_u = (sxx - n) / (n * (n - 1))
    kyy_u = (syy - m) / (m * (m - 1))
    kxx_b, kyy_b, kxy = sxx / n ** 2, syy / m ** 2, sxy / (n * m)
    mmd2_u = float(kxx_u + kyy_u - 2 * kxy)
    mmd2_b = float(kxx_b + kyy_b - 2 * kxy)
    return {"mmd2_u": mmd2_u, "mmd2_b": mmd2_b, "d_rho_u": mmd2_u / 2, "d_rho_b": mmd2_b / 2,
            "kxx_u": float(kxx_u), "kyy_u": float(kyy_u), "kxy": float(kxy), "n": n, "m": m, "sigma": sigma}


def gaussian_mmd2_closed_form(mu1, s1, mu2, s2, sigma: float = 1.0) -> float:
    """Population MMD^2 between N(mu1, s1^2 I) and N(mu2, s2^2 I) in R^d under k = exp(-||x-y||^2/(2 sigma^2)).
    E k(X, Y) for X ~ N(a, A I), Y ~ N(b, B I): (sigma^2/(sigma^2 + A + B))^{d/2} exp(-||a-b||^2 / (2 (sigma^2 + A + B)))."""
    mu1, mu2 = np.asarray(mu1, float), np.asarray(mu2, float)
    d = mu1.shape[0]

    def ek(a, A, b, B):
        v = sigma ** 2 + A + B
        return (sigma ** 2 / v) ** (d / 2) * np.exp(-np.sum((a - b) ** 2) / (2 * v))

    return float(ek(mu1, s1 ** 2, mu1, s1 ** 2) + ek(mu2, s2 ** 2, mu2, s2 ** 2) - 2 * ek(mu1, s1 ** 2, mu2, s2 ** 2))
