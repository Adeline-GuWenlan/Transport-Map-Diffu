"""Exact posterior p(x_0 | x_t) for the FFJORD checkerboard (diagnostic oracle, S4).

p_0 is uniform on the 8 occupied 2x2 squares of [-4, 4]^2 (src/ddm2d/data.py). With x_t = alpha x_0 + sigma eps, the likelihood as a function of
x_0 is N(x_0; x_t / alpha, (sigma / alpha)^2 I). So the posterior is a mixture over the 8 squares of products of 1-D truncated normals
(mean x_t / alpha, std sigma / alpha, truncated to the square's sides), with weights proportional to the Gaussian mass of each square.
Our derivation; checked against self-normalised importance sampling in tests/test_posterior_cb.py.
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri

from .schedule import alpha_sigma

SQUARES = np.array([(i, j) for i in range(-4, 4, 2) for j in range(-4, 4, 2) if ((i // 2) + (j // 2)) % 2 == 0], dtype=float)  # lower-left corners, side 2


def _log_mass(a, b):
    """log(Phi(b) - Phi(a)) for a < b, elementwise, stable in both tails."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    out = np.empty(a.shape)
    pos = a > 0                                   # both in the upper tail: use the symmetric form
    la, lb = np.where(pos, log_ndtr(-b), log_ndtr(a)), np.where(pos, log_ndtr(-a), log_ndtr(b))
    out[...] = lb + np.log1p(-np.exp(np.minimum(la - lb, -1e-300)))
    return out


def _tn_mean(m, s, lo, hi):
    """Mean of N(m, s^2) truncated to [lo, hi]."""
    a, b = (lo - m) / s, (hi - m) / s
    lz = _log_mass(a, b)
    pa = np.exp(-0.5 * a ** 2 - 0.5 * np.log(2 * np.pi) - lz)
    pb = np.exp(-0.5 * b ** 2 - 0.5 * np.log(2 * np.pi) - lz)
    return np.clip(m + s * (pa - pb), lo, hi)


def _components(xt, t, scale=1.0):
    xt = np.asarray(xt, float)
    al, si = alpha_sigma(float(t))
    al, si = float(al), float(si)
    m, s = xt / al, si / al                                   # [n, 2], scalar
    lo, hi = scale * SQUARES[None, :, :], scale * (SQUARES[None, :, :] + 2.0)   # [1, 8, 2]
    a, b = (lo - m[:, None, :]) / s, (hi - m[:, None, :]) / s
    logw = _log_mass(a, b).sum(-1)                            # [n, 8]
    logw -= logw.max(1, keepdims=True)
    w = np.exp(logw)
    w /= w.sum(1, keepdims=True)
    return m, s, lo, hi, w


def posterior_mean(xt, t, scale: float = 1.0):
    """E[X_0 | X_t = x_t] for each row of xt [n, 2]. `scale` rescales the board (diagnostic: scale 0.5 = support [-2, 2]^2)."""
    m, s, lo, hi, w = _components(xt, t, scale)
    mu = _tn_mean(m[:, None, :], s, lo, hi)                    # [n, 8, 2]
    return (w[..., None] * mu).sum(1)


def sample_posterior(xt, t, n_draws: int, rng: np.random.Generator, scale: float = 1.0):
    """n_draws exact posterior draws per row of xt [n, 2] -> [n, n_draws, 2] (inverse-CDF truncated normal; log-space in the far tails)."""
    m, s, lo, hi, w = _components(xt, t, scale)
    n = m.shape[0]
    out = np.empty((n, n_draws, 2))
    cw = np.cumsum(w, 1)
    u = rng.random((n, n_draws))
    k = np.minimum((u[..., None] > cw[:, None, :]).sum(-1), 7)  # component index [n, n_draws]
    for d in range(2):
        L = scale * SQUARES[k, d]
        a = (L - m[:, None, d]) / s
        b = (L + 2.0 * scale - m[:, None, d]) / s
        v = rng.random((n, n_draws))
        # sample on the side of the tail with more numerical room: flip so the interval is in the lower half where ndtr is accurate
        flip = (a + b) > 0
        aa, bb = np.where(flip, -b, a), np.where(flip, -a, b)
        Fa, Fb = ndtr(aa), ndtr(bb)
        z = ndtri(np.clip(Fa + v * (Fb - Fa), 1e-300, 1 - 1e-16))
        z = np.clip(np.where(flip, -z, z), a, b)
        bad = ~np.isfinite(z) | (Fb - Fa <= 0)
        z = np.where(bad, np.where(np.abs(a) < np.abs(b), a, b), z)   # degenerate tail: put the draw at the nearest edge
        out[..., d] = m[:, None, d] + s * z
    return out
