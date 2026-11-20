"""Exact checkerboard posterior (src/ddm2d/posterior_cb.py) vs self-normalised importance sampling with p_0 as the proposal."""
import numpy as np
import pytest

from ddm2d import data as D
from ddm2d.posterior_cb import posterior_mean, sample_posterior


@pytest.mark.parametrize("t", [0.15, 0.3, 0.6, 0.95])
def test_cb_posterior_vs_importance_sampling(t):
    rng = np.random.default_rng(1)
    xt = np.array([[0.5, 1.5], [-3.0, -1.0], [2.2, 3.9], [0.0, 0.0]])
    x0 = D.sample_checkerboard(1_000_000, rng)
    mean = posterior_mean(xt, t)
    draws = sample_posterior(xt, t, 100_000, rng)
    a, s = 1 - t, t
    for i in range(len(xt)):
        logw = -((xt[i] - a * x0) ** 2).sum(1) / (2 * s ** 2)
        w = np.exp(logw - logw.max())
        w /= w.sum()
        ess = 1 / (w ** 2).sum()
        assert ess > 500
        m_is = (w[:, None] * x0).sum(0)
        sd_is = np.sqrt((w[:, None] * (x0 - m_is) ** 2).sum(0))
        tol = 4 * sd_is / np.sqrt(ess) + 1e-3
        np.testing.assert_array_less(np.abs(mean[i] - m_is), tol)
        np.testing.assert_array_less(np.abs(draws[i].mean(0) - m_is), tol + 4 * sd_is / np.sqrt(100_000))
        np.testing.assert_allclose(draws[i].std(0), sd_is, rtol=0.05, atol=1e-3)
        assert D.in_checkerboard(draws[i]).mean() > 0.999


def test_cb_posterior_small_t_is_finite():
    rng = np.random.default_rng(0)
    from ddm2d.posterior_cb import SQUARES
    xt = np.array([[0.3, 0.3], [-3.0, 1.0], [5.0, -5.0], [1.0, 1.0]])   # (-3, 1): centre of an occupied square; (5, -5): outside the board
    for t in [1e-4, 0.01]:
        m = posterior_mean(xt, t)
        d = sample_posterior(xt, t, 1000, rng).reshape(-1, 2)
        assert np.isfinite(m).all() and np.isfinite(d).all()
        gap = np.clip(np.maximum(SQUARES[None] - d[:, None], d[:, None] - SQUARES[None] - 2), 0, None)   # distance to each square, per axis
        assert np.sqrt((gap ** 2).sum(-1)).min(1).max() < 1e-9           # every draw lies in (the closure of) an occupied square
