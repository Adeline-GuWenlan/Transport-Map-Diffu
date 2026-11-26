"""ddm2dx X1: general GMM posterior (per-component weights and stds) vs posterior.py and vs importance sampling."""
import numpy as np
import pytest

from ddm2d import data as D
from ddm2d import posterior as P
from ddm2d import posterior_gmm as PG


@pytest.mark.parametrize("t", [0.01, 0.23, 0.45, 0.88, 0.99])
def test_equals_paper_posterior_on_paper_gmm(t):
    rng = np.random.default_rng(1)
    xt = rng.normal(0, 3, (64, 2)) + np.array([0, 3])
    w0, nu0, v0 = P.posterior_params(xt, t)
    r, m, v = PG.posterior_params(xt, t, *PG.spec("gmm"))
    assert np.allclose(r, w0, rtol=1e-10, atol=1e-14)
    assert np.allclose(m, nu0, rtol=1e-12, atol=1e-12)
    assert np.allclose(v, v0, rtol=1e-12)
    m0, c0 = P.posterior_mean_cov(xt, t)
    m1, c1 = PG.posterior_mean_cov(xt, t, *PG.spec("gmm"))
    assert np.allclose(m1, m0, rtol=1e-10, atol=1e-12) and np.allclose(c1, c0, rtol=1e-8, atol=1e-12)


def _prior_logpdf(x, means, stds, w):
    d = x.shape[1]
    lp = (np.log(w)[None] - d * np.log(stds)[None] - 0.5 * d * np.log(2 * np.pi)
          - ((x[:, None, :] - means[None]) ** 2).sum(-1) / (2 * stds[None] ** 2))
    mx = lp.max(1, keepdims=True)
    return (mx + np.log(np.exp(lp - mx).sum(1, keepdims=True)))[:, 0]


@pytest.mark.parametrize("name", ["gmm9w", "gmm2w"])
@pytest.mark.parametrize("t", [0.1, 0.45, 0.7, 0.88, 0.99])
def test_vs_importance_sampling(name, t):
    """Self-normalised IS with the prior p_0 as proposal (mixed 50/50 with a local Gaussian around x_t/alpha)."""
    means, stds, w = PG.spec(name)
    rng = np.random.default_rng(int(100 * t))
    a, s = 1 - t, t
    x0_true = D.generate(name, 4, 7).astype(np.float64)
    xt = a * x0_true + s * rng.standard_normal(x0_true.shape)
    mean, var = PG.posterior_mean_cov(xt, t, means, stds, w)
    r = PG.responsibilities(xt, t, means, stds, w)
    draws, comp = PG.sample_posterior(xt, t, 200_000, rng, means, stds, w, return_comp=True)
    N = 400_000
    for i in range(len(xt)):
        xp, kp = D.sample_weighted_gmm(N, rng, means, stds, w, return_comp=True)
        c, sc = xt[i] / a, 2 * s / a
        xl = c + sc * rng.standard_normal((N, 2))
        use_p = rng.random(N) < 0.5
        x0 = np.where(use_p[:, None], xp, xl)
        lprior = _prior_logpdf(x0, means, stds, w)
        lloc = -((x0 - c) ** 2).sum(1) / (2 * sc ** 2) - 2 * np.log(sc) - np.log(2 * np.pi)
        lq = np.logaddexp(np.log(0.5) + lprior, np.log(0.5) + lloc)
        logw = -((xt[i] - a * x0) ** 2).sum(1) / (2 * s ** 2) + lprior - lq
        wt = np.exp(logw - logw.max())
        wt /= wt.sum()
        ess = 1 / (wt ** 2).sum()
        assert ess > 1000
        m_is = (wt[:, None] * x0).sum(0)
        v_is = (wt[:, None] * (x0 - m_is) ** 2).sum(0)
        assert np.allclose(mean[i], m_is, atol=0.02 + 5 * np.sqrt(v_is.max() / ess))
        assert np.allclose(var[i], v_is, rtol=0.08, atol=0.01)
        # responsibilities: component frequencies of the exact draws (the IS check of r_k is the next test)
        r_draws = np.bincount(comp[i], minlength=len(w)) / comp.shape[1]
        assert np.allclose(r_draws, r[i], atol=5 * np.sqrt(0.25 / comp.shape[1]))
        assert np.allclose(draws[i].mean(0), mean[i], atol=0.02 + 5 * np.sqrt(var[i].max() / 2e5))
        assert np.allclose(draws[i].var(0), var[i], rtol=0.03, atol=1e-3)


@pytest.mark.parametrize("t", [0.45, 0.88])
def test_responsibilities_vs_importance_sampling_gmm9w(t):
    """P(k | x_t) by IS over the prior (proposal = p_0, known component labels): r_k ∝ E_{p0}[1{k} N(x_t; a x0, s^2)]."""
    means, stds, w = PG.spec("gmm9w")
    rng = np.random.default_rng(3)
    a, s = 1 - t, t
    xt = np.array([[-2.0, 2.0], [3.0, 0.5], [0.0, -2.5]])
    r = PG.responsibilities(xt, t, means, stds, w)
    N = 2_000_000
    x0, k = D.sample_weighted_gmm(N, rng, means, stds, w, return_comp=True)
    for i in range(len(xt)):
        lw = -((xt[i] - a * x0) ** 2).sum(1) / (2 * s ** 2)
        wt = np.exp(lw - lw.max())
        r_is = np.bincount(k, weights=wt, minlength=9) / wt.sum()
        ess = wt.sum() ** 2 / (wt ** 2).sum()
        assert ess > 2000
        assert np.allclose(r_is, r[i], atol=0.01 + 4 * np.sqrt(0.25 / ess))


def test_limits_and_assign():
    means, stds, w = PG.spec("gmm9w")
    r, m, v = PG.posterior_params(np.zeros((1, 2)), 0.9999, means, stds, w)
    assert np.allclose(r[0], w, atol=1e-3) and np.allclose(v, stds ** 2, rtol=1e-3)
    x = means + 0.01
    r, m, v = PG.posterior_params(x * 0.999, 0.001, means, stds, w)
    assert np.all(r.argmax(1) == np.arange(9)) and np.all(v < 1e-5)
    assert np.array_equal(PG.assign_components(x, means, stds, w), np.arange(9))
