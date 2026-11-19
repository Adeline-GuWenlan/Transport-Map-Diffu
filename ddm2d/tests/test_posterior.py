import numpy as np
import pytest

from ddm2d.data import GMM_MEANS, GMM_STD
from ddm2d.posterior import posterior_mean_cov, posterior_params, sample_posterior


@pytest.mark.parametrize("t", [0.1, 0.45, 0.88, 0.99])
def test_posterior_vs_importance_sampling(t):
    """Self-normalised importance sampling. Proposal: defensive mixture 0.5 p_0 + 0.5 N(x_t/alpha_t, (2 sigma_t/alpha_t)^2 I)."""
    rng = np.random.default_rng(0)
    xt = np.array([[1.0, 2.5], [-2.0, 0.5], [0.2, 3.0]])
    N = 400_000
    a, s = 1 - t, t
    from scipy.stats import multivariate_normal as mvn

    def prior_pdf(x):
        return 0.5 * mvn(GMM_MEANS[0], GMM_STD ** 2).pdf(x) + 0.5 * mvn(GMM_MEANS[1], GMM_STD ** 2).pdf(x)
    mean, var = posterior_mean_cov(xt, t)
    draws = sample_posterior(xt, t, 200_000, rng)
    for i in range(len(xt)):
        comp = rng.integers(0, 2, N)
        x_prior = GMM_MEANS[comp] + GMM_STD * rng.standard_normal((N, 2))
        c, sc = xt[i] / a, 2 * s / a
        x_loc = c + sc * rng.standard_normal((N, 2))
        x0 = np.where(rng.random(N)[:, None] < 0.5, x_prior, x_loc)
        q = 0.5 * prior_pdf(x0) + 0.5 * mvn(c, sc ** 2).pdf(x0)
        logw = -((xt[i] - a * x0) ** 2).sum(1) / (2 * s ** 2) + np.log(prior_pdf(x0) + 1e-300) - np.log(q)
        w = np.exp(logw - logw.max())
        w /= w.sum()
        m_is = (w[:, None] * x0).sum(0)
        v_is = (w[:, None] * (x0 - m_is) ** 2).sum(0)
        ess = 1 / (w ** 2).sum()
        assert ess > 1000
        assert np.allclose(mean[i], m_is, atol=0.03 + 5 * np.sqrt(v_is.max() / ess))
        assert np.allclose(var[i], v_is, rtol=0.05, atol=0.01)
        assert np.allclose(draws[i].mean(0), mean[i], atol=0.02 + 5 * np.sqrt(var[i].max() / 200_000))
        assert np.allclose(draws[i].var(0), var[i], rtol=0.02, atol=1e-3)


def test_posterior_limits():
    xt = np.array([[0.5, 0.5]])
    w, nu, v = posterior_params(xt, 0.999)
    assert np.allclose(w, 0.5, atol=0.01) and v == pytest.approx(GMM_STD ** 2, rel=0.01)
    w, nu, v = posterior_params(np.array([[2.9, 3.1]]), 0.001)
    assert w[0, 0] > 0.999 and v < 1e-5
