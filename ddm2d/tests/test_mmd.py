import numpy as np
import pytest

from ddm2d.mmd import gaussian_mmd2_closed_form, rbf_mmd2


def naive(x, y, sigma):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    k = lambda a, b: np.exp(-((a[:, None] - b[None]) ** 2).sum(-1) / (2 * sigma ** 2))
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    n, m = len(x), len(y)
    u = (kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean()
    b = kxx.mean() + kyy.mean() - 2 * kxy.mean()
    return u, b


@pytest.mark.parametrize("sigma", [0.5, 1.0])
def test_against_naive_with_chunking(sigma):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(700, 2)).astype(np.float32)
    y = (rng.normal(size=(500, 2)) * 1.3 + 0.4).astype(np.float32)
    r = rbf_mmd2(x, y, sigma, chunk=128)                 # several blocks, ragged last block
    u, b = naive(x, y, sigma)
    assert r["mmd2_u"] == pytest.approx(u, rel=1e-10, abs=1e-13)
    assert r["mmd2_b"] == pytest.approx(b, rel=1e-10, abs=1e-13)
    assert r["d_rho_u"] == pytest.approx(u / 2)


def test_against_gaussian_closed_form():
    rng = np.random.default_rng(1)
    mu1, s1, mu2, s2 = np.array([0.0, 0.0]), 1.0, np.array([0.7, -0.2]), 0.8
    exact = gaussian_mmd2_closed_form(mu1, s1, mu2, s2, 1.0)
    vals = []
    for _ in range(5):
        x = mu1 + s1 * rng.normal(size=(4096, 2))
        y = mu2 + s2 * rng.normal(size=(4096, 2))
        vals.append(rbf_mmd2(x, y, 1.0)["mmd2_u"])
    vals = np.array(vals)
    se = vals.std(ddof=1) / np.sqrt(len(vals))
    assert abs(vals.mean() - exact) < 4 * se + 1e-4, (vals.mean(), exact, se)


def test_unbiased_is_centred_at_zero_and_biased_is_not():
    rng = np.random.default_rng(2)
    u, kx = [], []
    n = 1024
    for _ in range(10):
        x, y = rng.normal(size=(n, 2)), rng.normal(size=(n, 2))
        r = rbf_mmd2(x, y, 1.0)
        u.append(r["mmd2_u"])
        kx.append(r["kxx_u"])
        # exact identity: V-statistic = U-statistic + (1 - mean k_xx)/n + (1 - mean k_yy)/m
        assert r["mmd2_b"] - r["mmd2_u"] == pytest.approx((1 - r["kxx_u"]) / n + (1 - r["kyy_u"]) / n, rel=1e-9)
    assert abs(np.mean(u)) < 4 * np.std(u) / np.sqrt(10) + 1e-6
    # E k(x, x') = (1/(1+2))^(d/2) = 1/3 for N(0, I), sigma = 1, so the V-statistic bias is about 2 (2/3) / n
    assert np.mean(kx) == pytest.approx(1 / 3, rel=0.02)
