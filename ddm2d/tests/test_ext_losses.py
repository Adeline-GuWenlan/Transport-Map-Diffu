"""ddm2dx X1 (E4): rho option of Eq. (14). Golden test of the existing path, naive double loops, unbiasedness, finite gradients."""
import hashlib
import itertools
import math

import numpy as np
import pytest
import torch

from ddm2d.losses import DELTA, energy_loss, median_dist, median_sq_dist, rho_loss, rho_of_diff

# energy_loss values / input-gradient hashes computed at commit bff64b1 (before any ddm2dx change), inputs from _golden_inputs()
GOLDEN = {
    (2.0, 0.0): ("0x1.7a143c0000000p+0", "ff31d09550e0af2d0e248b20e8ae6b71523ef856ab29920f4558a7e99565d1d6"),
    (0.1, 1.0): ("0x1.a7747c0000000p-3", "8cddb07ebcd6f254e77fe09603c792797058a09a70e775ae94f28c0c4c9ca0df"),
    (1.0, 1.0): ("0x1.432f520000000p-2", "006b3619f3ae215af38085d2f5d267654f5cb8e33f9ebf3d49dec7220ef43d9b"),
    (0.5, 0.0): ("0x1.fec3c80000000p-2", "4245f004084c20cffadbd8c567979320178af144b58f4a06eb16c8254285db79"),
    (3.0, 0.0): ("0x1.f155bc0000000p+1", "ed0b3b638c25636fd09c18800b032480f075aa07ecf63c4796ec2b5b338339ac"),
}


def _golden_inputs():
    g = torch.Generator().manual_seed(123)
    return torch.randn(16, 2, generator=g), torch.randn(16, 32, 2, generator=g), torch.rand(16, generator=g)


@pytest.mark.parametrize("beta,lam", list(GOLDEN))
@pytest.mark.parametrize("via", ["energy_loss", "rho_loss"])
def test_golden_default_path(beta, lam, via):
    x0, xh, w = _golden_inputs()
    xh.requires_grad_(True)
    loss, _ = energy_loss(x0, xh, w, beta, lam) if via == "energy_loss" else rho_loss(x0, xh, w, "l2", beta, lam)
    loss.backward()
    assert loss.item().hex() == GOLDEN[(beta, lam)][0]
    assert hashlib.sha256(xh.grad.numpy().tobytes()).hexdigest() == GOLDEN[(beta, lam)][1]


@pytest.mark.parametrize("beta,lam", list(itertools.product([0.1, 1.0, 1.5, 2.0], [0.0, 1.0])))
def test_generic_l2_equals_energy_loss(beta, lam):
    g = torch.Generator().manual_seed(7)
    x0, xh, w = torch.randn(5, 2, generator=g, dtype=torch.float64), torch.randn(5, 6, 2, generator=g, dtype=torch.float64), \
        torch.rand(5, generator=g, dtype=torch.float64)
    a, _ = energy_loss(x0, xh, w, beta, lam)
    b, _ = rho_loss(x0, xh, w, "l2", beta, lam, generic=True)
    assert float(a) == pytest.approx(float(b), rel=1e-14, abs=1e-15)


def naive_rho(u, v, rho, beta, c, delta):
    d = [float(a - b) for a, b in zip(u, v)]
    sq = sum(x * x for x in d)
    if rho == "l2":
        return sq if beta == 2.0 else (sq + delta) ** (beta / 2)
    if rho == "l1":
        return sum(math.sqrt(x * x + delta) for x in d) ** beta
    if rho == "pseudo_huber":
        return math.sqrt(sq + c * c) - c
    if rho == "imq":
        return -1.0 / math.sqrt(sq + c)
    if rho == "rbf":
        return -math.exp(-sq / (2 * c))
    if rho == "laplace":
        return -math.exp(-math.sqrt(sq + delta) / c)


CASES = [("l1", 1.0, 0.0), ("l1", 0.5, 0.0), ("pseudo_huber", 1.0, 0.1), ("pseudo_huber", 1.0, 1.0), ("imq", 1.0, 1.0),
         ("rbf", 1.0, 12.5), ("laplace", 1.0, 3.5), ("l2", 0.5, 0.0)]


@pytest.mark.parametrize("rho,beta,c", CASES)
@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_rho_vs_double_loop(rho, beta, c, lam):
    g = torch.Generator().manual_seed(11)
    n, m, d = 3, 5, 2
    x0 = torch.randn(n, d, generator=g, dtype=torch.float64)
    xh = torch.randn(n, m, d, generator=g, dtype=torch.float64)
    w = torch.rand(n, generator=g, dtype=torch.float64)
    ours, diag = rho_loss(x0, xh, w, rho, beta, lam, c, generic=True)
    total = 0.0
    for i in range(n):
        for j in range(m):
            inner = sum(naive_rho(xh[i, j], xh[i, jp], rho, beta, c, DELTA) for jp in range(m) if jp != j)
            total += float(w[i]) * (naive_rho(x0[i], xh[i, j], rho, beta, c, DELTA) - lam / (2 * (m - 1)) * inner)
    assert float(ours) == pytest.approx(total / (n * m), rel=1e-12, abs=1e-14)


def test_l1_smoothing_is_negligible():
    g = torch.Generator().manual_seed(0)
    d = torch.randn(1000, 2, generator=g, dtype=torch.float64)
    for beta in [1.0, 0.5]:
        exact = d.abs().sum(-1) ** beta
        assert torch.allclose(rho_of_diff(d, "l1", beta, 0.0), exact, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("rho,beta,c", CASES)
def test_interaction_estimator_unbiased(rho, beta, c):
    """inter_i (j = j' excluded) averages to E rho(X, X') for X, X' iid; including the diagonal would be biased."""
    g = torch.Generator().manual_seed(5)
    n, m = 20_000, 4
    xh = torch.randn(n, m, 2, generator=g, dtype=torch.float64) * 2.0
    x0 = torch.zeros(n, 2, dtype=torch.float64)
    _, diag = rho_loss(x0, xh, torch.ones(n, dtype=torch.float64), rho, beta, 1.0, c, generic=True)
    a = torch.randn(2_000_000, 2, generator=g, dtype=torch.float64) * 2.0
    b = torch.randn(2_000_000, 2, generator=g, dtype=torch.float64) * 2.0
    vals = rho_of_diff(a - b, rho, beta, c)
    truth, sd = float(vals.mean()), float(vals.std())
    se = sd / math.sqrt(n * m * (m - 1) / 2 / 3)       # conservative: pairs within a row are correlated
    assert abs(float(diag["inter"]) - truth) < 5 * se + 1e-4
    with_diag = float(diag["inter"]) * (m - 1) / m + float(rho_of_diff(torch.zeros(1, 2, dtype=torch.float64), rho, beta, c)) / m
    assert abs(with_diag - truth) > 5 * se        # the biased (V-statistic) version is detectably off


@pytest.mark.parametrize("rho,beta,c", CASES + [("l2", 0.1, 0.0)])
def test_finite_gradients_at_coincident_points(rho, beta, c):
    """All particles equal to each other and to x0: d = 0 everywhere. Gradients must be finite (beta < 1 and L1 included)."""
    x0 = torch.zeros(3, 2)
    xh = torch.zeros(3, 8, 2, requires_grad=True)
    loss, _ = rho_loss(x0, xh, torch.ones(3), rho, beta, 1.0, c)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(xh.grad).all()
    xh2 = (torch.zeros(3, 8, 2) + 1e-7 * torch.randn(3, 8, 2)).requires_grad_(True)
    loss, _ = rho_loss(x0, xh2, torch.ones(3), rho, beta, 1.0, c)
    loss.backward()
    assert torch.isfinite(xh2.grad).all()


def test_strictly_proper_direction_kernels():
    """For lam = 1 the expected loss under a matched spread beats a collapsed (point-mass) prediction, for every rho."""
    g = torch.Generator().manual_seed(2)
    n, m = 4000, 16
    x0 = torch.randn(n, 2, generator=g, dtype=torch.float64)
    good = torch.randn(n, m, 2, generator=g, dtype=torch.float64)
    collapsed = torch.zeros(n, m, 2, dtype=torch.float64)
    for rho, beta, c in CASES:
        lg, _ = rho_loss(x0, good, torch.ones(n, dtype=torch.float64), rho, beta, 1.0, c, generic=True)
        lc, _ = rho_loss(x0, collapsed, torch.ones(n, dtype=torch.float64), rho, beta, 1.0, c, generic=True)
        assert float(lg) < float(lc), rho


def test_median_heuristic_deterministic():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5000, 2))
    a, b = median_sq_dist(x, 10, 128, 0), median_sq_dist(x, 10, 128, 0)
    assert a == b
    # for 2D N(0, I): ||x - x'||^2 ~ 2 chi^2_2 = Exp(mean 4); median = 4 ln 2
    assert median_sq_dist(x, 50, 256, 1) == pytest.approx(4 * math.log(2), rel=0.03)
    assert median_dist(x, 50, 256, 1) == pytest.approx(math.sqrt(4 * math.log(2)), rel=0.02)
