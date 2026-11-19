import itertools

import pytest
import torch

from ddm2d.losses import DELTA, diffusion_loss, energy_loss


def reference_eq14(x0, xhat, w, beta, lam, delta):
    """Explicit double loop over Eq. (14): (1/(nm)) sum_{i,j} w_i [ ||x0_i - xh_ij||^b - lam/(2(m-1)) sum_{j' != j} ||xh_ij - xh_ij'||^b ]."""
    n, m, d = xhat.shape

    def p(v):
        sq = float((v ** 2).sum())
        return sq if beta == 2.0 else (sq + delta) ** (beta / 2)

    total = 0.0
    for i in range(n):
        for j in range(m):
            inner = 0.0
            for jp in range(m):
                if jp != j:
                    inner += p(xhat[i, j] - xhat[i, jp])
            total += float(w[i]) * (p(x0[i] - xhat[i, j]) - lam / (2 * (m - 1)) * inner)
    return total / (n * m)


@pytest.mark.parametrize("beta,lam", list(itertools.product([0.1, 2.0], [0.0, 1.0])))
@pytest.mark.parametrize("delta", [0.0, DELTA])
def test_eq14_vs_double_loop(beta, lam, delta):
    g = torch.Generator().manual_seed(int(10 * beta + lam))
    n, m, d = 3, 5, 2
    x0 = torch.randn(n, d, generator=g, dtype=torch.float64)
    xhat = torch.randn(n, m, d, generator=g, dtype=torch.float64)
    w = torch.rand(n, generator=g, dtype=torch.float64)
    ours, _ = energy_loss(x0, xhat, w, beta, lam, delta)
    ref = reference_eq14(x0, xhat, w, beta, lam, delta)
    assert float(ours) == pytest.approx(ref, rel=1e-12, abs=1e-14)


def test_eq14_beta2_lam0_is_mean_regression():
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(4, 2, generator=g, dtype=torch.float64)
    xhat = torch.randn(4, 6, 2, generator=g, dtype=torch.float64)
    w = torch.rand(4, generator=g, dtype=torch.float64)
    ours, _ = energy_loss(x0, xhat, w, 2.0, 0.0)
    ref = (w[:, None] * ((x0[:, None] - xhat) ** 2).sum(-1)).mean()
    assert float(ours) == pytest.approx(float(ref), rel=1e-12)
    # m = 1 copy of the same prediction reduces to Eq. (6)
    same = x0[:, None, :].expand(4, 6, 2) + 0.3
    l14, _ = energy_loss(x0, same, w, 2.0, 0.0)
    l6, _ = diffusion_loss(x0, same[:, 0], w)
    assert float(l14) == pytest.approx(float(l6), rel=1e-12)


def test_eq14_strictly_proper_direction():
    """For lam = 1, beta < 2 the population score prefers the true spread: matching samples beat collapsed ones (in expectation)."""
    g = torch.Generator().manual_seed(3)
    n, m = 4000, 16
    x0 = torch.randn(n, 2, generator=g, dtype=torch.float64)
    good = torch.randn(n, m, 2, generator=g, dtype=torch.float64)
    collapsed = torch.zeros(n, m, 2, dtype=torch.float64) + 1e-3 * torch.randn(n, m, 2, generator=g, dtype=torch.float64)
    w = torch.ones(n, dtype=torch.float64)
    lg, _ = energy_loss(x0, good, w, 0.1, 1.0)
    lc, _ = energy_loss(x0, collapsed, w, 0.1, 1.0)
    assert float(lg) < float(lc)


def test_finite_gradients_with_coincident_particles():
    xhat = torch.zeros(2, 4, 2, dtype=torch.float32, requires_grad=True)
    x0 = torch.zeros(2, 2)
    loss, _ = energy_loss(x0, xhat, torch.ones(2), 0.1, 1.0)
    loss.backward()
    assert torch.isfinite(xhat.grad).all()
