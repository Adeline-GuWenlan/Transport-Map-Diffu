import numpy as np
import pytest
import torch

from ddm2d.model import build_model
from ddm2d.posterior import posterior_params
from ddm2d.samplers import posterior_draws, sample

MU, S = np.array([[1.0, -2.0]]), 0.7


class ExactGaussianPosterior(torch.nn.Module):
    """Oracle denoiser for p_0 = N(MU, S^2 I): stochastic -> exact posterior sample nu + sqrt(v) xi; else posterior mean."""

    def __init__(self, stochastic):
        super().__init__()
        self.stochastic = stochastic
        self.calls = 0

    def forward(self, t, xt, xi=None):
        self.calls += 1
        tt = float(t[0])
        _, nu, var = posterior_params(xt.double().numpy(), tt, means=MU, s=S, pis=np.array([1.0]))
        nu = torch.from_numpy(nu[:, 0]).to(xt.dtype)
        return nu + var ** 0.5 * xi if self.stochastic else nu


@pytest.mark.parametrize("n_steps", [1, 2, 5])
@pytest.mark.parametrize("churn", [0.0, 1.0])
def test_algorithm2_exact_with_exact_posterior(n_steps, churn):
    """With exact posterior samples, Algorithm 2 is exact for any N (Eq. 5). Safety 0 so the oracle sees the true times."""
    m = ExactGaussianPosterior(True)
    g = torch.Generator().manual_seed(0)
    x, nfe = sample(m, 200_000, n_steps, churn=churn, safety=0.0, generator=g, dtype=torch.float64, return_nfe=True)
    assert nfe == n_steps == m.calls
    assert np.allclose(x.mean(0).numpy(), MU[0], atol=0.01)
    assert np.allclose(x.std(0).numpy(), S, atol=0.01)


def test_deterministic_sampler_collapses_at_one_step_and_converges():
    g = torch.Generator().manual_seed(0)
    x1 = sample(ExactGaussianPosterior(False), 50_000, 1, churn=1.0, safety=0.0, generator=g, dtype=torch.float64)
    assert x1.std(0).max() < 1e-6          # one step returns E[X_0 | X_1] = E[X_0]
    x200 = sample(ExactGaussianPosterior(False), 50_000, 200, churn=1.0, safety=0.0, generator=g, dtype=torch.float64)
    assert np.allclose(x200.std(0).numpy(), S, atol=0.03)


def test_nfe_and_reproducibility_with_real_model():
    torch.manual_seed(0)
    for arm in ["A", "C"]:
        mdl = build_model(arm)
        a, nfe = sample(mdl, 64, 7, generator=torch.Generator().manual_seed(3), return_nfe=True)
        b = sample(mdl, 64, 7, generator=torch.Generator().manual_seed(3), chunk=16)
        assert nfe == 7 and a.shape == (64, 2) and torch.isfinite(a).all()
        assert torch.allclose(a, b, atol=1e-5)


def test_posterior_draws_shapes():
    torch.manual_seed(0)
    xt = torch.randn(5, 2)
    d = posterior_draws(build_model("C"), 0.5, xt, 8, generator=torch.Generator().manual_seed(0))
    assert d.shape == (5, 8, 2) and d.std(1).min() > 0
    d = posterior_draws(build_model("A"), 0.5, xt, 8)
    assert d.shape == (5, 8, 2) and d.std(1).max() == 0
