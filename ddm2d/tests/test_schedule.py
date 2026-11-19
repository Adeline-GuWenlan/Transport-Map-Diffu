import numpy as np
import pytest
import torch

from ddm2d.schedule import alpha_sigma, bridge_coeffs, bridge_mean_std, forward_sample, sample_train_times, sampling_grid, sigmoid_weight


def test_schedule_endpoints():
    a0, s0 = alpha_sigma(0.0)
    a1, s1 = alpha_sigma(1.0)
    assert (a0, s0, a1, s1) == (1.0, 0.0, 0.0, 1.0)


@pytest.mark.parametrize("t", [0.01, 0.3, 0.7, 0.99])
def test_forward_marginal_moments(t):
    g = torch.Generator().manual_seed(0)
    n = 400_000
    x0 = torch.tensor([[3.0, -1.0]]).expand(n, 2)
    xt = forward_sample(x0, torch.full((n,), t), torch.randn(n, 2, generator=g))
    mean, std = xt.mean(0), xt.std(0)
    assert torch.allclose(mean, (1 - t) * x0[0], atol=5 * t / n ** 0.5 + 1e-6)
    assert torch.allclose(std, torch.full((2,), t), rtol=0.01)


@pytest.mark.parametrize("churn", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("s,t", [(0.0, 0.2), (0.1, 0.5), (0.45, 0.9), (0.8, 1.0), (0.3, 0.31)])
def test_bridge_preserves_marginal_mc(churn, s, t):
    """x_t ~ p(x_t|x0), x_s ~ N(mu_{s,t}, Sigma_{s,t}) must give x_s ~ N(alpha_s x0, sigma_s^2) (Eq. 4 construction), Monte Carlo."""
    g = torch.Generator().manual_seed(1)
    n = 400_000
    x0 = torch.tensor([1.5, -2.0]).expand(n, 2)
    xt = forward_sample(x0, torch.tensor(t), torch.randn(n, 2, generator=g))
    mu, std = bridge_mean_std(s, t, x0, xt, churn)
    xs = mu + std * torch.randn(n, 2, generator=g)
    a_s, sg_s = alpha_sigma(s)
    tol = 6 * max(sg_s, 1e-3) / n ** 0.5
    assert torch.allclose(xs.mean(0), a_s * x0[0], atol=tol + 1e-6)
    assert torch.allclose(xs.std(0), torch.full((2,), sg_s), atol=0.01 * max(sg_s, 1e-3) + 1e-6)


@pytest.mark.parametrize("churn", [0.0, 0.5, 1.0])
def test_bridge_marginal_closed_form(churn):
    """Exact: c_xt * alpha_t + c_x0 = alpha_s and c_xt^2 sigma_t^2 + std^2 = sigma_s^2."""
    for s, t in [(0.0, 0.5), (0.2, 0.4), (0.5, 0.99), (0.7, 1.0), (0.01, 0.02)]:
        c_xt, c_x0, std = bridge_coeffs(s, t, churn)
        a_s, sg_s = alpha_sigma(s)
        a_t, sg_t = alpha_sigma(t)
        assert c_xt * a_t + c_x0 == pytest.approx(a_s, abs=1e-12)
        assert c_xt ** 2 * sg_t ** 2 + std ** 2 == pytest.approx(sg_s ** 2, abs=1e-12)


def test_bridge_special_cases():
    s, t = 0.3, 0.6
    a_s, sg_s = alpha_sigma(s)
    a_t, sg_t = alpha_sigma(t)
    # churn 0: DDIM, deterministic
    c_xt, c_x0, std = bridge_coeffs(s, t, 0.0)
    assert std == pytest.approx(0.0, abs=1e-12)
    assert c_xt == pytest.approx(sg_s / sg_t) and c_x0 == pytest.approx(a_s - a_t * sg_s / sg_t)
    # churn 1: DDPM posterior q(x_s | x_t, x_0)
    c_xt, c_x0, std = bridge_coeffs(s, t, 1.0)
    a_ts = a_t / a_s
    var_ts = sg_t ** 2 - a_ts ** 2 * sg_s ** 2
    assert c_xt == pytest.approx(a_ts * sg_s ** 2 / sg_t ** 2)
    assert c_x0 == pytest.approx(a_s * var_ts / sg_t ** 2)
    assert std ** 2 == pytest.approx(var_ts * sg_s ** 2 / sg_t ** 2)
    # s = 0: returns x0 exactly
    for churn in (0.0, 0.5, 1.0):
        c_xt, c_x0, std = bridge_coeffs(0.0, 0.2, churn)
        assert (c_xt, c_x0, std) == (0.0, 1.0, 0.0)


def test_sigmoid_weight_and_times():
    t = torch.tensor([0.01, 0.5, 0.99])
    w = sigmoid_weight(t, 0.0)
    assert w[1].item() == pytest.approx(0.5)
    snr = ((1 - t) / t) ** 2
    assert torch.allclose(w, snr / (1 + snr), rtol=1e-5)   # b = 0: w = SNR / (1 + SNR); w(0.99) ~ 1.0e-4
    assert sigmoid_weight(torch.tensor([0.5]), 1.0).item() == pytest.approx(1 / (1 + np.e))
    tt = sample_train_times(100_000, 0.01, generator=torch.Generator().manual_seed(0))
    assert tt.min() >= 0.01 and tt.max() <= 0.99
    assert sampling_grid(4) == [0.0, 0.25, 0.5, 0.75, 1.0]
