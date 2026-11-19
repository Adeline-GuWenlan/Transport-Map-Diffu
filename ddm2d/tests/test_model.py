import pytest
import torch

from ddm2d.losses import diffusion_loss, energy_loss
from ddm2d.model import DenoiserMLP, build_model, n_params


def test_shapes_and_layer_count():
    a, c = build_model("A"), build_model("C")
    n = 5
    t, xt = torch.rand(n), torch.randn(n, 2)
    assert a(t, xt).shape == (n, 2)
    assert c(t, xt, torch.randn(n, 7, 2)).shape == (n, 7, 2)
    assert c(t, xt, torch.randn(n, 2)).shape == (n, 2)
    assert a.out.out_features == 2 and c.out.out_features == 4
    assert c.x_pre[0].in_features == 4 and a.x_pre[0].in_features == 2
    lin = lambda s: sum(isinstance(l, torch.nn.Linear) for l in s)
    assert lin(c.t_pre) == lin(c.x_pre) == lin(c.joint) == 4          # 4 + 4 + 1 = 9 backbone layers
    assert c.time_mlp[0].in_features == 2048 and c.time_mlp[2].out_features == 2048
    assert c.t_pre[0].in_features == 2048 and c.joint[0].in_features == 128
    assert all(l.out_features == 64 for l in c.joint if isinstance(l, torch.nn.Linear))
    assert 8e6 < n_params(c) < 9e6


def test_prediction_is_last_two_output_dims():
    torch.manual_seed(0)
    c = build_model("C")
    t, xt, xi = torch.rand(3), torch.randn(3, 2), torch.randn(3, 4, 2)
    h_t = c.time_features(t)
    xin = torch.cat([xt[:, None].expand(3, 4, 2), xi], -1)
    full = c.out(c.joint(torch.cat([h_t[:, None].expand(3, 4, 64), c.x_pre(xin)], -1)))
    assert torch.allclose(c(t, xt, xi), full[..., 2:4])


def test_independent_particles_and_broadcasting():
    torch.manual_seed(0)
    c = build_model("C")
    n, m = 4, 6
    t, xt = torch.rand(n), torch.randn(n, 2)
    xi = torch.randn(n, m, 2)
    y = c(t, xt, xi)
    # distinct xi -> distinct outputs
    d = (y[:, :, None] - y[:, None]).norm(dim=-1)
    assert (d + torch.eye(m) > 1e-7).all()
    # same xi -> same output; batched == per-particle evaluation (each particle uses its own x_t, t)
    xi2 = xi.clone()
    xi2[:, 3] = xi2[:, 1]
    y2 = c(t, xt, xi2)
    assert torch.allclose(y2[:, 3], y2[:, 1])
    for j in range(m):
        assert torch.allclose(y[:, j], c(t, xt, xi[:, j]), atol=1e-6)
    for i in range(n):
        assert torch.allclose(y[i], c(t[i:i + 1].expand(m), xt[i:i + 1].expand(m, 2), xi[i])[:, :], atol=1e-6)


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_backprop_through_losses(arm):
    torch.manual_seed(0)
    mdl = build_model(arm)
    n, m = 8, 4
    t, x0 = torch.rand(n) * 0.98 + 0.01, torch.randn(n, 2)
    xt = (1 - t)[:, None] * x0 + t[:, None] * torch.randn(n, 2)
    w = torch.ones(n)
    if arm == "A":
        loss, _ = diffusion_loss(x0, mdl(t, xt), w)
    else:
        beta, lam = (2.0, 0.0) if arm == "B" else (0.1, 1.0)
        loss, _ = energy_loss(x0, mdl(t, xt, torch.randn(n, m, 2)), w, beta, lam)
    loss.backward()
    grads = [p.grad for p in mdl.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0
    assert mdl.time_mlp[0].weight.grad.abs().sum() > 0


def test_standard_model_rejects_xi():
    with pytest.raises(AssertionError):
        build_model("A")(torch.rand(2), torch.randn(2, 2), torch.randn(2, 2))
