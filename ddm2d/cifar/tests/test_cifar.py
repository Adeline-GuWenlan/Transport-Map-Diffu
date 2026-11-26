"""CPU tests for ddmcifar with a tiny U-Net. Run: PYTHONPATH=src:cifar/src python -m pytest cifar/tests -q (from ddm2d/)."""
from __future__ import annotations

import copy
import itertools

import pytest
import torch

from ddmcifar import losses as L
from ddmcifar.data import EpochSampler, load_cifar, to_uint8
from ddmcifar.sampler import sample, x0_hat, xi_spread
from ddmcifar.train import Trainer, lr_factor, resolve_config
from ddmcifar.unet import DDMUNet, build_unet

TINY = dict(base_channels=16, channel_mult=[1, 2], num_res_blocks=1, attention_resolutions=[16], num_heads=2, emb_dim=32)


def fake_data(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand((n, 3, 32, 32), generator=g) * 2 - 1, torch.randint(0, 10, (n,), generator=g)


def tiny_cfg(arm, **kw):
    c = dict(arm=arm, model=TINY, batch=8, m=4, device="cpu", bf16=False, steps=6, log_every=2, ckpt_every=3, snapshots=[])
    if arm == "B":
        c.update(beta=2.0, lam=0.0)
    if arm == "C":
        c.update(beta=0.1, lam=1.0)
    c.update(kw)
    return c


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_shapes(arm):
    torch.manual_seed(0)
    m = build_unet(arm, TINY)
    x = torch.randn(5, 3, 32, 32)
    t = torch.rand(5)
    c = torch.randint(0, 10, (5,))
    out = m(t, x, c, torch.randn_like(x)) if m.stochastic else m(t, x, c)
    assert out.shape == (5, 3, 32, 32)
    assert m.net.conv_in.in_channels == (6 if arm != "A" else 3)
    assert m.net.conv_out.out_channels == (6 if arm != "A" else 3)


def test_channel_drop():
    """B / C return channels 3..5 of the 6-channel UNet output (the first half is dropped)."""
    torch.manual_seed(0)
    m = build_unet("B", TINY)
    x, xi = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
    t, c = torch.rand(2), torch.tensor([1, 2])
    full = m.net(torch.cat([x, xi], 1), 1000 * t, c)
    assert full.shape[1] == 6
    assert torch.equal(m(t, x, c, xi), full[:, 3:])


def test_full_size_params():
    m = DDMUNet("B")
    n = sum(p.numel() for p in m.parameters())
    assert 139.9e6 < n < 140.1e6          # transport_map's App. I.2 U-Net: 140.0 M


@pytest.mark.parametrize("arm", ["B", "C"])
def test_outputs_depend_on_xi_at_init(arm):
    torch.manual_seed(0)
    m = build_unet(arm, TINY)
    x, t, c = torch.randn(3, 3, 32, 32), torch.rand(3), torch.tensor([0, 1, 2])
    a = m(t, x, c, torch.randn_like(x))
    b = m(t, x, c, torch.randn_like(x))
    assert (a - b).abs().mean() > 1e-3 * a.abs().mean()


def naive_energy(x0, xh, w, lam, rho, beta, c):
    n, m = xh.shape[:2]
    tot = 0.0
    for i in range(n):
        f = lambda a, b: L.rho_from_sq(((a - b) ** 2).sum().reshape(1), rho, beta, c)[0]
        conf = sum(f(x0[i], xh[i, j]) for j in range(m)) / m
        inter = sum(f(xh[i, j], xh[i, k]) for j, k in itertools.product(range(m), range(m)) if j != k) / (m * (m - 1))
        tot = tot + w[i] * (conf - lam / 2 * inter)
    return tot / n


@pytest.mark.parametrize("rho,beta,lam,c", [("l2pow", 2.0, 0.0, 0.0), ("l2pow", 1.0, 0.0, 0.0), ("l2pow", 0.1, 1.0, 0.0),
                                            ("pseudo_huber", 2.0, 0.0, 0.03), ("pseudo_huber", 2.0, 1.0, 0.03)])
def test_loss_vs_naive(rho, beta, lam, c):
    g = torch.Generator().manual_seed(1)
    x0 = torch.randn(3, 3, 4, 4, generator=g)
    xh = torch.randn(3, 4, 3, 4, 4, generator=g)
    w = torch.rand(3, generator=g)
    loss, _ = L.energy_loss(x0, xh, w, lam, rho, beta, c)
    ref = naive_energy(x0.float(), xh.float(), w.float(), lam, rho, beta, c)
    assert torch.allclose(loss, ref, rtol=1e-5, atol=1e-6)


def test_loss_matches_ddm2d_energy_loss():
    """rho = l2pow gives exactly ddm2d's Eq. (14) (on flattened vectors)."""
    from ddm2d.losses import energy_loss as e2d
    g = torch.Generator().manual_seed(2)
    x0, xh, w = torch.randn(5, 3, 4, 4, generator=g), torch.randn(5, 4, 3, 4, 4, generator=g), torch.rand(5, generator=g)
    for beta, lam in [(2.0, 0.0), (1.0, 0.0), (0.1, 1.0), (1.5, 1.0)]:
        a, _ = L.energy_loss(x0, xh, w, lam, "l2pow", beta)
        b, _ = e2d(x0.flatten(1), xh.flatten(2), w, beta, lam)
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_velocity_loss_and_A_x0_from_v():
    g = torch.Generator().manual_seed(3)
    x0, x1 = torch.randn(4, 3, 8, 8, generator=g), torch.randn(4, 3, 8, 8, generator=g)
    t = torch.rand(4, generator=g)
    tb = t.view(-1, 1, 1, 1)
    xt = (1 - tb) * x0 + tb * x1
    v = x0 - x1
    assert torch.allclose(xt + tb * v, x0, atol=1e-6)            # x0 = x_t + t v (the sign used by the A sampler)
    loss, _ = L.velocity_loss(v, v + 1.0, torch.ones(4))
    assert torch.isclose(loss, torch.tensor(3 * 8 * 8.0))

    class Oracle(torch.nn.Module):                               # returns the true v for the fixed (x0, x1) pair
        stochastic = False
        input_size = 8

        def forward(self, t, x, c):
            return (x0 - x) / t.view(-1, 1, 1, 1)                # = x0 - x1 at x = x_t
    out = x0_hat(Oracle(), 0.37, (1 - 0.37) * x0 + 0.37 * x1, torch.zeros(4, dtype=torch.long), None)
    assert torch.allclose(out, x0, atol=1e-5)


class CountingNet(torch.nn.Module):
    def __init__(self, stochastic):
        super().__init__()
        self.stochastic, self.input_size, self.calls, self.ts = stochastic, 32, 0, []

    def forward(self, t, x, c, xi=None):
        self.calls += 1
        self.ts.append(float(t[0]))
        return torch.zeros_like(x)


@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("nfe", [1, 2, 10])
def test_sampler_nfe(stochastic, nfe):
    net = CountingNet(stochastic)
    x, n = sample(net, torch.zeros(6, dtype=torch.long), nfe, seed=0, batch=6, return_nfe=True)
    assert n == nfe and net.calls == nfe and x.shape == (6, 3, 32, 32)
    assert net.ts[0] == pytest.approx(0.99) and net.ts[-1] == pytest.approx(min(max(1 / nfe, 0.01), 0.99))
    if stochastic:
        assert torch.equal(x, torch.zeros_like(x))                # last step: s = 0, output = last x0_hat (= 0)


def test_sampler_deterministic_and_batch_invariant():
    torch.manual_seed(0)
    m = build_unet("C", TINY)
    lab = torch.arange(6) % 10
    a = sample(m, lab, 3, seed=5, batch=6)
    b = sample(m, lab, 3, seed=5, batch=2)
    assert torch.allclose(a, b, atol=1e-5)
    assert not torch.allclose(a, sample(m, lab, 3, seed=6, batch=6))


def test_xi_spread():
    torch.manual_seed(0)
    x0, c = fake_data(4)
    sb = xi_spread(build_unet("B", TINY), x0, c, [0.2, 0.8], n_draws=3)
    sa = xi_spread(build_unet("A", TINY), x0, c, [0.2, 0.8], n_draws=3)
    assert all(v > 0 for v in sb.values()) and all(v == 0 for v in sa.values())


def test_sampler_order_and_resume():
    x, y = fake_data(20)
    s = EpochSampler(x, y, 8, seed=0)
    s2 = copy.deepcopy(s)
    b1 = [s.next() for _ in range(5)]                             # crosses two epoch boundaries (20 // 8 = 2 per epoch)
    st = s2.state_dict()
    s3 = EpochSampler(x, y, 8, seed=99)
    s3.load_state_dict(st)
    b2 = [s3.next() for _ in range(5)]
    assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(b1, b2))
    assert s.epoch == 2


def test_load_cifar_range():
    x, y = load_cifar("train", n=16)
    assert x.shape == (16, 3, 32, 32) and x.min() >= -1 and x.max() <= 1 and y.dtype == torch.int64
    assert torch.equal(to_uint8(x).float().div(127.5).sub(1), x)


def test_lr_warmup():
    import math
    assert lr_factor(0, 100) == pytest.approx(0.5 * (1 - math.cos(math.pi / 100)))
    assert lr_factor(99, 100) == 1.0 and lr_factor(5000, 100) == 1.0


def test_arms_share_draws():
    """Same seed: A, B and C see the same images, labels, t and X_1."""
    data = fake_data(64)
    ds = [Trainer(tiny_cfg(a), data=data).draw() for a in "ABC"]
    for d in ds[1:]:
        for k in range(4):
            assert torch.equal(ds[0][k], d[k])
    assert ds[0][4] is None and ds[1][4].shape == (8 * 4, 3, 32, 32)


@pytest.mark.parametrize("arm,feat", [("A", None), ("C", None), ("C", {"encoder": "random_conv", "scale": 2.0})])
def test_chunked_gradient_exact(arm, feat):
    data = fake_data(64)
    grads = []
    for chunk in (0, 5):
        tr = Trainer(tiny_cfg(arm, grad_chunk=chunk, feature=feat), data=data)
        tr.init_feature_scale()
        b = tr.draw()
        tr.opt.zero_grad()
        loss, _ = tr.loss_and_grad(*b)
        grads.append((loss, torch.cat([p.grad.flatten() for p in tr.model.parameters()])))
    assert torch.allclose(grads[0][0], grads[1][0], rtol=1e-5)
    g0, g1 = grads[0][1], grads[1][1]
    assert (g0 - g1).norm() <= 1e-4 * g0.norm(), float((g0 - g1).norm() / g0.norm())


def test_feature_scale_auto():
    data = fake_data(64)
    tr = Trainer(tiny_cfg("C", feature={"encoder": "random_conv", "scale": "auto", "scale_n_per_class": 5}), data=data)
    tr.init_feature_scale()
    assert tr.feat["scale"] > 0 and tr.feat["lambda_f"] == tr.feat["scale"]


@pytest.mark.parametrize("arm", ["A", "C"])
def test_resume_bitwise(arm, tmp_path):
    from ddmcifar.train import run
    data = fake_data(64)
    cfg = tiny_cfg(arm)
    run(cfg, tmp_path / "full", data=data)
    run(cfg, tmp_path / "part", max_steps=3, data=data)          # checkpoint at 3 (ckpt_every = 3), then resume
    run(cfg, tmp_path / "part", data=data)
    a = torch.load(tmp_path / "full" / "ckpt" / "latest.pt", weights_only=False)
    b = torch.load(tmp_path / "part" / "ckpt" / "latest.pt", weights_only=False)
    assert a["step"] == b["step"] == 6
    for k in ("model", "ema"):
        for n in a[k]:
            assert torch.equal(a[k][n], b[k][n]), (k, n)
    assert torch.equal(a["g_xi"], b["g_xi"]) and a["sampler"]["pos"] == b["sampler"]["pos"]
    la = (tmp_path / "full" / "log.jsonl").read_text().splitlines()
    lb = (tmp_path / "part" / "log.jsonl").read_text().splitlines()
    import json
    strip = lambda l: {k: v for k, v in json.loads(l).items() if k not in ("wall_s", "s_per_step")}
    assert [strip(l) for l in la] == [strip(l) for l in lb]


def test_config_A_forces_m1():
    c = resolve_config({"arm": "A", "m": 4, "beta": 0.1, "lam": 1.0})
    assert c["m"] == 1 and c["lam"] == 0.0
    with pytest.raises(ValueError):
        resolve_config({"bogus": 1})
