"""UNet transport map (src/tmap/unet.py, De Bortoli et al. App. I.2): shapes, F == 0 at init, class conditioning,
exact chunked gradient, checkpoint round trip, and the full-size configs."""
import torch

from tmap.energy_score import energy_score_loss
from tmap.model import build_model, count_params
from tmap.train import load_config, train
from tmap.unet import UNetTransportMap

TINY = dict(input_size=8, in_channels=3, num_classes=3, base_channels=16, channel_mult=(1, 2), num_res_blocks=1,
            attention_resolutions=(4,), num_heads=2, emb_dim=32)
CFG_A = "configs/expA_cifar10_unet_pixel.yaml"
CFG_B = "configs/expB_cifar10_unet_pixel-dinov2.yaml"
SMALL = ["device=cpu", "num_threads=1", "model.base_channels=16", "model.channel_mult=[1, 2]",
         "model.num_res_blocks=1", "model.attention_resolutions=[16]", "model.num_heads=2", "model.emb_dim=32",
         "batch.classes_per_batch=2", "batch.n_per_class=4", "data.subset={classes: [0, 1, 2], n_per_class: 32}",
         "train.log_every=1", "train.ckpt_every=1000", "train.sample_every=1000", "eval.every=1000",
         "eval.energy_distance.n_per_class=8", "train.grad_chunk=3"]


def _perturbed(seed=0, **kw):
    torch.manual_seed(seed)
    m = UNetTransportMap(**{**TINY, **kw})
    with torch.no_grad():                       # leave the zero init so that the map is not constant
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m


def test_shapes_and_zero_init():
    torch.manual_seed(0)
    m = UNetTransportMap(**TINY)
    z, c = torch.randn(5, 3, 8, 8), torch.tensor([0, 1, 2, 0, 1])
    x = m(z, c)
    assert x.shape == z.shape
    assert torch.count_nonzero(x) == 0, "last conv zero-initialised: F == 0 at step 0"
    assert m.sample(torch.tensor([0, 1])).shape == (2, 3, 8, 8)
    m2 = UNetTransportMap(**{**TINY, "in_channels": 6, "out_channels": 3})
    assert m2(torch.randn(2, 6, 8, 8), torch.tensor([0, 1])).shape == (2, 3, 8, 8)
    assert (m.in_channels, m.out_channels, m.input_size, m.num_classes) == (3, 3, 8, 3)


def test_full_size_configs_build_with_zero_output():
    """The App. I.2 model of both experiment configs: 140.0 M params, F == 0 at init, 3x32x32 out."""
    for path in (CFG_A, CFG_B):
        cfg = load_config(path)
        assert cfg["model"]["backbone"] == "unet"
        torch.manual_seed(0)
        m = build_model(cfg)
        assert isinstance(m, UNetTransportMap)
        assert abs(count_params(m) / 1e6 - 140.01) < 0.01, count_params(m)
        with torch.no_grad():
            x = m(torch.randn(2, 3, 32, 32), torch.tensor([3, 7]))
        assert x.shape == (2, 3, 32, 32) and torch.count_nonzero(x) == 0


def test_class_conditioning_after_one_step():
    torch.manual_seed(0)
    m = UNetTransportMap(**TINY)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    y = torch.randn(12, 3, 8, 8)
    lab = torch.arange(3).repeat_interleave(4)
    loss = energy_score_loss(m(torch.randn_like(y), lab), y, lab)
    loss.backward()
    opt.step()
    z = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        a, b, c = m(z, torch.tensor([0])), m(z, torch.tensor([1])), m(torch.randn(1, 3, 8, 8), torch.tensor([0]))
    assert a.abs().sum() > 0
    assert not torch.allclose(a, b), "output must depend on the class"
    assert not torch.allclose(a, c), "output must depend on the noise"


def test_gradients_reach_every_parameter():
    torch.manual_seed(0)
    m = UNetTransportMap(**TINY)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    y = torch.randn(12, 3, 8, 8)
    lab = torch.arange(3).repeat_interleave(4)
    for _ in range(4):
        opt.zero_grad()
        energy_score_loss(m(torch.randn_like(y), lab), y, lab).backward()
        opt.step()
    opt.zero_grad()
    energy_score_loss(m(torch.randn_like(y), lab), y, lab).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()
               or p.grad.abs().sum() == 0]
    assert not missing, missing


def test_chunked_two_pass_gradient_equals_one_pass():
    """tmap.train.forward_backward with grad_chunk > 0 gives the one-pass loss and gradients for the UNet."""
    from tmap.features import RandomConv
    from tmap.train import forward_backward
    m = _perturbed()
    g = torch.Generator().manual_seed(1)
    xl = torch.tensor([0] * 5 + [1] * 5 + [2] * 5)
    z = torch.randn(15, 3, 8, 8, generator=g)
    y = torch.randn(15, 3, 8, 8, generator=g)
    lc = {"beta": 1.0, "interaction_weight": 0.5}
    for enc in (None, RandomConv(3, 8)):
        out = []
        for chunk in (0, 4):
            m.zero_grad(set_to_none=True)
            loss, parts = forward_backward(m, enc, z, xl, y, xl, lc, 0.7, False, chunk)
            out.append((loss, [p.grad.clone() for p in m.parameters()]))
        (l0, g0), (l1, g1) = out
        assert torch.allclose(l0, l1, rtol=1e-5), (l0, l1)
        assert len(g0) == len(g1) == len(list(m.parameters()))
        for a, b in zip(g0, g1):
            assert torch.allclose(a, b, rtol=1e-4, atol=1e-6), (a - b).abs().max()


def test_checkpoint_round_trip(tmp_path):
    """train() a small UNet from the A config; tmap.sample.load_checkpoint_model rebuilds the same network from the
    checkpoint's stored config, for both raw and EMA weights."""
    from tmap.sample import load_checkpoint_model
    cfg = load_config(CFG_A, SMALL + ["train.max_steps=2", "train.snapshot_steps=[2]"])
    train(cfg, str(tmp_path / "run"))
    for ck_name in ("latest.pt", "step0000002.pt"):
        path = tmp_path / "run" / "ckpt" / ck_name
        raw, cfg2, ck = load_checkpoint_model(str(path), "raw")
        ema, _, _ = load_checkpoint_model(str(path), "ema")
        assert isinstance(raw, UNetTransportMap) and cfg2["model"]["backbone"] == "unet" and ck["step"] == 2
        for k, v in ck["model"].items():
            assert torch.equal(raw.state_dict()[k], v), k
        z, c = torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 0])
        with torch.no_grad():
            xr, xe = raw(z, c), ema(z, c)
        assert xr.abs().sum() > 0 and torch.isfinite(xr).all() and not torch.allclose(xr, xe)
        rebuilt = build_model(cfg2)
        rebuilt.load_state_dict(ck["ema"])
        with torch.no_grad():
            assert torch.equal(rebuilt.eval()(z, c), xe)
