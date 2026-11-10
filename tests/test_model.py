"""DiT transport map: shapes, zero output at init, gradients reach every trainable parameter."""
import torch

from tmap.energy_score import energy_score_loss
from tmap.model import TransportMap, build_model
from tmap.train import load_config

TINY = dict(input_size=16, patch_size=4, in_channels=1, hidden_size=32, depth=2, num_heads=2, num_classes=4)


def test_shapes_and_zero_init_smoke_config(root):
    cfg = load_config("configs/smoke_cpu.yaml")
    m = build_model(cfg)
    z = torch.randn(6, 1, 32, 32)
    c = torch.tensor([0, 1, 2, 3, 4, 9])
    x = m(z, c)
    assert x.shape == z.shape
    assert torch.count_nonzero(x) == 0, "adaLN-Zero: F == 0 at step 0"
    n = sum(p.numel() for p in m.parameters())
    assert abs(n / 1e6 - 7.39) < 0.01, n            # stage 2: 7.39 M params


def test_shapes_rgb_and_out_channels():
    m = TransportMap(**{**TINY, "in_channels": 3})
    assert m(torch.randn(2, 3, 16, 16), torch.tensor([0, 1])).shape == (2, 3, 16, 16)
    m2 = TransportMap(**{**TINY, "in_channels": 6, "out_channels": 3})
    x = m2(torch.randn(2, 6, 16, 16), torch.tensor([0, 1]))
    assert x.shape == (2, 3, 16, 16) and torch.count_nonzero(x) == 0
    assert m.sample(torch.tensor([0, 1, 2])).shape == (3, 3, 16, 16)


def test_gradients_reach_every_parameter():
    torch.manual_seed(0)
    m = TransportMap(**TINY)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    y = torch.randn(16, 1, 16, 16)
    lab = torch.arange(4).repeat_interleave(4)
    # step 0: zero-init head => only final_layer gets gradient; after a few steps every parameter does
    for _ in range(3):
        loss = energy_score_loss(m(torch.randn_like(y), lab), y, lab)
        opt.zero_grad()
        loss.backward()
        opt.step()
    loss = energy_score_loss(m(torch.randn_like(y), lab), y, lab)
    opt.zero_grad()
    loss.backward()
    trainable = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    missing = [n for n, p in trainable if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
    assert not missing, missing
    frozen = [n for n, p in m.named_parameters() if not p.requires_grad]
    assert frozen == ["net.pos_embed"], frozen      # DiT's fixed sin-cos position embedding


def test_output_depends_on_class_and_noise():
    torch.manual_seed(1)
    m = TransportMap(**TINY)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    z = torch.randn(1, 1, 16, 16)
    a, b = m(z, torch.tensor([0])), m(z, torch.tensor([1]))
    c = m(torch.randn(1, 1, 16, 16), torch.tensor([0]))
    assert not torch.allclose(a, b) and not torch.allclose(a, c)
