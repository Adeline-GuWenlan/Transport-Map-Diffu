"""App. I.1 2D backbone (arXiv 2502.02483 v3, p. 33).

time t -> sinusoidal(1000 t, dim 2048) -> Linear(2048,2048) -> GELU -> Linear(2048,2048)          (2-layer time MLP)
time branch : 4 x [Linear(->64), GELU] on the 2048-d time embedding                                (preprocessing, 4 layers)
x branch    : 4 x [Linear(->64), GELU] on x_t (standard) or [x_t, xi] (distributional)              (preprocessing, 4 layers, in parallel)
concat(128) -> 4 x [Linear(->64), GELU] -> Linear(64 -> out)                                       (4 joint layers + output layer)
out = 2 (standard) or 4 (distributional; the first 2 dims are ignored, prediction = out[..., 2:4]).

Choices not fixed by the paper (docs/plan.md): input scaling 1000 t, max period 10000, [sin, cos] order, GELU after every hidden layer,
no GELU after the second time-MLP layer, PyTorch default init.

Exact speed-up: the time path depends only on t, shared by the m particles of a data point, so it is evaluated once per data point and broadcast.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = 2048, max_period: float = 10000.0, scale: float = 1000.0):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)
        self.scale = scale

    def forward(self, t):
        args = (self.scale * t)[:, None] * self.freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def _stack(d_in: int, width: int, n: int) -> nn.Sequential:
    layers = []
    for i in range(n):
        layers += [nn.Linear(d_in if i == 0 else width, width), nn.GELU()]
    return nn.Sequential(*layers)


class DenoiserMLP(nn.Module):
    """x_hat_theta(t, x_t) (stochastic=False) or x_hat_theta(t, x_t, xi) (stochastic=True)."""

    def __init__(self, stochastic: bool, data_dim: int = 2, width: int = 64, temb_dim: int = 2048, temb_hidden: int = 2048,
                 n_pre: int = 4, n_joint: int = 4):
        super().__init__()
        self.stochastic = stochastic
        self.data_dim = data_dim
        self.temb = SinusoidalEmbedding(temb_dim)
        self.time_mlp = nn.Sequential(nn.Linear(temb_dim, temb_hidden), nn.GELU(), nn.Linear(temb_hidden, temb_hidden))
        self.t_pre = _stack(temb_hidden, width, n_pre)
        x_in = 2 * data_dim if stochastic else data_dim
        self.x_pre = _stack(x_in, width, n_pre)
        self.joint = _stack(2 * width, width, n_joint)
        self.out_dim = x_in if stochastic else data_dim
        self.out = nn.Linear(width, self.out_dim)

    def time_features(self, t):
        return self.t_pre(self.time_mlp(self.temb(t)))

    def forward(self, t, xt, xi=None):
        """t: [n]; xt: [n, d].
        standard: returns [n, d].
        stochastic: xi is [n, m, d] (m particles per x_t) or [n, d]; returns [n, m, d] or [n, d].
        """
        h_t = self.time_features(t)                                   # [n, width], once per data point
        if not self.stochastic:
            assert xi is None
            h = torch.cat([h_t, self.x_pre(xt)], dim=-1)
            return self.out(self.joint(h))
        assert xi is not None, "stochastic model needs xi"
        squeeze = xi.ndim == 2
        if squeeze:
            xi = xi[:, None, :]
        n, m, d = xi.shape
        x_in = torch.cat([xt[:, None, :].expand(n, m, d), xi], dim=-1)  # [n, m, 2d]
        h = torch.cat([h_t[:, None, :].expand(n, m, h_t.shape[-1]), self.x_pre(x_in)], dim=-1)
        y = self.out(self.joint(h))[..., self.data_dim:]               # ignore the first d dims (App. I.1)
        return y[:, 0] if squeeze else y


def build_model(arm: str) -> DenoiserMLP:
    """Arm A: standard (no xi, output 2). Arms B and C: distributional (xi, output 4 -> last 2)."""
    return DenoiserMLP(stochastic=arm in ("B", "C"))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
