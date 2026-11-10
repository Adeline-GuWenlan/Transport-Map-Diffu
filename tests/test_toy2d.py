"""Stage 1 check (c) as a test: MLP 2 -> 256^3 -> 2, z ~ N(0, I_2), 8 Gaussians on a circle (radius 2, std 0.1),
N = M = 256, Adam 1e-3 with cosine decay, 3000 steps. Must reach a low held-out energy distance in under a minute.
Stage 1 reference (docs/stage1_checks/results): corrected beta = 1 reaches D1 = 0.0098 +- 0.0005; the source
Gaussian scores 0.241 and a point mass 1.47.
"""
import math
import time

import numpy as np
import torch
from torch import nn

from tmap.energy_score import energy_distance_u, energy_score_loss


def eight_gaussians(n, rng, radius=2.0, std=0.1):
    k = rng.integers(0, 8, size=n)
    ang = 2 * np.pi * k / 8
    return radius * np.stack([np.cos(ang), np.sin(ang)], 1) + std * rng.standard_normal((n, 2))


def test_toy2d_trains_to_low_energy_distance():
    t0 = time.time()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    F = nn.Sequential(nn.Linear(2, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                      nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2))
    steps, N = 3000, 256
    opt = torch.optim.Adam(F.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    lab = torch.zeros(N, dtype=torch.long)
    for _ in range(steps):
        y = torch.from_numpy(eight_gaussians(N, rng)).float()
        loss = energy_score_loss(F(torch.randn(N, 2)), y, lab)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        x = F(torch.randn(4000, 2))
    y = torch.from_numpy(eight_gaussians(4000, rng)).float()
    d = energy_distance_u(x, y)
    noise = energy_distance_u(torch.randn(4000, 2), y)
    elapsed = time.time() - t0
    print(f"toy 2-D: D1 {d:.4f} (source Gaussian {noise:.3f}) in {elapsed:.1f} s")
    assert elapsed < 60, elapsed
    assert d < 0.03, d                 # stage 1: 0.0098 +- 0.0005; noise ~0.24
    assert x.std(0).min() > 1.0, "collapse check: samples must spread over the circle (data std ~1.4)"
