"""Check (f): DiT zero-initialises its output layer (adaLN-Zero; facebookresearch/DiT models.py,
initialize_weights), so F(z, c) == 0 for every z at step 0 and every generated pair has distance exactly 0.
Here: the 2-D toy MLP with a zero-initialised last layer, trained with the beta = 1 loss using
  (i)  the safe distance (gradient 0 at 0, es_core.torch_pow_dist),
  (ii) the naive sqrt of squared distances (NaN at 0).
Output: results/check_f.json
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from es_core import energy_distance_u, eight_gaussians, torch_pow_dist
from check_c_toy2d import make_mlp

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
torch.set_num_threads(8)


def naive_dist(a, b):
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1).sqrt()


def offdiag_mean(K):
    m = K.shape[0]
    return (K.sum() - torch.diagonal(K).sum()) / (m * (m - 1))


def run(dist, steps=3000, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    F = make_mlp()
    torch.nn.init.zeros_(F[-1].weight); torch.nn.init.zeros_(F[-1].bias)
    opt = torch.optim.Adam(F.parameters(), lr=1e-3)
    first_grad_nan, n_nan_steps = None, 0
    for step in range(steps):
        y = torch.from_numpy(eight_gaussians(256, rng)).float()
        x = F(torch.randn(256, 2))
        loss = dist(x, y).mean() - 0.5 * offdiag_mean(dist(x, x))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        bad = any(torch.isnan(p.grad).any() for p in F.parameters())
        if step == 0:
            first_grad_nan = bool(bad)
            x0_spread = float(x.detach().std())
        if bad:
            n_nan_steps += 1
        opt.step()
    with torch.no_grad():
        xs = F(torch.randn(4000, 2)).double().numpy()
    d1 = float(energy_distance_u(xs, eight_gaussians(4000, rng))) if np.isfinite(xs).all() else float("nan")
    return dict(first_step_grad_has_nan=first_grad_nan, steps_with_nan_grad=n_nan_steps,
                generated_std_at_step0=x0_spread, final_D1=d1)


out = {"safe_dist": run(lambda a, b: torch_pow_dist(a, b, 1.0)),
       "naive_sqrt": run(naive_dist)}
with open(os.path.join(RES, "check_f.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
