"""Check (c2): is the incomplete mode separation in check (c) a training-budget effect?
Corrected loss (beta = 1), same MLP and batch as check (c), seed 0, cosine LR decay, 3000 vs 20000 steps.
Metric besides D_1: fraction of samples within 0.3 (= 3 target std) of a mode centre; target ~0.989.
Outputs: results/check_c2.json, ../figures/check_c2_long.pdf
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from es_core import BLUE, energy_distance_u, eight_gaussians, torch_pow_dist
from check_c_toy2d import make_mlp, offdiag_mean

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "results")
torch.set_num_threads(8)
CENTERS = 2.0 * np.stack([np.cos(2 * np.pi * np.arange(8) / 8), np.sin(2 * np.pi * np.arange(8) / 8)], 1)


def near_mode(xs, r=0.3):
    d = np.sqrt(((xs[:, None, :] - CENTERS[None]) ** 2).sum(-1)).min(1)
    return float((d < r).mean())


def train(steps, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(100 + seed)
    F = make_mlp()
    opt = torch.optim.Adam(F.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for _ in range(steps):
        y = torch.from_numpy(eight_gaussians(256, rng)).float()
        x = F(torch.randn(256, 2))
        loss = torch_pow_dist(x, y, 1).mean() - 0.5 * offdiag_mean(torch_pow_dist(x, x, 1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        xs = F(torch.randn(4000, 2)).double().numpy()
    return dict(D1=float(energy_distance_u(xs, eight_gaussians(4000, rng))), frac_near_mode=near_mode(xs)), xs


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(5)
    out = {"target_frac_near_mode": near_mode(eight_gaussians(4000, rng))}
    keep = {}
    for steps in (3000, 20000):
        r, xs = train(steps)
        out[str(steps)] = r
        keep[steps] = xs
        print(steps, r, flush=True)
    out["runtime_sec"] = time.time() - t0
    with open(os.path.join(RES, "check_c2.json"), "w") as f:
        json.dump(out, f, indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.5), constrained_layout=True)
    tgt = eight_gaussians(1500, rng)
    for ax, steps in zip(axes, (3000, 20000)):
        xs = keep[steps]
        ax.scatter(tgt[:, 0], tgt[:, 1], s=1.0, color="#d9d8d3", lw=0, rasterized=True)
        ax.scatter(xs[:2000, 0], xs[:2000, 1], s=1.5, color=BLUE, lw=0, rasterized=True)
        ax.set_title(f"{steps} steps: $D_1$ = {out[str(steps)]['D1']:.3g}\n"
                     f"within 0.3 of a mode: {100 * out[str(steps)]['frac_near_mode']:.0f}%", fontsize=7.5)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(os.path.join(FIG, "check_c2_long.pdf"), dpi=200)
    print(json.dumps(out, indent=1))
