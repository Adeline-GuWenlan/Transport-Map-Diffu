"""Check (d2), secondary: class-conditional grouping of the interaction term.

Two classes on the 8-Gaussian circle: class 0 = even modes, class 1 = odd modes (pi = 1/2 each).
Conditional MLP F(z, c) = MLP([z, onehot(c)]).
  grouped : interaction term only between generated samples of the SAME class (correct estimator)
  pooled  : interaction term over all generated pairs regardless of class (a plausible bug)
Report per-class D_1 to Q_c and the within-mode spread (target std 0.1 per coordinate).
Outputs: results/check_d2.json, ../figures/check_d2_conditional_2d.pdf
"""
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from es_core import BLUE, ORANGE, INK2, energy_distance_u, eight_gaussians, torch_es_loss_grouped

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "results")
torch.set_num_threads(8)
STEPS, N, LR, SEEDS, N_EVAL = 3000, 256, 1e-3, (0, 1), 3000
CENTERS = 2.0 * np.stack([np.cos(2 * np.pi * np.arange(8) / 8), np.sin(2 * np.pi * np.arange(8) / 8)], 1)


def sample_data(n, rng):
    y, k = eight_gaussians(n, rng, return_labels=True)
    return y, k % 2


def sample_class(n, c, rng):
    out = []
    while sum(len(o) for o in out) < n:
        y, cc = sample_data(4 * n, rng)
        out.append(y[cc == c])
    return np.concatenate(out)[:n]


class CondMLP(nn.Module):
    def __init__(self, K=2, h=256):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(nn.Linear(2 + K, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))

    def forward(self, z, c):
        return self.net(torch.cat([z, nn.functional.one_hot(c, self.K).float()], 1))


def within_mode_std(xs):
    k = np.argmin(((xs[:, None, :] - CENTERS[None]) ** 2).sum(-1), 1)
    return float(np.sqrt(((xs - CENTERS[k]) ** 2).mean()))  # per-coordinate RMS around nearest centre


def run(mode, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(200 + seed)
    F = CondMLP()
    opt = torch.optim.Adam(F.parameters(), lr=LR)
    for step in range(STEPS):
        y, c = sample_data(N, rng)
        y, c = torch.from_numpy(y).float(), torch.from_numpy(c).long()
        # one generated sample per data point, same class label: groups = classes, m_c = n_c
        x = F(torch.randn(N, 2), c)
        loss = torch_es_loss_grouped(x, c, y, c, beta=1.0, pool_interaction=(mode == "pooled"))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    res, xs_all = {}, {}
    with torch.no_grad():
        for cl in (0, 1):
            xs = F(torch.randn(N_EVAL, 2), torch.full((N_EVAL,), cl)).double().numpy()
            xs_all[cl] = xs
            res[f"D1_class{cl}"] = float(energy_distance_u(xs, sample_class(N_EVAL, cl, rng)))
            res[f"within_mode_std_class{cl}"] = within_mode_std(xs)
    return res, xs_all


if __name__ == "__main__":
    t0 = time.time()
    out, keep = {}, {}
    for mode in ("grouped", "pooled"):
        out[mode] = {}
        for seed in SEEDS:
            r, xs = run(mode, seed)
            out[mode][seed] = r
            if seed == SEEDS[0]:
                keep[mode] = xs
            print(mode, seed, r, flush=True)
    rng = np.random.default_rng(3)
    out["target_within_mode_std"] = within_mode_std(sample_class(N_EVAL, 0, rng))
    out["runtime_sec"] = time.time() - t0

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5})
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.8))
    for ax, mode in zip(axes, ("grouped", "pooled")):
        for cl, col in ((0, BLUE), (1, ORANGE)):
            xs = keep[mode][cl][:1500]
            ax.scatter(xs[:, 0], xs[:, 1], s=1.2, color=col, lw=0, rasterized=True, label=f"class {cl}")
        ax.scatter(CENTERS[:, 0], CENTERS[:, 1], s=14, facecolors="none", edgecolors="#52514e", lw=0.7)
        r0 = out[mode][SEEDS[0]]
        ax.set_title(("interaction within class (correct)" if mode == "grouped" else
                      "interaction pooled over classes (bug)") +
                     f"\nmode std {r0['within_mode_std_class0']:.3f}/{r0['within_mode_std_class1']:.3f}"
                     f" (target {out['target_within_mode_std']:.3f})", fontsize=7.5)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=7, loc="center", markerscale=5)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, "check_d2_conditional_2d.pdf"), dpi=200)
    with open(os.path.join(RES, "check_d2.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
