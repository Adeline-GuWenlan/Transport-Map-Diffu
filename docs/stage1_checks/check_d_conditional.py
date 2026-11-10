"""Check (d): class-conditional grouping of the interaction term (1-D version of the example in
Proposition 'pooling the interaction term across classes is wrong').

Two classes, pi = 1/2 each: Q_0 = N(-a, 1), Q_1 = N(+a, 1), a = 3.  Conditional MLP F(z, c), z ~ N(0, 1).
  grouped : interaction term only between generated samples of the SAME class (correct estimator)
  pooled  : interaction term over all generated pairs regardless of class (the bug)
Prediction: grouped -> per-class std 1; pooled -> under-dispersed. Within the Gaussian family
N(+-a, s^2) the pooled loss is minimised at the s* computed below (about sqrt(1/7) = 0.378 for large a).
Outputs: results/check_d.json, ../figures/check_d_conditional.pdf
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import optimize
from scipy.stats import norm
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from es_core import BLUE, ORANGE, MUTED, energy_distance_u, torch_es_loss_grouped

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "results")
torch.set_num_threads(8)
A, STEPS, N, LR, SEEDS, N_EVAL = 3.0, 3000, 256, 1e-3, (0, 1), 4000


def folded(delta, s):
    return s * np.sqrt(2 / np.pi) * np.exp(-delta ** 2 / (2 * s * s)) + delta * (1 - 2 * norm.cdf(-delta / s))


def pooled_loss_gauss(s, a=A):
    # sum_c pi_c [E|X_c - Y_c| - pi_c/2 E|X_c - X_c'|] - 1/2 sum_{c != c'} pi_c pi_c' E|X_c - X_c'|
    return np.sqrt(2 / np.pi) * np.sqrt(1 + s * s) - s / (2 * np.sqrt(np.pi)) - 0.25 * folded(2 * a, np.sqrt(2) * s)


s_star = optimize.minimize_scalar(pooled_loss_gauss, bounds=(1e-3, 5), method="bounded").x
deriv_at_1 = (1 - np.exp(-A ** 2)) / (2 * np.sqrt(np.pi))
num_deriv = (pooled_loss_gauss(1 + 1e-5) - pooled_loss_gauss(1 - 1e-5)) / 2e-5


class CondMLP(nn.Module):
    def __init__(self, K=2, h=128):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(nn.Linear(1 + K, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, z, c):
        return self.net(torch.cat([z, nn.functional.one_hot(c, self.K).float()], 1))


def sample_data(n, rng):
    c = rng.integers(0, 2, n)
    y = np.where(c == 0, -A, A) + rng.standard_normal(n)
    return y[:, None], c


def run(mode, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(300 + seed)
    F = CondMLP()
    opt = torch.optim.Adam(F.parameters(), lr=LR)
    for _ in range(STEPS):
        y, c = sample_data(N, rng)
        y, c = torch.from_numpy(y).float(), torch.from_numpy(c).long()
        x = F(torch.randn(N, 1), c)                      # one generated sample per data point
        loss = torch_es_loss_grouped(x, c, y, c, beta=1.0, pool_interaction=(mode == "pooled"))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    res, xs_all = {}, {}
    with torch.no_grad():
        for cl in (0, 1):
            xs = F(torch.randn(N_EVAL, 1), torch.full((N_EVAL,), cl)).double().numpy()[:, 0]
            xs_all[cl] = xs
            tgt = (-A if cl == 0 else A) + rng.standard_normal(N_EVAL)
            res[f"class{cl}"] = dict(mean=float(xs.mean()), std=float(xs.std()),
                                     D1=float(energy_distance_u(xs, tgt)))
    return res, xs_all


if __name__ == "__main__":
    t0 = time.time()
    out = dict(a=A, gaussian_family_pooled_minimiser_s=float(s_star),
               pooled_deriv_at_s1_formula=float(deriv_at_1), pooled_deriv_at_s1_numeric=float(num_deriv),
               lambda_half_limit=float(np.sqrt(1 / 7)))
    keep = {}
    for mode in ("grouped", "pooled"):
        out[mode] = {}
        for seed in SEEDS:
            r, xs = run(mode, seed)
            out[mode][seed] = r
            if seed == SEEDS[0]:
                keep[mode] = xs
            print(mode, seed, r, flush=True)
    out["runtime_sec"] = time.time() - t0
    with open(os.path.join(RES, "check_d.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    grid = np.linspace(-7, 7, 400)
    bins = np.linspace(-7, 7, 90)
    for ax, mode in zip(axes, ("grouped", "pooled")):
        for cl, col in ((0, BLUE), (1, ORANGE)):
            ax.hist(keep[mode][cl], bins=bins, density=True, color=col, alpha=0.7, label=f"class {cl} samples")
            ax.plot(grid, norm.pdf(grid, -A if cl == 0 else A), color="#52514e", lw=1.2,
                    label="target density" if cl == 0 else None)
        r0 = out[mode][SEEDS[0]]
        ax.set_title(("interaction within class (correct)" if mode == "grouped" else
                      "interaction pooled over classes (bug)") +
                     f"\nsample std {r0['class0']['std']:.2f} / {r0['class1']['std']:.2f} (target 1)", fontsize=8)
        ax.set_xlabel("x")
    axes[0].set_ylabel("density (per class)")
    axes[0].legend(frameon=False, fontsize=7, loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "check_d_conditional.pdf"))
    print(json.dumps(out, indent=1, default=float))
