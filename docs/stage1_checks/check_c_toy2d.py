"""Check (c): train a small MLP map F: R^2 -> R^2, z ~ N(0, I_2), to 8 Gaussians on a circle,
with the corrected loss and with several readings of the user's formula.

All losses are minibatch estimates with n = 256 data points and M = 256 generated samples
(one pooled group, unconditional), Adam lr 1e-3 with cosine decay to 0, 3000 steps, 3 seeds.
Evaluation: unbiased D_1 (beta = 1 energy distance) between 4000 model samples and 4000 fresh
target samples.  Outputs: results/check_c.json, ../figures/check_c_samples.pdf,
../figures/check_c_curves.pdf
"""
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from es_core import (BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED, INK2, MUTED,
                     energy_distance_u, eight_gaussians, torch_pow_dist)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "results")
torch.set_num_threads(8)
STEPS, N, M, LR, SEEDS = 3000, 256, 256, 1e-3, (0, 1, 2)
EVAL_EVERY, N_EVAL = 250, 4000


def offdiag_mean(K):
    m = K.shape[0]
    return (K.sum() - torch.diagonal(K).sum()) / (m * (m - 1))


# x = F(z), x2 = F(z') with z' independent of z; y, y2 independent data batches.
VARIANTS = {
    "es_beta1": dict(label=r"corrected, $\beta=1$",
                     f=lambda x, x2, y, y2: torch_pow_dist(x, y, 1).mean() - 0.5 * offdiag_mean(torch_pow_dist(x, x, 1))),
    "es_beta1.5": dict(label=r"corrected, $\beta=1.5$",
                       f=lambda x, x2, y, y2: torch_pow_dist(x, y, 1.5).mean() - 0.5 * offdiag_mean(torch_pow_dist(x, x, 1.5))),
    "literal_rawdiff": dict(label=r"literal: $-\frac{1}{2} E[F(x)-F(x')]$",
                            f=lambda x, x2, y, y2: torch_pow_dist(x, y, 1).mean() - 0.5 * (x - x2).mean()),
    "data_pairs": dict(label=r"$x,x'$ from data: $-\frac{1}{2} E\|y-y'\|$",
                       f=lambda x, x2, y, y2: torch_pow_dist(x, y, 1).mean() - 0.5 * (y - y2).norm(dim=1).mean()),
    "beta2_squared": dict(label=r"squared norms ($\beta=2$)",
                          f=lambda x, x2, y, y2: torch_pow_dist(x, y, 2).mean() - 0.5 * offdiag_mean(torch_pow_dist(x, x, 2))),
    "coef1_no_half": dict(label=r"coefficient 1 instead of $\frac{1}{2}$",
                          f=lambda x, x2, y, y2: torch_pow_dist(x, y, 1).mean() - 1.0 * offdiag_mean(torch_pow_dist(x, x, 1))),
    "mixed_norm_sq": dict(label=r"$\|\cdot\|$ then $-\frac{1}{2}\|\cdot\|^2$",
                          f=lambda x, x2, y, y2: torch_pow_dist(x, y, 1).mean() - 0.5 * offdiag_mean(torch_pow_dist(x, x, 2))),
}


def make_mlp():
    return nn.Sequential(nn.Linear(2, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                         nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2))


def d1_eval(F, rng_np, n_eval):
    with torch.no_grad():
        xs = F(torch.randn(n_eval, 2)).double().numpy()
    if not np.all(np.isfinite(xs)):
        return float("nan"), xs
    return float(energy_distance_u(xs, eight_gaussians(n_eval, rng_np))), xs


def run(name, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(100 + seed)
    F = make_mlp()
    opt = torch.optim.Adam(F.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)   # same schedule for every variant
    f = VARIANTS[name]["f"]
    curve, losses = [], []
    for step in range(1, STEPS + 1):
        y = torch.from_numpy(eight_gaussians(N, rng)).float()
        y2 = torch.from_numpy(eight_gaussians(N, rng)).float()
        x = F(torch.randn(M, 2))
        x2 = F(torch.randn(M, 2))
        loss = f(x, x2, y, y2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        if step % EVAL_EVERY == 0 or step == 1:
            curve.append((step, d1_eval(F, rng, 1500)[0]))
    d1, xs = d1_eval(F, rng, N_EVAL)
    fin = np.isfinite(xs).all(1)
    stats = dict(D1=d1, final_loss=losses[-1], loss_min=float(np.nanmin(losses)),
                 sample_mean=xs[fin].mean(0).tolist(), sample_std=xs[fin].std(0).tolist(),
                 mean_radius=float(np.linalg.norm(xs[fin], axis=1).mean()),
                 frac_outside_4=float((np.abs(xs) > 4).any(1).mean()), curve=curve)
    return stats, xs


if __name__ == "__main__":
    t0 = time.time()
    rng0 = np.random.default_rng(7)
    tgt_a, tgt_b = eight_gaussians(N_EVAL, rng0), eight_gaussians(N_EVAL, rng0)
    from scipy.spatial.distance import cdist
    Kyy = cdist(tgt_a, tgt_a)
    baselines = dict(
        target_vs_target=float(energy_distance_u(tgt_a, tgt_b)),
        gaussian_source_vs_target=float(energy_distance_u(rng0.standard_normal((N_EVAL, 2)), tgt_a)),
        # point mass at the origin (the geometric median of the symmetric target):
        # D_1(delta_0, Q) = 2 E||Y|| - E||Y - Y'||
        point_mass_origin_vs_target=float(2 * np.linalg.norm(tgt_a, axis=1).mean()
                                          - (Kyy.sum() - np.trace(Kyy)) / (N_EVAL * (N_EVAL - 1))),
    )
    results, samples = {}, {}
    for name in VARIANTS:
        results[name] = {}
        for seed in SEEDS:
            ts = time.time()
            st, xs = run(name, seed)
            st["sec"] = time.time() - ts
            results[name][seed] = st
            if seed == SEEDS[0]:
                samples[name] = xs
            print(f"{name:16s} seed {seed}: D1={st['D1']:.4f} loss={st['final_loss']:.4f} "
                  f"std={np.round(st['sample_std'], 3)} out={st['frac_outside_4']:.3f} ({st['sec']:.1f}s)", flush=True)
        d = np.array([results[name][s]["D1"] for s in SEEDS])
        results[name]["summary"] = dict(D1_mean=float(np.mean(d)), D1_sd=float(np.std(d, ddof=1)))
    np.savez_compressed(os.path.join(RES, "check_c_samples_seed0.npz"), target=tgt_a, **samples)
    out = dict(config=dict(steps=STEPS, n=N, M=M, lr=LR, seeds=SEEDS, n_eval=N_EVAL,
                           mlp="2-256-256-256-2 SiLU", target="8 Gaussians, radius 2, std 0.1"),
               baselines=baselines, target_std=tgt_a.std(0).tolist(), results=results,
               runtime_sec=time.time() - t0)
    with open(os.path.join(RES, "check_c.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)

    # ------------------------------------------------------------- figures
    import plot_c
    plot_c.main()

    out["runtime_sec"] = time.time() - t0
    with open(os.path.join(RES, "check_c.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(dict(baselines=baselines,
                          summary={k: results[k]["summary"] for k in VARIANTS}), indent=1))
    print("runtime", time.time() - t0)
