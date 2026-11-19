"""Shared evaluation protocol (docs/plan.md, "Evaluation protocol"). Identical for every arm: EMA weights, the same sample count, the same
sampler/reference seeds per (NFE, repeat), the same NFE grid.

Seeds: sampler generator seed = 1_000_000 + 1000 * repeat + nfe; reference set = data.eval_set(dataset, n, k=repeat).
"""
from __future__ import annotations

import numpy as np
import torch

from . import data as D
from .mmd import rbf_mmd2
from .posterior import posterior_mean_cov, sample_posterior
from .samplers import posterior_draws, sample
from .schedule import forward_sample

GMM_NFE = [5, 10, 20, 30, 50, 70, 90, 100]           # Fig. 3 left x-ticks (source figures/mmd_2d.pdf)
CB_NFE = [5, 10, 50, 100, 1000]                        # Table 2


def sampler_seed(nfe: int, repeat: int) -> int:
    return 1_000_000 + 1000 * repeat + nfe


def mmd_vs_nfe(model, dataset: str, nfes, n: int = 4096, repeats: int = 1, churn: float = 1.0, safety: float = 0.01, sigma: float = 1.0,
               return_samples: bool = False, clamp_low: bool = True):
    rows, samples = [], {}
    for r in range(repeats):
        ref = D.eval_set(dataset, n, k=r)
        for nfe in nfes:
            g = torch.Generator().manual_seed(sampler_seed(nfe, r))
            x, used = sample(model, n, nfe, churn=churn, safety=safety, generator=g, return_nfe=True, chunk=4096, clamp_low=clamp_low)
            assert used == nfe
            x = x.numpy()
            res = rbf_mmd2(x, ref, sigma)
            res.update(nfe=nfe, repeat=r, churn=churn, clamp_low=clamp_low, n_finite=int(np.isfinite(x).all(1).sum()))
            if dataset == "checkerboard":
                res["frac_in_support"] = float(D.in_checkerboard(x).mean())
            rows.append(res)
            if return_samples and r == 0:
                samples[nfe] = x
    return (rows, samples) if return_samples else rows


def posterior_std_curve(model, ts, n: int = 4096, n_draws: int = 8, seed: int = 2_000_000):
    """Fig. 3 right protocol: 4096 (x0, x_t) pairs per t, 8 draws, per-coordinate std (ddof 0) averaged over coordinates and pairs.
    model=None -> exact posterior (8 draws, same protocol); also returns the exact posterior std (sqrt of the exact variance)."""
    out = []
    for i, t in enumerate(ts):
        x0 = torch.from_numpy(D.eval_set("gmm", n, k=100 + i))
        g = torch.Generator().manual_seed(seed + i)
        xt = forward_sample(x0, torch.tensor(float(t)), torch.randn(x0.shape, generator=g))
        if model is None:
            draws = sample_posterior(xt.numpy(), float(t), n_draws, np.random.default_rng(seed + i))
        else:
            draws = posterior_draws(model, float(t), xt, n_draws, generator=g).numpy()
        std = draws.std(axis=1, ddof=0).mean()
        _, var = posterior_mean_cov(xt.numpy(), float(t))
        out.append({"t": float(t), "std_mean": float(std), "exact_std_mean": float(np.sqrt(var).mean())})
    return out
