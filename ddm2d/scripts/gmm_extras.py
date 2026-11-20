"""S3 investigations for Fig. 3 left (extras, not paper protocol changes):

  oracle  The same sampler (ddm2d.samplers.sample, uniform grid, safety clamp, churn 1 and 0) driven by the EXACT GMM posterior instead of a
          network: 'oracle_mean' returns E[X0 | x_t] (the ideal arm A / B), 'oracle_draw' returns a draw from p(X0 | x_t) (the ideal arm C).
          MMD^2 vs NFE then measures the discretisation error of the sampler alone, i.e. the best any trained network can do.
  grid    Oracle E[X0|x_t] (ideal A) with an alternative time grid, to test whether the paper's lower A/B values at NFE 10-30 could come from
          a different grid: 'safety' = N denoiser times equally spaced on [0.01, 0.99] (t_N = 0.99 ... t_1 = 0.01), then t_0 = 0.
  churn0  All 9 trained runs with churn 0 (DDIM-type bridge) instead of 1, same seeds and NFE grid for every arm (sensitivity only).

Usage: python scripts/gmm_extras.py oracle|grid|churn0 [--repeats 5] [--threads 2]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "runs" / "paper" / "eval_gmm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["oracle", "churn0", "grid"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    from ddm2d.evaluate import GMM_NFE, mmd_vs_nfe
    from ddm2d.posterior import posterior_mean_cov, sample_posterior
    from ddm2d.train import load_ema_model

    class Oracle:
        def __init__(self, draw: bool, seed: int):
            self.stochastic = draw
            self.rng = np.random.default_rng(seed)

        def eval(self):
            return self

        def __call__(self, t, xt, xi=None):
            x = xt.double().numpy()
            if self.stochastic:
                out = sample_posterior(x, float(t), 1, self.rng)[:, 0]
            else:
                out = posterior_mean_cov(x, float(t))[0]
            return torch.from_numpy(out).to(xt.dtype)

    rows = []
    if a.part == "oracle":
        for churn in (1.0, 0.0):
            for kind in ("oracle_mean", "oracle_draw"):
                res = mmd_vs_nfe(Oracle(kind == "oracle_draw", seed=7 + int(churn)), "gmm", GMM_NFE, n=4096, repeats=a.repeats, churn=churn)
                for r in res:
                    r["who"] = kind
                rows += res
                print(kind, churn, [(n, f"{np.mean([r['mmd2_u'] for r in res if r['nfe'] == n]):.2e}") for n in GMM_NFE], flush=True)
        (OUT / "oracle_mmd.json").write_text(json.dumps(rows, indent=0) + "\n")
    elif a.part == "grid":
        from ddm2d import data as D
        from ddm2d.evaluate import sampler_seed
        from ddm2d.mmd import rbf_mmd2
        from ddm2d.schedule import bridge_coeffs
        orc = Oracle(False, 0)
        for churn in (1.0, 0.0):
            for nfe in (5, 10, 20, 30, 50, 100):
                grid = [0.0] + [float(x) for x in np.linspace(0.01, 0.99, nfe)]
                for r in range(a.repeats):
                    g = torch.Generator().manual_seed(sampler_seed(nfe, r))
                    x = torch.randn((4096, 2), generator=g)
                    for k in range(nfe - 1, -1, -1):
                        s_, t_ = grid[k], grid[k + 1]
                        x0h = orc(t_, x)
                        c1, c2, sd = bridge_coeffs(s_, t_, churn)
                        x = c1 * x + c2 * x0h + sd * torch.randn(x.shape, generator=g)
                    res = rbf_mmd2(x.numpy(), D.eval_set("gmm", 4096, k=r), 1.0)
                    res.update(who="oracle_mean_safety_grid", nfe=nfe, repeat=r, churn=churn)
                    rows.append(res)
                print("safety grid churn", churn, nfe, f"{np.mean([q['mmd2_u'] for q in rows if q['nfe'] == nfe and q['churn'] == churn]):.2e}", flush=True)
        (OUT / "oracle_grid_mmd.json").write_text(json.dumps(rows, indent=0) + "\n")
    else:
        for arm in "ABC":
            for s in range(3):
                run = f"gmm_{arm}_s{s}"
                res = mmd_vs_nfe(load_ema_model(ROOT / "runs" / "paper" / run), "gmm", GMM_NFE, n=4096, repeats=a.repeats, churn=0.0)
                for r in res:
                    r["who"] = run
                rows += res
                print(run, [(n, f"{np.mean([r['mmd2_u'] for r in res if r['nfe'] == n]):.2e}") for n in GMM_NFE], flush=True)
        (OUT / "churn0_mmd.json").write_text(json.dumps(rows, indent=0) + "\n")


if __name__ == "__main__":
    main()
