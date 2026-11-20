"""S4 checkerboard investigations (diagnostics; never mixed into the Table 2 numbers):

  oracle  The same sampler (ddm2d.samplers.sample: uniform grid, safety clamp, churn 1 and 0) driven by the EXACT checkerboard posterior
          (src/ddm2d/posterior_cb.py) instead of a network: 'oracle_mean' returns E[X0 | x_t] (ideal arm A / B), 'oracle_draw' a draw from
          p(X0 | x_t) (ideal arm C). Measures the sampler's own discretisation error, i.e. the floor for any trained network.
          NFE 1000 is run with clamp_low True and False.
  churn0  All 9 trained checkerboard runs with churn 0 instead of 1 at Table 2's NFEs (same seeds for every arm; sensitivity only).

Usage: python scripts/cb_extras.py oracle|churn0 [--repeats 5] [--threads 2]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "runs" / "paper" / "eval_cb"
CURVE_NFE = [5, 10, 20, 30, 50, 70, 100, 200, 500, 1000]
TABLE_NFE = [5, 10, 50, 100, 1000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["oracle", "churn0"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    from ddm2d.evaluate import mmd_vs_nfe
    from ddm2d.posterior_cb import posterior_mean, sample_posterior
    from ddm2d.train import load_ema_model

    class Oracle:
        def __init__(self, draw: bool, seed: int):
            self.stochastic = draw
            self.rng = np.random.default_rng(seed)

        def eval(self):
            return self

        def __call__(self, t, xt, xi=None):
            x = xt.double().numpy()
            out = sample_posterior(x, float(t), 1, self.rng)[:, 0] if self.stochastic else posterior_mean(x, float(t))
            return torch.from_numpy(out).to(xt.dtype)

    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    if a.part == "oracle":
        for churn in (1.0, 0.0):
            for kind in ("oracle_mean", "oracle_draw"):
                for cl in (True, False):
                    nfes = CURVE_NFE if cl else [1000]
                    res = mmd_vs_nfe(Oracle(kind == "oracle_draw", seed=7 + int(churn)), "checkerboard", nfes, n=4096, repeats=a.repeats,
                                     churn=churn, clamp_low=cl)
                    for r in res:
                        r["who"] = kind
                    rows += res
                    print(kind, churn, cl, [(n, f"{np.mean([r['mmd2_u'] for r in res if r['nfe'] == n]):.2e}") for n in nfes], flush=True)
        (OUT / "oracle_mmd.json").write_text(json.dumps(rows, indent=0) + "\n")
    else:
        for arm in "ABC":
            for s in range(3):
                run = f"cb_{arm}_s{s}"
                res = mmd_vs_nfe(load_ema_model(ROOT / "runs" / "paper" / run), "checkerboard", TABLE_NFE, n=4096, repeats=a.repeats, churn=0.0)
                for r in res:
                    r["who"] = run
                rows += res
                print(run, [(n, f"{np.mean([r['mmd2_u'] for r in res if r['nfe'] == n]):.2e}") for n in TABLE_NFE], flush=True)
        (OUT / "churn0_mmd.json").write_text(json.dumps(rows, indent=0) + "\n")


if __name__ == "__main__":
    main()
