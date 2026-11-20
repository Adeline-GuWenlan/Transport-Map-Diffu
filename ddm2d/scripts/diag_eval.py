"""S4 DIAGNOSTIC evaluation of the lower-lr fine-tunes (scripts/diag_finetune.py). Same protocol and seeds as Table 2 (scripts/eval_cb.py mmd:
EMA weights, 4096 samples, sampler seed 1e6 + 1000 r + nfe, reference eval set r, churn 1, clamp_low=True), at Table 2's NFEs, plus the
denoiser error vs the exact posterior at low t (as scripts/cb_extras.py dnerr). Output: runs/diag/diag_eval.json. Never part of Table 2.

Usage: python scripts/diag_eval.py runs/diag/cb_A_s0_lr0.0001_10000 runs/diag/cb_C_s0_lr0.0001_10000 [--repeats 10] [--threads 4] [--out FILE]
GMM run dirs (name starting with "gmm") get the Fig. 3 left NFE grid, the Fig. 3 right std protocol and the mean error vs the exact posterior.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="runs/diag/diag_eval.json")
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    from ddm2d import data as D
    from ddm2d.evaluate import CB_NFE, GMM_NFE, mmd_vs_nfe, posterior_std_curve
    from ddm2d.posterior import posterior_mean_cov
    from ddm2d.posterior_cb import posterior_mean, sample_posterior
    from ddm2d.samplers import posterior_draws
    from ddm2d.schedule import forward_sample
    from ddm2d.train import load_ema_model

    out = {}
    for d in a.dirs:
        model = load_ema_model(ROOT / d)
        if Path(d).name.startswith("gmm"):           # GMM: Fig. 3 left NFE grid + Fig. 3 right protocol + mean error vs the exact posterior
            rows = mmd_vs_nfe(model, "gmm", GMM_NFE, n=4096, repeats=a.repeats)
            ts = [0.01, 0.12, 0.45, 0.88, 0.99]
            std = posterior_std_curve(model, ts, n=4096, n_draws=8)
            err = []
            for i, t in enumerate(ts):
                x0 = torch.from_numpy(D.eval_set("gmm", 4096, k=400 + i))
                xt = forward_sample(x0, torch.tensor(t), torch.randn(x0.shape, generator=torch.Generator().manual_seed(8_000_000 + i)))
                m, _ = posterior_mean_cov(xt.double().numpy(), t)
                dr = posterior_draws(model, t, xt, 32, generator=torch.Generator().manual_seed(8_500_000 + i)).double().numpy()
                err.append(dict(t=t, rms_err_mean=float(np.sqrt(((dr.mean(1) - m) ** 2).sum(1).mean())), draw_std=float(dr.std(1).mean())))
            out[d] = {"mmd": rows, "denoiser": err, "posterior_std": std,
                      "summary": {str(n): float(np.mean([r["mmd2_u"] for r in rows if r["nfe"] == n])) for n in GMM_NFE}}
            print(d, out[d]["summary"], [(q["t"], round(q["std_mean"], 3), round(q["exact_std_mean"], 3)) for q in std], flush=True)
            continue
        rows = mmd_vs_nfe(model, "checkerboard", CB_NFE, n=4096, repeats=a.repeats)
        err = []
        for i, t in enumerate([0.01, 0.02, 0.05, 0.1, 0.3]):
            x0 = torch.from_numpy(D.eval_set("checkerboard", 4096, k=400 + i))
            g = torch.Generator().manual_seed(8_000_000 + i)
            xt = forward_sample(x0, torch.tensor(t), torch.randn(x0.shape, generator=g))
            m = posterior_mean(xt.double().numpy(), t)
            ex = sample_posterior(xt.double().numpy(), t, 32, np.random.default_rng(9_000_000 + i))
            dr = posterior_draws(model, t, xt, 32, generator=torch.Generator().manual_seed(8_500_000 + i)).double().numpy()
            err.append(dict(t=t, rms_err_mean=float(np.sqrt(((dr.mean(1) - m) ** 2).sum(1).mean())), draw_std=float(dr.std(1).mean()),
                            exact_std_32=float(ex.std(1).mean())))
        out[d] = {"mmd": rows, "denoiser": err,
                  "summary": {str(n): float(np.mean([r["mmd2_u"] for r in rows if r["nfe"] == n])) for n in CB_NFE}}
        print(d, out[d]["summary"], [(e["t"], round(e["rms_err_mean"], 3), round(e["draw_std"], 3)) for e in err], flush=True)
    (ROOT / a.out).write_text(json.dumps(out, indent=0) + "\n")


if __name__ == "__main__":
    main()
