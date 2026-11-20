"""S3 GMM evaluation of the paper runs (runs/paper/gmm_{A,B,C}_s{0,1,2}). Same protocol for every arm (docs/plan.md "Evaluation protocol"):
EMA weights of final.pt, 4096 samples, RBF sigma = 1, the same sampler / reference / x_t seeds for every arm.

Parts (each writes raw JSON/CSV into runs/paper/eval_gmm/; plotting is done by scripts/plot_gmm.py):
  mmd      Fig. 3 left: MMD^2 vs NFE {5,10,20,30,50,70,90,100}, R repeats (sampler seed 1e6 + 1000 r + nfe, reference eval set k = r),
           plus the reference-vs-reference null (eval set k vs k + 50) and generated samples (repeat 0) as .npz.
  snaps    MMD^2 vs NFE for the EMA snapshots at 10k ... 100k steps (low-noise check from S1).
  std      Fig. 3 right: posterior std vs t (paper grid linspace(0.01, 0.99, 10) and a fine grid), exact posterior and all runs.
  fig4     Fig. 4 draws: x0 in {(3,3), (-3,3)}, t in {0.99, 0.88, 0.45, 0.23, 0.01}, one x_t per panel (shared by every arm),
           4096 exact-posterior draws, 4096 draws of each run's x_hat; per-panel summary statistics.

Usage: python scripts/eval_gmm.py <part> [--runs gmm_A_s0 ...] [--threads 2] [--repeats 5]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "runs" / "paper" / "eval_gmm"
ALL_RUNS = [f"gmm_{a}_s{s}" for a in "ABC" for s in range(3)]
FIG4_X0 = [(3.0, 3.0), (-3.0, 3.0)]           # rows, in the order of docs/task_spec.md section 6
FIG4_T = [0.99, 0.88, 0.45, 0.23, 0.01]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["mmd", "snaps", "std", "fig4"])
    ap.add_argument("--runs", nargs="+", default=ALL_RUNS)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--nfe", nargs="+", type=int, default=None)
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    from ddm2d import data as D
    from ddm2d.evaluate import GMM_NFE, mmd_vs_nfe, posterior_std_curve
    from ddm2d.mmd import rbf_mmd2
    from ddm2d.posterior import posterior_mean_cov, sample_posterior
    from ddm2d.samplers import posterior_draws
    from ddm2d.schedule import forward_sample
    from ddm2d.train import load_ema_model

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    nfes = a.nfe or GMM_NFE

    if a.part == "mmd":
        for run in a.runs:
            model = load_ema_model(ROOT / "runs" / "paper" / run)
            rows, samples = mmd_vs_nfe(model, "gmm", nfes, n=4096, repeats=a.repeats, return_samples=True)
            for r in rows:
                r["run"] = run
            (OUT / f"mmd_{run}.json").write_text(json.dumps(rows, indent=0) + "\n")
            np.savez_compressed(OUT / f"samples_{run}.npz", **{f"nfe{k}": v.astype(np.float32) for k, v in samples.items()})
            print(run, [(r["nfe"], r["repeat"], f"{r['mmd2_u']:.2e}") for r in rows if r["repeat"] == 0], f"{time.time() - t0:.0f}s", flush=True)
        if a.runs == ALL_RUNS or "null" in os.environ.get("DDM2D_EXTRA", ""):
            null = []
            for r in range(a.repeats):
                res = rbf_mmd2(D.eval_set("gmm", 4096, k=50 + r), D.eval_set("gmm", 4096, k=r), 1.0)
                res.update(repeat=r, what="p0 sample (eval set 50+r) vs reference (eval set r)")
                null.append(res)
            (OUT / "mmd_null.json").write_text(json.dumps(null, indent=0) + "\n")
            print("null", [f"{q['mmd2_u']:.2e}/{q['mmd2_b']:.2e}" for q in null])

    elif a.part == "snaps":
        snap_nfe = a.nfe or [5, 10, 20, 50, 100]
        for run in a.runs:
            rows = []
            for step in range(10_000, 100_001, 10_000):
                model = load_ema_model(ROOT / "runs" / "paper" / run, step=step)
                for r in mmd_vs_nfe(model, "gmm", snap_nfe, n=4096, repeats=min(a.repeats, 2)):
                    r.update(run=run, step=step)
                    rows.append(r)
            (OUT / f"snaps_{run}.json").write_text(json.dumps(rows, indent=0) + "\n")
            print(run, f"{time.time() - t0:.0f}s", flush=True)

    elif a.part == "std":
        grids = {"paper": [float(x) for x in np.linspace(0.01, 0.99, 10)], "fine": [float(x) for x in np.round(np.arange(0.01, 0.995, 0.02), 4)]}
        res = {"protocol": "4096 (x0, x_t) pairs per t (x0 = eval set 100+i, x_t seed 2e6+i, shared by every arm), 8 draws per x_t, "
                           "per-coordinate std over the 8 draws (ddof 0), mean over the 2 coordinates and the 4096 x_t; "
                           "exact_std_mean = mean over x_t and coordinates of the exact posterior std",
               "grids": grids, "curves": {}}
        targets = ["exact"] + a.runs
        for name in targets:
            model = None if name == "exact" else load_ema_model(ROOT / "runs" / "paper" / name)
            res["curves"][name] = {g: posterior_std_curve(model, ts, n=4096, n_draws=8) for g, ts in grids.items()}
            print(name, [round(p["std_mean"], 4) for p in res["curves"][name]["paper"]], f"{time.time() - t0:.0f}s", flush=True)
        tag = "" if a.runs == ALL_RUNS else "_" + "_".join(a.runs)
        (OUT / f"posterior_std{tag}.json").write_text(json.dumps(res, indent=0) + "\n")

    elif a.part == "fig4":
        n = 4096
        xts = np.zeros((len(FIG4_X0), len(FIG4_T), 2), np.float32)
        for i, x0 in enumerate(FIG4_X0):
            for j, t in enumerate(FIG4_T):
                g = torch.Generator().manual_seed(3_000_000 + 10 * i + j)
                xts[i, j] = forward_sample(torch.tensor([x0], dtype=torch.float32), torch.tensor(t), torch.randn((1, 2), generator=g)).numpy()[0]
        draws = {"x0": np.array(FIG4_X0, np.float32), "t": np.array(FIG4_T, np.float32), "xt": xts}
        post = np.zeros((len(FIG4_X0), len(FIG4_T), n, 2), np.float32)
        for i in range(len(FIG4_X0)):
            for j, t in enumerate(FIG4_T):
                post[i, j] = sample_posterior(xts[i, j][None].astype(np.float64), t, n, np.random.default_rng(4_000_000 + 10 * i + j))[0]
        draws["posterior"] = post
        summ = []
        for run in a.runs:
            model = load_ema_model(ROOT / "runs" / "paper" / run)
            arr = np.zeros_like(post)
            for i in range(len(FIG4_X0)):
                for j, t in enumerate(FIG4_T):
                    g = torch.Generator().manual_seed(5_000_000 + 10 * i + j)       # same xi seeds for every run
                    arr[i, j] = posterior_draws(model, t, torch.from_numpy(xts[i, j][None]), n, generator=g)[0].numpy()
            draws[run] = arr
        for i in range(len(FIG4_X0)):
            for j, t in enumerate(FIG4_T):
                m, v = posterior_mean_cov(xts[i, j][None].astype(np.float64), t)
                base = {"row": i, "x0": FIG4_X0[i], "t": t, "xt": xts[i, j].tolist(), "exact_mean": m[0].tolist(), "exact_std": np.sqrt(v[0]).tolist(),
                        "exact_p_left": float((post[i, j][:, 0] < 0).mean())}
                for name in ["posterior"] + a.runs:
                    d = draws[name][i, j].astype(np.float64)
                    e = dict(base, who=name, mean=d.mean(0).tolist(), std=d.std(0).tolist(), frac_left=float((d[:, 0] < 0).mean()))
                    if name != "posterior":
                        e["mmd2_u_vs_posterior"] = rbf_mmd2(d, post[i, j], 1.0)["mmd2_u"]
                        e["mean_err_vs_exact_mean"] = float(np.linalg.norm(d.mean(0) - m[0]))
                    summ.append(e)
        np.savez_compressed(OUT / "fig4_draws.npz", **draws)
        (OUT / "fig4_summary.json").write_text(json.dumps(summ, indent=0) + "\n")
        print("fig4 done", f"{time.time() - t0:.0f}s")
    print("total", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
