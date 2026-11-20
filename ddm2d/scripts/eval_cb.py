"""S4 checkerboard evaluation of the paper runs (runs/paper/cb_{A,B,C}_s{0,1,2}). Same protocol for every arm (docs/plan.md "Evaluation
protocol"): EMA weights of final.pt, 4096 samples, the same sampler seeds and reference sets for every arm, churn 1, safety 0.01.

Parts (raw JSON / npz into runs/paper/eval_cb/; plots and tables are made by scripts/plot_cb.py):
  mmd    Table 2 + metric-vs-NFE: NFE grid CURVE_NFE (contains Table 2's 5, 10, 50, 100, 1000), R repeats (sampler seed 1e6 + 1000 r + nfe,
         reference eval set k = r), clamp_low=True; NFE 1000 also with clamp_low=False (see src/ddm2d/samplers.py). RBF MMD^2 at sigma = 1
         (primary) and 0.5, 2 (metric sensitivity); unbiased and biased estimators. Generated samples of repeat 0 saved as .npz.
  null   p0 vs p0 (eval set 50 + r vs eval set r): the Monte Carlo floor of every estimator, for n in {1000, 2048, 4096, 10000}, sigma in
         {0.5, 1, 2}. Paired with `big` to test the V-statistic-floor hypothesis of docs/audit.md section 1.12.
  big    n = 10000 samples at Table 2's NFEs (3 repeats; sampler seed 7e6 + 1000 r + nfe; reference eval set 200 + r), sigma 1.
  snaps  EMA snapshots 10k ... 100k, NFE {5, 10, 100}, 2 repeats (training-stability check, as S3 did for the GMM).

Usage: python scripts/eval_cb.py <part> [--runs cb_A_s0 ...] [--threads 2] [--repeats 10]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "runs" / "paper" / "eval_cb"
ALL_RUNS = [f"cb_{a}_s{s}" for a in "ABC" for s in range(3)]
TABLE_NFE = [5, 10, 50, 100, 1000]                       # paper Table 2 (App. K.1)
CURVE_NFE = [5, 10, 20, 30, 50, 70, 100, 200, 500, 1000]
SIGMAS = [1.0, 0.5, 2.0]                                 # 1.0 = primary (the paper's GMM metric, §6.1); others = sensitivity


def mmd_all_sigmas(x, ref, sigmas):
    from ddm2d.mmd import rbf_mmd2
    out = {}
    for s in sigmas:
        r = rbf_mmd2(x, ref, s)
        tag = "" if s == 1.0 else f"_s{s:g}"
        out.update({f"mmd2_u{tag}": r["mmd2_u"], f"mmd2_b{tag}": r["mmd2_b"]})
        if s == 1.0:
            out.update(kxx_u=r["kxx_u"], kyy_u=r["kyy_u"], kxy=r["kxy"], n=r["n"], m=r["m"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["mmd", "null", "big", "snaps"])
    ap.add_argument("--runs", nargs="+", default=ALL_RUNS)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    from ddm2d import data as D
    from ddm2d.evaluate import sampler_seed
    from ddm2d.samplers import sample
    from ddm2d.train import load_ema_model

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def gen(model, n, nfe, seed, clamp_low=True):
        g = torch.Generator().manual_seed(seed)
        x, used = sample(model, n, nfe, churn=1.0, safety=0.01, generator=g, return_nfe=True, chunk=4096, clamp_low=clamp_low)
        assert used == nfe
        return x.numpy()

    if a.part == "mmd":
        configs = [(nfe, True) for nfe in CURVE_NFE] + [(1000, False)]
        for run in a.runs:
            model = load_ema_model(ROOT / "runs" / "paper" / run)
            rows, samples = [], {}
            for r in range(a.repeats):
                ref = D.eval_set("checkerboard", 4096, k=r)
                for nfe, cl in configs:
                    x = gen(model, 4096, nfe, sampler_seed(nfe, r), cl)
                    row = dict(run=run, nfe=nfe, repeat=r, clamp_low=cl, churn=1.0, safety=0.01,
                               n_finite=int(np.isfinite(x).all(1).sum()), frac_in_support=float(D.in_checkerboard(x).mean()))
                    row.update(mmd_all_sigmas(x, ref, SIGMAS))
                    rows.append(row)
                    if r == 0:
                        samples[f"nfe{nfe}" + ("" if cl else "_noclamp")] = x.astype(np.float32)
                print(run, "repeat", r, f"{time.time() - t0:.0f}s", flush=True)
            (OUT / f"mmd_{run}.json").write_text(json.dumps(rows, indent=0) + "\n")
            np.savez_compressed(OUT / f"samples_{run}.npz", **samples)
            print(run, [(q["nfe"], q["clamp_low"], f"{q['mmd2_u']:.2e}") for q in rows if q["repeat"] == 0], flush=True)

    elif a.part == "null":
        rows = []
        for n in [1000, 2048, 4096, 10000]:
            for r in range(a.repeats):
                x = D.eval_set("checkerboard", n, k=50 + r) if n == 4096 else D.eval_set("checkerboard", n, k=300 + r)
                ref = D.eval_set("checkerboard", n, k=r) if n == 4096 else D.eval_set("checkerboard", n, k=200 + r)
                row = dict(n=n, repeat=r, what="p0 sample vs p0 reference (independent eval sets)")
                row.update(mmd_all_sigmas(x, ref, SIGMAS))
                rows.append(row)
            print("null n", n, f"{time.time() - t0:.0f}s", flush=True)
        (OUT / "mmd_null.json").write_text(json.dumps(rows, indent=0) + "\n")

    elif a.part == "big":
        for run in a.runs:
            model = load_ema_model(ROOT / "runs" / "paper" / run)
            rows = []
            for r in range(min(a.repeats, 3)):
                ref = D.eval_set("checkerboard", 10000, k=200 + r)
                for nfe in TABLE_NFE:
                    x = gen(model, 10000, nfe, 7_000_000 + 1000 * r + nfe)
                    row = dict(run=run, nfe=nfe, repeat=r, clamp_low=True, frac_in_support=float(D.in_checkerboard(x).mean()))
                    row.update(mmd_all_sigmas(x, ref, [1.0]))
                    rows.append(row)
            (OUT / f"big_{run}.json").write_text(json.dumps(rows, indent=0) + "\n")
            print(run, "big", f"{time.time() - t0:.0f}s", flush=True)

    elif a.part == "snaps":
        for run in a.runs:
            rows = []
            for step in range(10_000, 100_001, 10_000):
                model = load_ema_model(ROOT / "runs" / "paper" / run, step=step)
                for r in range(2):
                    ref = D.eval_set("checkerboard", 4096, k=r)
                    for nfe in [5, 10, 100]:
                        x = gen(model, 4096, nfe, sampler_seed(nfe, r))
                        row = dict(run=run, step=step, nfe=nfe, repeat=r, frac_in_support=float(D.in_checkerboard(x).mean()))
                        row.update(mmd_all_sigmas(x, ref, [1.0]))
                        rows.append(row)
            (OUT / f"snaps_{run}.json").write_text(json.dumps(rows, indent=0) + "\n")
            print(run, "snaps", f"{time.time() - t0:.0f}s", flush=True)
    print("total", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
