"""Quick end-to-end evaluation of smoke runs (NOT paper results): MMD vs NFE, posterior std at a few t, sample scatter figure.

Usage: python scripts/smoke_eval.py --runs runs/smoke/<run> [...] --dataset gmm --out runs/smoke/<name>_eval [--nfe 5 20 100] [--threads 8]
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
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dataset", default="gmm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nfe", nargs="+", type=int, default=[5, 20, 100])
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    torch.set_num_threads(a.threads)
    from ddm2d import data as D
    from ddm2d.evaluate import mmd_vs_nfe, posterior_std_curve
    from ddm2d.train import load_ema_model

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"WARNING": "smoke evaluation, not a paper-scale result", "runs": {}}
    fig, axes = plt.subplots(len(a.runs), len(a.nfe) + 1, figsize=(3 * (len(a.nfe) + 1), 3 * len(a.runs)), squeeze=False)
    ref = D.eval_set(a.dataset, a.n, 0)
    for i, rd in enumerate(a.runs):
        model = load_ema_model(rd)
        name = Path(rd).name
        rows, samples = mmd_vs_nfe(model, a.dataset, a.nfe, n=a.n, return_samples=True)
        entry = {"mmd": rows}
        if a.dataset == "gmm":
            entry["posterior_std"] = posterior_std_curve(model, [0.05, 0.25, 0.5, 0.75, 0.95], n=1024)
        res["runs"][name] = entry
        axes[i, 0].scatter(ref[:, 0], ref[:, 1], s=1, alpha=0.3, c="tab:blue")
        axes[i, 0].set_title("reference p_0", fontsize=8)
        axes[i, 0].set_ylabel(name, fontsize=7)
        for j, nfe in enumerate(a.nfe):
            x = samples[nfe]
            r = [q for q in rows if q["nfe"] == nfe][0]
            axes[i, j + 1].scatter(x[:, 0], x[:, 1], s=1, alpha=0.3, c="tab:green")
            axes[i, j + 1].set_title(f"NFE {nfe}  MMD2_u {r['mmd2_u']:.2e}", fontsize=8)
        lim = (-6, 6) if a.dataset == "gmm" else (-5, 5)
        ylim = (0, 6) if a.dataset == "gmm" else (-5, 5)
        for ax in axes[i]:
            ax.set_xlim(*lim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
    fig.suptitle("SMOKE evaluation (EMA weights) - not paper results", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=110)
    fig.savefig(out / "samples.pdf")
    (out / "eval.json").write_text(json.dumps(res, indent=1) + "\n")
    for name, e in res["runs"].items():
        print(name, [(r["nfe"], round(r["mmd2_u"], 5)) for r in e["mmd"]],
              [(p["t"], round(p["std_mean"], 3), round(p["exact_std_mean"], 3)) for p in e.get("posterior_std", [])])


if __name__ == "__main__":
    main()
