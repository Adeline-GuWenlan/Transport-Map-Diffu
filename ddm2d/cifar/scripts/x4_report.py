"""X4: E3 tables and figures from the committed evaluation JSONs (no model calls; CPU, seconds).

Inputs:  cifar/runs/<run>/eval/snap_*_ema_nfe*.json (protocol, sampler seed 0), xi_spread_snap_0030000_ema.json,
         cifar/runs/eval/seeds/*.json (sampler seeds 1, 2; scripts/eval_seeds.py), cifar/runs/eval/real_floor.json.
Outputs: cifar/runs/eval/fid_table.csv, fid_summary.json, fid_summary.md; cifar/figures/{fid_vs_nfe,fid_vs_step,xi_spread}.{png,pdf},
         cifar/figures/grids_nfe{10,100}.{png,pdf}.

    python cifar/scripts/x4_report.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["cifar_A", "cifar_B", "cifar_B_b1", "cifar_B_pH", "cifar_C", "cifar_C_dino"]
LABEL = {"cifar_A": "A (diffusion, v-target)", "cifar_B": "B (β=2, λ=0)", "cifar_B_b1": "B (β=1, λ=0)",
         "cifar_B_pH": "B (pseudo-Huber, λ=0)", "cifar_C": "C (β=0.1, λ=1)", "cifar_C_dino": "C + DINOv2 term"}
# validated categorical palette (dataviz skill, light mode; contrast WARN -> direct labels + markers + tables)
COLOR = dict(zip(RUNS, ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17"]))
MARK = dict(zip(RUNS, ["o", "s", "^", "v", "D", "P"]))
NFES = [2, 4, 8, 10, 15, 30, 100]
SNAPS = [10_000, 20_000, 30_000]
# paper, App. Table app_table:cifar10 (FID-50k, 1e6 steps, 3 seeds, best (beta, lambda) per NFE) -- context only
PAPER = {"nfe": [2, 4, 8, 10, 12, 15], "Diffusion": [80.77, 23.31, 7.53, 5.67, 4.83, 3.61],
         "Distributional": [29.33, 4.67, 3.21, 3.19, 3.03, 2.87]}
EV = ROOT / "runs" / "eval"
FIG = ROOT / "figures"


def load():
    rows = []
    for r in RUNS:
        for f in sorted(glob.glob(str(ROOT / "runs" / r / "eval" / "snap_*_ema_nfe*.json"))):
            d = json.load(open(f))
            if "fid" in d:
                rows.append({"run": r, "step": d["step"], "nfe": d["nfe"], "seed": d["seed"], "fid": d["fid"],
                             "pixel_std": d["pixel_std"], "file": os.path.relpath(f, ROOT)})
    for f in sorted(glob.glob(str(EV / "seeds" / "*.json"))):
        d = json.load(open(f))
        rows.append({"run": d["run"], "step": d["step"], "nfe": d["nfe"], "seed": d["seed"], "fid": d["fid"],
                     "pixel_std": d["pixel_std"], "file": os.path.relpath(f, ROOT)})
    return rows


def get(rows, run, step, nfe, seed=None):
    return [x["fid"] for x in rows if x["run"] == run and x["step"] == step and x["nfe"] == nfe
            and (seed is None or x["seed"] == seed)]


def main():
    rows = load()
    EV.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    with open(EV / "fid_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: (RUNS.index(x["run"]), x["step"], x["nfe"], x["seed"])))
    real = json.load(open(EV / "real_floor.json")) if (EV / "real_floor.json").exists() else None

    summ = {"final_seed0": {}, "final_seeds": {}, "snapshots_seed0": {}, "real_floor": real}
    for r in RUNS:
        summ["final_seed0"][r] = {k: (get(rows, r, 30_000, k, 0) or [None])[0] for k in NFES}
        summ["snapshots_seed0"][r] = {f"{s}_nfe{k}": (get(rows, r, s, k, 0) or [None])[0] for s in SNAPS for k in (10, 100)}
        summ["final_seeds"][r] = {}
        for k in NFES:
            v = get(rows, r, 30_000, k)
            if len(v) > 1:
                summ["final_seeds"][r][k] = {"n_seeds": len(v), "mean": float(np.mean(v)), "min": float(min(v)), "max": float(max(v)),
                                             "sd": float(np.std(v, ddof=1))}
    (EV / "fid_summary.json").write_text(json.dumps(summ, indent=1) + "\n")

    # markdown tables
    L = ["Final (30k) EMA, FID-10k, sampler seed 0 (the protocol). Lower is better.", "",
         "| run | " + " | ".join(f"NFE {k}" for k in NFES) + " |", "|---|" + "---|" * len(NFES)]
    for r in RUNS:
        L.append(f"| {LABEL[r]} | " + " | ".join(f"{summ['final_seed0'][r][k]:.2f}" for k in NFES) + " |")
    if any(summ["final_seeds"][r] for r in RUNS):
        ks = sorted({k for r in RUNS for k in summ["final_seeds"][r]})
        L += ["", "Final EMA, mean ± sd over sampler seeds (n given), [min, max]. One training seed per run.", "",
              "| run | " + " | ".join(f"NFE {k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
        for r in RUNS:
            c = []
            for k in ks:
                s = summ["final_seeds"][r].get(k)
                c.append(f"{s['mean']:.2f} ± {s['sd']:.2f} (n={s['n_seeds']}) [{s['min']:.2f}, {s['max']:.2f}]" if s else "–")
            L.append(f"| {LABEL[r]} | " + " | ".join(c) + " |")
    L += ["", "EMA snapshots (checkpoint selection), FID-10k, sampler seed 0.", "",
          "| run | 10k NFE 10 | 20k NFE 10 | 30k NFE 10 | 10k NFE 100 | 20k NFE 100 | 30k NFE 100 |", "|---|---|---|---|---|---|---|"]
    for r in RUNS:
        s = summ["snapshots_seed0"][r]
        L.append(f"| {LABEL[r]} | " + " | ".join(f"{s[f'{st}_nfe{k}']:.2f}" for k in (10, 100) for st in SNAPS) + " |")
    if real:
        L += ["", "Real-data FID-10k floor (real CIFAR-10 images vs the same train statistics).", "", "| set | FID-10k |", "|---|---|"]
        L += [f"| {x['set']} | {x['fid']:.2f} |" for x in real]
    (EV / "fid_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e6", "grid.linewidth": 0.6})

    # FID vs NFE: ours (left) and the paper's table (right, context only)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    for r in RUNS:
        y = [summ["final_seed0"][r][k] for k in NFES]
        ax[0].plot(NFES, y, color=COLOR[r], marker=MARK[r], ms=5, lw=2, label=LABEL[r])
        for k, s in summ["final_seeds"][r].items():
            ax[0].vlines(k, s["min"], s["max"], color=COLOR[r], lw=3, alpha=0.5)
        end = {"cifar_A": "A", "cifar_B": "B β=2", "cifar_B_b1": "B β=1 / pH", "cifar_C": "C", "cifar_C_dino": "C+DINOv2"}
        if r in end:   # B_pH ends on top of B_b1: one shared end label
            ax[0].annotate(end[r], (NFES[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                           color="#333333")
    if real:
        fl = min(x["fid"] for x in real)
        ax[0].axhline(fl, color="#777777", lw=1, ls="--")
        ax[0].text(2, fl * 1.08, f"real-data FID-10k floor ({fl:.1f})", fontsize=7.5, color="#555555")
    ax[0].set_xlim(1.8, 140)
    ax[0].set(xscale="log", yscale="log", xlabel="NFE", ylabel="FID-10k (final EMA, 30k steps)",
              title="This trial: 30k steps, batch 128, one training seed")
    ax[0].set_xticks(NFES, [str(k) for k in NFES])
    ax[0].legend(fontsize=7.5, frameon=False, loc="upper right")
    for name, c, m in (("Diffusion", "#257de0", "o"), ("Distributional", "#ee7fac", "D")):
        ax[1].plot(PAPER["nfe"], PAPER[name], color=c, marker=m, ms=5, lw=2, label=name)
    ax[1].set(xscale="log", yscale="log", xlabel="NFE", ylabel="FID-50k",
              title="Paper App. table (1e6 steps, FID-50k)\ncontext only, not comparable")
    ax[1].set_xticks(PAPER["nfe"], [str(k) for k in PAPER["nfe"]])
    for a in ax:
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_yticks([7, 10, 20, 30, 50, 100, 200], ["7", "10", "20", "30", "50", "100", "200"])
    ax[1].set_yticks([3, 5, 10, 20, 30, 50, 80], ["3", "5", "10", "20", "30", "50", "80"])
    ax[1].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"fid_vs_nfe.{e}", dpi=150)
    plt.close(fig)

    # FID vs training step (checkpoint selection)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    for i, k in enumerate((10, 100)):
        for r in RUNS:
            y = [summ["snapshots_seed0"][r][f"{s}_nfe{k}"] for s in SNAPS]
            ax[i].plot([s / 1000 for s in SNAPS], y, color=COLOR[r], marker=MARK[r], ms=5, lw=2, label=LABEL[r])
        ax[i].set(xlabel="training step (k)", ylabel="FID-10k (EMA)", title=f"NFE {k}", xticks=[10, 20, 30], yscale="log")
        ax[i].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax[i].set_yticks([10, 20, 30, 50, 100], ["10", "20", "30", "50", "100"])
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"fid_vs_step.{e}", dpi=150)
    plt.close(fig)

    # xi-spread vs t
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for r in RUNS[1:]:
        d = json.load(open(ROOT / "runs" / r / "eval" / "xi_spread_snap_0030000_ema.json"))["spread"]
        ts = sorted(d, key=float)
        ax.plot([float(t) for t in ts], [d[t] for t in ts], color=COLOR[r], marker=MARK[r], ms=4, lw=2, label=LABEL[r])
    ax.set(yscale="log", xlabel="t", ylabel="per-pixel std of x̂ over 8 ξ draws",
           title="ξ-spread (final EMA, 256 test images)")
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(FIG / f"xi_spread.{e}", dpi=150)
    plt.close(fig)

    # sample grids, 2 x 3 montage per NFE
    for k in (10, 100):
        fig, ax = plt.subplots(2, 3, figsize=(10, 7.2))
        for a, r in zip(ax.flat, RUNS):
            a.imshow(Image.open(ROOT / "runs" / r / "eval" / f"grid_snap_0030000_ema_nfe{k}.png"))
            fid0 = summ["final_seed0"][r][k]
            a.set_title(f"{LABEL[r]}\nFID-10k {fid0:.1f}", fontsize=9)
            a.axis("off")
        fig.suptitle(f"Final EMA, NFE {k}, sampler seed 0; one row per class (plane … truck)", fontsize=10)
        fig.tight_layout()
        for e in ("png", "pdf"):
            fig.savefig(FIG / f"grids_nfe{k}.{e}", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
