"""Figures and tables for the S4 checkerboard evaluation, from the raw files of scripts/eval_cb.py (runs/paper/eval_cb/).
Colours as in scripts/plot_gmm.py (green = A standard diffusion, orange = B beta 2 lambda 0, blue = C beta 0.1 lambda 1).
Paper values (Table 2 of arXiv 2502.02483 v3, App. K.1; docs/paper_values/table2_paper_values.json) are drawn only in a separate panel or with
an explicit "paper" label, never as ours.

Usage: python scripts/plot_cb.py gt|grids|curve|table|snaps|dnerr|all
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PAPER = ROOT / "runs" / "paper"
EV = PAPER / "eval_cb"
FIG = ROOT / "figures" / "cb"
COL = {"A": "#2ca02c", "B": "#ff7f0e", "C": "#1f77b4"}
LAB = {"A": "A: standard diffusion (Eq. 6)", "B": r"B: $\hat x_\theta(t,X_t,\xi)$, $\beta=2,\lambda=0$",
       "C": r"C: $\hat x_\theta(t,X_t,\xi)$, $\beta=0.1,\lambda=1$"}
SEEDS = [0, 1, 2]
TABLE_NFE = [5, 10, 50, 100, 1000]
PAPER_T2 = json.loads((ROOT / "docs" / "paper_values" / "table2_paper_values.json").read_text())


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=130, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def rows_all():
    out = []
    for arm in "ABC":
        for s in SEEDS:
            p = EV / f"mmd_cb_{arm}_s{s}.json"
            if p.exists():
                out += json.loads(p.read_text())
    return out


def null_rows():
    p = EV / "mmd_null.json"
    return json.loads(p.read_text()) if p.exists() else []


def board(ax):
    for i in range(-4, 4, 2):
        for j in range(-4, 4, 2):
            if ((i // 2) + (j // 2)) % 2 == 0:
                ax.add_patch(plt.Rectangle((i, j), 2, 2, fill=False, ec="#999999", lw=0.5))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])


def gt():
    from ddm2d import data as D
    tr = D.train_set("checkerboard")
    ev = D.eval_set("checkerboard", 4096, k=0)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].scatter(*tr[:20000].T, s=0.5, alpha=0.4, c="k", rasterized=True)
    ax[0].set_title(f"training set (data seed 0)\nfirst 20,000 of {len(tr):,} points", fontsize=9)
    ax[1].scatter(*ev.T, s=0.8, alpha=0.5, c="k", rasterized=True)
    ax[1].set_title("evaluation reference set k=0\n(4096 points, never trained on)", fontsize=9)
    h = ax[2].hist2d(tr[:, 0], tr[:, 1], bins=100, range=[[-5, 5], [-5, 5]], cmap="Greys", density=True)
    fig.colorbar(h[3], ax=ax[2], fraction=0.046)
    ax[2].set_title("training-set density\n(100x100 bins)", fontsize=9)
    for a in ax:
        board(a)
    fig.suptitle("Checkerboard ground truth: FFJORD inf_train_gen('checkerboard') generator, support [-4,4]^2\n(a documented approximation of the paper's dataset, docs/audit.md §3-4)", fontsize=10)
    fig.tight_layout()
    save(fig, "ground_truth")


def grids():
    from ddm2d import data as D
    ref = D.eval_set("checkerboard", 4096, k=0)
    cols = [5, 10, 50, 100, 1000]
    for s in SEEDS:
        fig, axes = plt.subplots(3, len(cols) + 1, figsize=(2.6 * (len(cols) + 1), 8.0))
        for i, arm in enumerate("ABC"):
            d = np.load(EV / f"samples_cb_{arm}_s{s}.npz")
            rows = [q for q in json.loads((EV / f"mmd_cb_{arm}_s{s}.json").read_text()) if q["repeat"] == 0 and q["clamp_low"]]
            axes[i, 0].scatter(*ref.T, s=0.4, alpha=0.4, c="k", rasterized=True)
            axes[i, 0].set_title("ground truth (eval set)", fontsize=8)
            axes[i, 0].set_ylabel(LAB[arm], fontsize=8)
            for j, nfe in enumerate(cols):
                x = d[f"nfe{nfe}"]
                q = [r for r in rows if r["nfe"] == nfe][0]
                a = axes[i, j + 1]
                a.scatter(*x.T, s=0.4, alpha=0.4, c=COL[arm], rasterized=True)
                a.set_title(f"NFE {nfe}: MMD²_u {q['mmd2_u']:.1e}\nin support {q['frac_in_support']:.3f}", fontsize=8)
            for a in axes[i]:
                board(a)
                a.tick_params(labelsize=6)
        fig.suptitle(f"Checkerboard samples, training seed {s}: 4096 samples each (EMA, final.pt; sampler repeat 0, same seeds for every arm)", fontsize=10)
        fig.tight_layout()
        save(fig, f"samples_grid_seed{s}")


def _agg(rows, key, clamp=True):
    """{arm: {nfe: [per-seed mean over repeats]}}"""
    out = {}
    for arm in "ABC":
        out[arm] = {}
        for nfe in sorted({q["nfe"] for q in rows}):
            per = []
            for s in SEEDS:
                v = [q[key] for q in rows if q["run"] == f"cb_{arm}_s{s}" and q["nfe"] == nfe and q["clamp_low"] == clamp]
                if v:
                    per.append(float(np.mean(v)))
            if per:
                out[arm][nfe] = per
    return out


def curve():
    rows = rows_all()
    nu = [q for q in null_rows() if q["n"] == 4096]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for k, (key, title) in enumerate([("mmd2_u", "ours: unbiased MMD²_u (RBF σ=1, n=m=4096)"), ("mmd2_b", "ours: biased MMD²_b (V-statistic), same samples")]):
        ag = _agg(rows, key)
        ax = axes[k]
        for arm in "ABC":
            nf = sorted(ag[arm])
            m = np.array([np.mean(ag[arm][n]) for n in nf])
            lo = np.array([np.min(ag[arm][n]) for n in nf])
            hi = np.array([np.max(ag[arm][n]) for n in nf])
            ax.plot(nf, np.clip(m, 1e-6, None), "-o", ms=4, color=COL[arm], label=LAB[arm] + " (mean of 3 seeds; band = seed min-max)")
            ax.fill_between(nf, np.clip(lo, 1e-6, None), np.clip(hi, 1e-6, None), color=COL[arm], alpha=0.18, lw=0)
            nc = _agg(rows, key, clamp=False)[arm].get(1000)
            if nc:
                ax.plot([1000], [max(np.mean(nc), 1e-6)], marker="s", mfc="none", color=COL[arm], ms=8, ls="")
        if nu:
            v = [q[key] for q in nu]
            if key == "mmd2_u":       # the null of MMD²_u is centred at 0 (half the draws are negative): show its +2 sd level only
                ax.axhline(2 * np.std(v), color="#555555", ls=":", lw=1.2, label=f"+2 sd of the p0-vs-p0 null ({len(v)} draws; one 4096-sample evaluation)")
            else:
                ax.axhspan(min(v), max(v), color="#888888", alpha=0.25, lw=0, label=f"p0 vs p0 null, min-max of {len(v)} draws (V-statistic floor)")
        ax.plot([], [], marker="s", mfc="none", color="k", ls="", label="NFE 1000 with clamp_low=False")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("NFE (sampling steps)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=6.5, loc="upper right")
        ax.grid(alpha=0.3, which="both")
    ax = axes[2]
    for arm, key in [("A", "diffusion"), ("C", "distributional")]:
        ax.plot(TABLE_NFE, [PAPER_T2["values"][key][str(n)] for n in TABLE_NFE], "-o", color=COL[arm], ms=5,
                label="paper Table 2: " + {"A": "Diffusion", "C": "Distributional"}[arm])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("NFE")
    ax.set_title("PAPER values (Table 2, App. K.1; metric 'MMD', convention unknown)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    lo = min(a.get_ylim()[0] for a in axes)
    hi = max(a.get_ylim()[1] for a in axes)
    for a in axes:
        a.set_ylim(max(lo, 3e-5), hi)
    fig.tight_layout()
    save(fig, "metric_vs_nfe")


def table():
    rows = rows_all()
    # raw CSV
    keys = sorted({k for q in rows for k in q})
    front = ["run", "nfe", "clamp_low", "repeat", "mmd2_u", "mmd2_b", "frac_in_support"]
    keys = front + [k for k in keys if k not in front]
    with open(EV / "cb_mmd_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for q in rows:
            w.writerow(q)
    summ = {"protocol": "per seed: mean over 10 repeats (sampler seed 1e6+1000r+nfe, reference eval set r, 4096 vs 4096 samples); "
                        "EMA weights of final.pt; churn 1; safety 0.01; clamp_low=True unless noted. Paper values are NOT ours.",
            "paper_table2": PAPER_T2["values"], "ours": {}}
    for key in ["mmd2_u", "mmd2_b", "mmd2_u_s0.5", "mmd2_b_s0.5", "mmd2_u_s2", "mmd2_b_s2", "frac_in_support"]:
        for clamp in [True, False]:
            ag = _agg(rows, key, clamp)
            for arm in "ABC":
                for nfe, per in ag[arm].items():
                    e = summ["ours"].setdefault(f"{arm}|{nfe}|{'clamp' if clamp else 'noclamp'}", {})
                    rep = [q[key] for q in rows if q["run"].startswith(f"cb_{arm}_") and q["nfe"] == nfe and q["clamp_low"] == clamp]
                    e[key] = {"per_seed": per, "mean": float(np.mean(per)), "sd_over_seeds": float(np.std(per, ddof=1)) if len(per) > 1 else None,
                              "min": float(np.min(per)), "max": float(np.max(per)), "sd_over_repeats_pooled": float(np.std(rep, ddof=1))}
    nu = null_rows()
    summ["null"] = {}
    for n in sorted({q["n"] for q in nu}):
        sub = [q for q in nu if q["n"] == n]
        summ["null"][str(n)] = {k: {"mean": float(np.mean([q[k] for q in sub])), "sd": float(np.std([q[k] for q in sub], ddof=1)),
                                    "min": float(np.min([q[k] for q in sub])), "max": float(np.max([q[k] for q in sub]))}
                                for k in sub[0] if k.startswith("mmd2")}
        summ["null"][str(n)]["draws"] = len(sub)
    big = {}
    for arm in "ABC":
        for s in SEEDS:
            p = EV / f"big_cb_{arm}_s{s}.json"
            if p.exists():
                for q in json.loads(p.read_text()):
                    big.setdefault(f"{arm}|{q['nfe']}", {}).setdefault(s, []).append((q["mmd2_u"], q["mmd2_b"]))
    summ["big_n10000"] = {k: {"per_seed_mmd2_u": [float(np.mean([a for a, _ in v[s]])) for s in sorted(v)],
                              "per_seed_mmd2_b": [float(np.mean([b for _, b in v[s]])) for s in sorted(v)]} for k, v in big.items()}
    (EV / "table2_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    # markdown Table 2
    L = ["| NFE | arm | MMD²_u seed 0 / 1 / 2 | MMD²_u mean (sd over seeds) | MMD²_b mean | in-support frac | paper Table 2 (NOT ours) | numeric comparison |",
         "|---|---|---|---|---|---|---|---|"]
    for nfe in TABLE_NFE:
        for arm in "ABC":
            e = summ["ours"].get(f"{arm}|{nfe}|clamp")
            if not e:
                continue
            u = e["mmd2_u"]
            pk = {"A": "diffusion", "C": "distributional"}.get(arm)
            pv = PAPER_T2["values"][pk][str(nfe)] if pk else None
            L.append(f"| {nfe} | {arm} | {' / '.join(f'{v:.2e}' for v in u['per_seed'])} | {u['mean']:.2e} ({u['sd_over_seeds']:.1e}) | "
                     f"{e['mmd2_b']['mean']:.2e} | {e['frac_in_support']['mean']:.3f} | {f'{pv:.2e}' if pv else '—'} | "
                     f"{'unverified' if pv else '—'} |")
        for arm in "ABC":
            e = summ["ours"].get(f"{arm}|{nfe}|noclamp")
            if e:
                u = e["mmd2_u"]
                L.append(f"| {nfe} (clamp_low=False) | {arm} | {' / '.join(f'{v:.2e}' for v in u['per_seed'])} | {u['mean']:.2e} ({u['sd_over_seeds']:.1e}) | "
                         f"{e['mmd2_b']['mean']:.2e} | {e['frac_in_support']['mean']:.3f} | — | — |")
    (EV / "table2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def snaps():
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for k, nfe in enumerate([5, 10, 100]):
        ax = axes[k]
        for arm in "ABC":
            for s in SEEDS:
                p = EV / f"snaps_cb_{arm}_s{s}.json"
                if not p.exists():
                    continue
                r = json.loads(p.read_text())
                st = sorted({q["step"] for q in r})
                y = [np.mean([q["mmd2_u"] for q in r if q["step"] == t and q["nfe"] == nfe]) for t in st]
                ax.plot(st, np.clip(y, 1e-6, None), "-o", ms=3, color=COL[arm], alpha=[1, 0.7, 0.45][s], label=f"{arm} seed {s}")
        ax.set_yscale("log")
        ax.set_title(f"EMA snapshots, NFE {nfe} (MMD²_u, mean of 2 repeats)", fontsize=9)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=6, ncol=3)
    fig.tight_layout()
    save(fig, "snapshots_mmd2")


def dnerr():
    r = json.loads((EV / "denoiser_err.json").read_text())
    ts = sorted({q["t"] for q in r})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ex = [[q for q in r if q["t"] == t][0] for t in ts]
    axes[0].plot(ts, [q["exact_std_32"] for q in ex], "k-", lw=2, label="exact posterior std (per coordinate)")
    axes[0].plot(ts, [q["exact_mc_err_32"] for q in ex], "k:", lw=1.2, label="MC error of a 32-draw mean (exact draws)")
    for arm in "ABC":
        for s in SEEDS:
            y = [np.mean([q["rms_err_mean"] for q in r if q["t"] == t and q["run"] == f"cb_{arm}_s{s}" and q["which"] == "clamped"]) for t in ts]
            axes[0].plot(ts, y, "-o", ms=3, color=COL[arm], alpha=[1, 0.7, 0.45][s], label=(LAB[arm] if s == 0 else None))
            y = [np.mean([q["draw_std"] for q in r if q["t"] == t and q["run"] == f"cb_{arm}_s{s}" and q["which"] == "clamped"]) for t in ts]
            if arm != "A":
                axes[1].plot(ts, y, "-o", ms=3, color=COL[arm], alpha=[1, 0.7, 0.45][s], label=(LAB[arm] if s == 0 else None))
    axes[1].plot(ts, [q["exact_std_32"] for q in ex], "k-", lw=2, label="exact posterior (32 draws)")
    axes[0].set_title("RMS error of E[x̂] vs the exact posterior mean\n(A: x̂; B, C: mean of 32 draws); t < 0.01 fed as 0.01", fontsize=9)
    axes[1].set_title("spread of the 32 draws (per-coordinate std)", fontsize=9)
    for a in axes:
        a.set_xscale("log")
        a.set_yscale("log")
        a.axvspan(1e-3, 0.01, color="#dddddd", alpha=0.5, lw=0)
        a.set_xlabel("diffusion time t (grey: below the training range [0.01, 0.99])")
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=7)
    fig.suptitle("Checkerboard denoisers vs the exact posterior (4096 x_t per t; runs/paper/eval_cb/denoiser_err.json)", fontsize=10)
    fig.tight_layout()
    save(fig, "denoiser_error_vs_t")


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("gt", gt), ("table", table), ("curve", curve), ("grids", grids), ("snaps", snaps), ("dnerr", dnerr)]:
        if part in (name, "all"):
            fn()
