"""ddm2dx X3 figures (figures/ext/, PNG + PDF) from runs/ext/eval/*.csv|json (written by scripts/ext_eval.py summary).

Usage: python scripts/ext_plot.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "runs" / "ext" / "eval"
FIG = ROOT / "figures" / "ext"
sys.path.insert(0, str(ROOT / "src"))
from ddm2d.data import GMM_SPECS  # noqa: E402

BLUE_RAMP = ["#b6d5f8", "#84b9f3", "#519ced", "#338ced", "#257de0", "#2370cd", "#1757ac", "#0f4a96", "#0e3d7e"]
CAT = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
GRAY, INK, INK2 = "#8a8a8a", "#191919", "#666666"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 2, "legend.frameon": False})

LAM0 = [("gmm_B_b0.1", 0.1), ("gmm_B_b0.5", 0.5), ("gmm_B_b1", 1.0), ("gmm_B_b1.5", 1.5), ("gmm_B_b1.99", 1.99), ("gmm_B", 2.0),
        ("gmm_B_b3", 3.0), ("gmm_B_b3.5", 3.5)]
LAM1 = [("gmm_C", 0.1), ("gmm_C_b0.5", 0.5), ("gmm_C_b1", 1.0), ("gmm_C_b1.5", 1.5), ("gmm_C_b1.99", 1.99), ("gmm_C_b2", 2.0)]
E4_GMM = ["gmm_C", "gmm_C_L1b1", "gmm_C_L1b0.5", "gmm_C_pH0.1", "gmm_C_pH1", "gmm_C_imq", "gmm_C_rbf", "gmm_C_lap", "gmm_B", "gmm_B_L1b1", "gmm_B_pH0.1"]
E4_9W = ["gmm9w_A", "gmm9w_B", "gmm9w_B_b1", "gmm9w_B_L1b1", "gmm9w_B_pH0.1", "gmm9w_C", "gmm9w_C_b1.5", "gmm9w_C_L1b1", "gmm9w_C_L1b0.5", "gmm9w_C_pH0.1",
         "gmm9w_C_pH1", "gmm9w_C_imq", "gmm9w_C_rbf", "gmm9w_C_lap"]


def rd(fn):
    return list(csv.DictReader(open(EV / fn)))


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def per_run_mmd():
    """(group, nfe) -> list of per-seed mean MMD^2_u"""
    out = defaultdict(list)
    for r in rd("mmd_per_run.csv"):
        if int(r["seed"]) in (0, 1):           # matched seeds 0, 1 (seed 2: separate table)
            out[(r["group"], int(r["nfe"]))].append(float(r["mmd2_u"]))
    return out


def have(g):
    return (EV / "fig4" / f"draws_{g}_s0.npz").exists()


def main():
    global E4_9W
    E4_9W = [g for g in E4_9W if have(g)]          # gmm9w_C_b1.5 (X3 extra) appears once evaluated
    pm = per_run_mmd()
    nulls = json.loads((EV / "nulls.json").read_text())
    floor = abs(nulls["gmm"]["mmd2_u_mean"]) + 2 * nulls["gmm"]["mmd2_u_sd"] / np.sqrt(10)   # for a mean of 10 repeats
    nfes = sorted({k[1] for k in pm})

    # F1: MMD^2 vs beta at NFE 5 / 10 / 100
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, nfe in zip(axs, [5, 10, 100]):
        a = pm[("gmm_A", nfe)]
        ax.axhspan(min(a), max(a), color=GRAY, alpha=0.18, lw=0)
        ax.axhline(np.mean(a), color=GRAY, ls="--", lw=1.5, label="A (standard diffusion), seed range")
        ax.axhline(floor, color=INK2, ls=":", lw=1, label="null floor, 10-repeat mean (p0 vs p0: |mean| + 2 sd/√10)")
        for fam, col, lab in [(LAM0, CAT[0], "λ = 0 (B family)"), (LAM1, CAT[1], "λ = 1 (C family)")]:
            bs = [b for _, b in fam]
            v = [pm[(g, nfe)] for g, _ in fam]
            TOP = 0.1
            mv = [np.mean(x) for x in v]
            ok = [np.isfinite(m) and m < TOP for m in mv]
            ax.plot([b for b, o in zip(bs, ok) if o], [m for m, o in zip(mv, ok) if o], "-o", color=col, ms=5, label=lab)
            for b, x, o in zip(bs, v, ok):
                xs = [min(q, TOP * 0.8) if np.isfinite(q) else TOP * 0.8 for q in x]
                if o:
                    ax.plot([b, b], [min(xs), max(xs)], color=col, lw=1)
                ax.plot([b] * len(xs), xs, "_", color=col, ms=8, mew=1.2)
                if not o:
                    ax.plot([b], [TOP * 0.8], "^", color=col, ms=7)
                    ax.annotate("β = 1.99 / 2, λ = 1:\ndiverged or collapsed ▲", (b, TOP * 0.8), textcoords="offset points", xytext=(-4, -4),
                                ha="right", va="top", fontsize=6, color=INK2)
            ax.set_ylim(3e-5, TOP)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([0.1, 0.5, 1, 2, 3.5])
        ax.set_xticklabels(["0.1", "0.5", "1", "2", "3.5"])
        ax.set_title(f"NFE {nfe}", color=INK)
        ax.set_xlabel("β")
    axs[0].set_ylabel("MMD²_u (RBF σ=1), mean of 10 repeats\nline = seed mean, ticks = seeds 0, 1")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Paper GMM: MMD² vs β (log y)", color=INK)
    save(fig, "mmd_vs_beta")

    # F2: MMD^2 vs NFE per beta
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, fam, title in [(axs[0], LAM0, "λ = 0 (B family)"), (axs[1], LAM1, "λ = 1 (C family)")]:
        a = [np.mean(pm[("gmm_A", n)]) for n in nfes]
        ax.plot(nfes, a, "--", color=GRAY, label="A")
        ramp = BLUE_RAMP[-len(fam):] if len(fam) < len(BLUE_RAMP) else BLUE_RAMP
        for (g, b), col in zip(fam, np.linspace(1, len(BLUE_RAMP) - 1, len(fam)).astype(int)):
            ax.plot(nfes, [np.mean(pm[(g, n)]) for n in nfes], "-o", ms=4, color=BLUE_RAMP[col], label=f"β = {b:g}")
        ax.axhline(floor, color=INK2, ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-5)
        ax.set_ylim(-1e-5, 0.1)
        ax.set_xticks(nfes)
        ax.set_xticklabels([str(n) for n in nfes])
        ax.set_xlabel("NFE")
        ax.set_title(title, color=INK)
        ax.legend(fontsize=7, ncol=2)
    axs[0].set_ylabel("MMD²_u, mean over 2 seeds × 10 repeats")
    save(fig, "mmd_vs_nfe_beta")

    # F3: posterior std vs t per beta (fine grid)
    sr = [r for r in rd("posterior_std.csv") if r["grid"] == "fine" and int(r["seed"]) in (0, 1)]
    curves = defaultdict(lambda: defaultdict(list))
    exact = {}
    for r in sr:
        curves[r["group"]][float(r["t"])].append(float(r["std_mean"]))
        if r["dataset"] == "gmm":
            exact[float(r["t"])] = float(r["exact_std_mean"])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, fam, title in [(axs[0], LAM0, "λ = 0 (B family)"), (axs[1], LAM1, "λ = 1 (C family)")]:
        ts = sorted(exact)
        ax.plot(ts, [exact[t] for t in ts], color=INK, lw=2.5, label="exact posterior")
        for (g, b), col in zip(fam, np.linspace(1, len(BLUE_RAMP) - 1, len(fam)).astype(int)):
            ax.plot(ts, [np.mean(curves[g][t]) for t in ts], color=BLUE_RAMP[col], lw=1.6, label=f"β = {b:g}")
        ax.set_ylim(0, 2.2)
        ax.set_xlabel("t")
        ax.set_title(title, color=INK)
        ax.legend(fontsize=7)
    axs[0].set_ylabel("std of x̂ over 8 ξ draws (seed mean)\nβ = 2, λ = 1 is off scale (1e5–1e8)")
    save(fig, "posterior_std_beta")

    # F3b: posterior std vs t, E2 and E4
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    sets = [("gmm", ["gmm_A", "gmm_B", "gmm_C", "gmm_C_L1b1", "gmm_C_pH0.1", "gmm_C_imq", "gmm_C_rbf", "gmm_C_lap"]),
            ("gmm2w", ["gmm2w_A", "gmm2w_B", "gmm2w_C"]),
            ("gmm9w", ["gmm9w_A", "gmm9w_B", "gmm9w_C", "gmm9w_C_L1b1", "gmm9w_C_pH0.1", "gmm9w_C_imq", "gmm9w_C_rbf", "gmm9w_C_lap"])]
    for ax, (ds, gs) in zip(axs, sets):
        ex = {}
        for r in sr:
            if r["dataset"] == ds:
                ex[float(r["t"])] = float(r["exact_std_mean"])
        ts = sorted(ex)
        ax.plot(ts, [ex[t] for t in ts], color=INK, lw=2.5, label="exact")
        for g, col in zip(gs, CAT):
            ax.plot(ts, [np.mean(curves[g][t]) for t in ts], color=col, lw=1.4, label=g.split("_", 1)[1])
        ax.set_title(ds, color=INK)
        ax.set_xlabel("t")
        ax.legend(fontsize=6.5)
    axs[0].set_ylabel("posterior std of x̂")
    save(fig, "posterior_std_e2_e4")

    # F4: Fig. 4-style grids
    def grid(name, ds, rows, xrow, title):
        T = [0.99, 0.88, 0.45, 0.23, 0.01]
        means = GMM_SPECS[ds][0]
        fig, axs = plt.subplots(len(rows) + 1, len(T), figsize=(1.9 * len(T), 1.75 * (len(rows) + 1)), squeeze=False)
        d0 = np.load(EV / "fig4" / f"draws_{rows[0]}_s0.npz")
        for j in range(len(T)):
            for i, run in enumerate([None] + rows):
                ax = axs[i, j]
                dd = d0["posterior"][xrow, j] if run is None else np.load(EV / "fig4" / f"draws_{run}_s0.npz")["model"][xrow, j]
                ax.scatter(dd[:, 0], dd[:, 1], s=1.5, color=INK if run is None else CAT[0], alpha=0.35, lw=0)
                ax.scatter(means[:, 0], means[:, 1], s=8, marker="+", color=CAT[1], lw=0.8)
                lim = 7 if ds == "gmm9w" else 6
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim if ds == "gmm9w" else -1, lim if ds == "gmm9w" else 6)
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                if i == 0:
                    ax.set_title(f"t = {T[j]}", fontsize=8, color=INK)
                if j == 0:
                    ax.set_ylabel("exact" if run is None else run.split("_", 1)[1], fontsize=7, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.3, w_pad=0.3)
        fig.suptitle(title, color=INK, fontsize=9, y=0.995)
        save(fig, name)

    grid("fig4_gmm_lambda0", "gmm", [g for g, _ in LAM0], 0, "Paper GMM, x0 = (3, 3): λ = 0 family (seed 0), 1024 of 4096 draws per panel")
    grid("fig4_gmm_lambda1", "gmm", [g for g, _ in LAM1], 0, "Paper GMM, x0 = (3, 3): λ = 1 family (seed 0)")
    grid("fig4_gmm_e4", "gmm", ["gmm_A"] + E4_GMM, 0, "Paper GMM, x0 = (3, 3): other distances ρ (seed 0)")
    for xr in (0, 1):
        grid(f"fig4_gmm2w_x{xr}", "gmm2w", ["gmm2w_A", "gmm2w_B", "gmm2w_C"], xr,
             f"gmm2w, x0 = mean {xr} (weight {GMM_SPECS['gmm2w'][2][xr]:.1f}), seed 0")
        grid(f"fig4_gmm9w_x{xr}", "gmm9w", E4_9W, xr, f"gmm9w, x0 = mean k={xr} (weight {GMM_SPECS['gmm9w'][2][xr]:.2f}), seed 0")

    # F5: final-sample mode weights (E2), relative error per mode, with the multinomial null band
    raw = [r for r in rd("mmd_raw.csv") if int(r["seed"]) in (0, 1)]
    fr = defaultdict(list)
    for r in raw:
        fr[(r["group"], int(r["nfe"]))].append([float(v) for v in r["frac"].split()])
    for ds, gs in [("gmm2w", ["gmm2w_A", "gmm2w_B", "gmm2w_C"]), ("gmm9w", [g for g in ["gmm9w_A", "gmm9w_B", "gmm9w_C", "gmm9w_B_b1", "gmm9w_C_b1.5"] if have(g)])]:
        w = GMM_SPECS[ds][2]
        nm = json.loads((EV / "null" / f"{ds}.json").read_text())["modes"]
        nf = np.array([m["frac"] for m in nm])
        lo, hi = np.quantile(nf, 0.025, 0), np.quantile(nf, 0.975, 0)
        K = len(w)
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
        for ax, nfe in zip(axs, [5, 10, 100]):
            x = np.arange(K)
            ax.bar(x, (hi - lo) / w, bottom=(lo - w) / w, width=0.9, color="#e6e6e6", label="null 95% (4096 true samples)")
            bw = 0.8 / len(gs)
            for i, (g, col) in enumerate(zip(gs, CAT)):
                f = np.array(fr[(g, nfe)])          # all seeds x repeats
                m = (f.mean(0) - w) / w
                ax.bar(x - 0.4 + bw * (i + 0.5), m, width=bw * 0.85, color=col, label=g.split("_", 1)[1])
            ax.axhline(0, color=INK, lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{k}\n{w[k]:.2f}" for k in range(K)], fontsize=7)
            ax.set_xlabel("mode k / weight π_k")
            ax.set_title(f"NFE {nfe}", color=INK)
        axs[0].set_ylabel("(p̂_k − π_k) / π_k\nmean of seeds × 10 repeats")
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=7.5, bbox_to_anchor=(0.5, -0.08))
        fig.suptitle(f"{ds}: relative error of final-sample mode weights", color=INK)
        save(fig, f"mode_weights_{ds}")

    # F5b: E4 on gmm9w, heatmap of relative mode-weight error at NFE 10 and 100
    w = GMM_SPECS["gmm9w"][2]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, nfe in zip(axs, [10, 100]):
        M = np.array([(np.array(fr[(g, nfe)]).mean(0) - w) / w for g in E4_9W])
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6, color=INK)
        ax.set_xticks(range(9))
        ax.set_xticklabels([f"{k}\n{w[k]:.2f}" for k in range(9)], fontsize=7)
        ax.set_yticks(range(len(E4_9W)))
        ax.set_yticklabels([g.split("_", 1)[1] for g in E4_9W], fontsize=7)
        ax.set_title(f"NFE {nfe}", color=INK)
        ax.grid(False)
    fig.colorbar(im, ax=axs, shrink=0.8, label="(p̂_k − π_k) / π_k")
    fig.suptitle("gmm9w (seed 0; A, B, C: seeds 0, 1): relative mode-weight error", color=INK)
    save(fig, "mode_weights_gmm9w_e4")

    # F6: conditional mode weights at high t: non-dominant mass, model vs exact
    cr = [r for r in rd("condw.csv") if int(r["seed"]) in (0, 1)]
    nd = defaultdict(list)
    ex = {}
    for r in cr:
        nd[(r["group"], float(r["t"]))].append(float(r["nondominant_mass_model"]))
        ex[(r["dataset"], float(r["t"]))] = float(r["nondominant_mass_exact"])
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    T = [0.99, 0.88, 0.7, 0.45]
    for ax, (ds, gs) in zip(axs, [("gmm", ["gmm_C", "gmm_C_b0.5", "gmm_C_b1", "gmm_C_b1.5", "gmm_C_b1.99", "gmm_C_b2"]),
                                  ("gmm2w", ["gmm2w_C"]),
                                  ("gmm9w", [g for g in ["gmm9w_C", "gmm9w_C_b1.5"] if have(g)] + ["gmm9w_C_L1b1", "gmm9w_C_L1b0.5", "gmm9w_C_pH0.1", "gmm9w_C_imq", "gmm9w_C_rbf", "gmm9w_C_lap"])]):  # <= 8 series (pH1 ~ pH0.1, in tables.md)
        ax.plot(T, [ex[(ds, t)] for t in T], color=INK, lw=2.5, label="exact responsibilities")
        for g, col in zip(gs, CAT):
            ax.plot(T, [np.mean(nd[(g, t)]) for t in T], "-o", ms=4, color=col, lw=1.4, label=g.split("_", 1)[1])
        ax.set_title(ds, color=INK)
        ax.set_xlabel("t")
        ax.legend(fontsize=6.5)
    axs[0].set_ylabel("mass outside the dominant mode\n(mean over 256 x_t)")
    fig.suptitle("Conditional mode weights at high noise: x̂ draws vs exact posterior", color=INK)
    save(fig, "condw_nondominant")

    # F7: denoiser location (E1a)
    dr = [r for r in rd("denoiser.csv") if int(r["seed"]) in (0, 1)]
    dd = defaultdict(list)
    for r in dr:
        dd[(r["group"], float(r["t"]))].append(r)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, t in zip(axs, [0.99, 0.88, 0.45]):
        bs = [b for _, b in LAM0]
        for key, col, lab in [("d_mean", CAT[0], "to exact posterior mean"), ("d_gmed", CAT[1], "to geometric median"),
                              ("d_nearest_mode", CAT[2], "to nearest component mean")]:
            ax.plot(bs, [np.mean([float(r[key]) for r in dd[(g, t)]]) for g, _ in LAM0], "-o", ms=4, color=col, label=lab)
        a = dd[("gmm_A", t)]
        ax.axhline(np.mean([float(r["ref_d_mean_gmed"]) for r in a]), color=GRAY, ls=":", lw=1, label="exact mean ↔ geometric median")
        ax.set_xscale("log")
        ax.set_xticks([0.1, 0.5, 1, 2, 3.5]); ax.set_xticklabels(["0.1", "0.5", "1", "2", "3.5"])
        ax.set_xlabel("β (λ = 0)")
        ax.set_title(f"t = {t}", color=INK)
    axs[0].set_ylabel("mean distance of x̂ (256 x_t, seeds 0, 1)")
    axs[0].legend(fontsize=7)
    fig.suptitle("Paper GMM, B family: where does the deterministic denoiser sit?", color=INK)
    save(fig, "denoiser_location")

    # F8: E4 metric table as a figure
    rows = []
    for ds, gs in [("gmm", ["gmm_A"] + E4_GMM), ("gmm9w", E4_9W)]:
        for g in gs:
            cells = [ds, g.split("_", 1)[1]]
            for nfe in (5, 10, 100):
                v = pm[(g, nfe)]
                cells.append(f"{np.mean(v):.1e}" + (f" [{min(v):.1e}, {max(v):.1e}]" if len(v) > 1 else ""))
            rows.append(cells)
    fig, ax = plt.subplots(figsize=(10, 0.24 * len(rows) + 0.8))
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=["dataset", "run", "MMD² NFE 5", "MMD² NFE 10", "MMD² NFE 100"], loc="center", cellLoc="left")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7)
    tb.scale(1, 1.15)
    ax.set_title("E4: MMD²_u per distance ρ (seed mean [seed min, max]; gmm9w ρ runs: seed 0 only)", color=INK, fontsize=9)
    save(fig, "e4_table")


if __name__ == "__main__":
    main()
