"""Figures and tables for the S3 GMM evaluation, from the raw files of scripts/eval_gmm.py (runs/paper/eval_gmm/) and the run logs.
Colours follow the paper's Fig. 3/4 (blue = C beta 0.1 lambda 1, orange = B beta 2 lambda 0, green = A standard diffusion / exact posterior),
so our panels can be compared with the paper's side by side. Paper values (docs/paper_values/fig3_paper_values.json, read off the paper's vector
figures) are always drawn in a separate panel or with an explicit "paper" label, never as ours.

Usage: python scripts/plot_gmm.py loss|samples|fig3left|fig3right|fig4|snaps|all
"""
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
EV = PAPER / "eval_gmm"
FIG = ROOT / "figures" / "gmm"
COL = {"A": "#2ca02c", "B": "#ff7f0e", "C": "#1f77b4", "exact": "#2ca02c"}
LAB = {"A": "A: standard diffusion (Eq. 6)", "B": r"B: $\hat x_\theta(t,X_t,\xi)$, $\beta=2,\lambda=0$",
       "C": r"C: $\hat x_\theta(t,X_t,\xi)$, $\beta=0.1,\lambda=1$"}
PAPER_KEY = {"A": "Diffusion model (arm A)", "B": "B (beta=2, lambda=0)", "C": "C (beta=0.1, lambda=1)"}
SEEDS = [0, 1, 2]


def save(fig, name):
    (FIG / name).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=130, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def load_log(run):
    return [json.loads(l) for l in (PAPER / run / "log.jsonl").read_text().splitlines() if l.strip()]


def smooth(y, k=20):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    c = np.convolve(y, np.ones(k) / k, mode="valid")
    return np.concatenate([np.full(k - 1, np.nan), c])


def loss():
    """Per-run training-loss curves (all 18 paper runs) and one overview per dataset."""
    summary = []
    for ds in ["gmm", "cb"]:
        fig_o, axes_o = plt.subplots(1, 3, figsize=(13, 3.4), squeeze=False)
        for j, arm in enumerate("ABC"):
            for s in SEEDS:
                run = f"{ds}_{arm}_s{s}"
                log = load_log(run)
                st = np.array([q["step"] for q in log])
                lo = np.array([q["loss"] for q in log])
                # per-run figure: loss (and for B/C the confinement and interaction terms)
                ncol = 2 if arm != "A" else 1
                fig, ax = plt.subplots(1, ncol, figsize=(4.6 * ncol, 3.2), squeeze=False)
                ax[0, 0].plot(st, lo, lw=0.5, alpha=0.35, color=COL[arm], label="mean over 100 steps")
                ax[0, 0].plot(st, smooth(lo), lw=1.6, color=COL[arm], label="moving average (2k steps)")
                ax[0, 0].set_xlabel("step")
                ax[0, 0].set_ylabel("weighted training loss")
                ax[0, 0].set_title(f"{run}: loss", fontsize=9)
                ax[0, 0].legend(fontsize=7)
                ax[0, 0].grid(alpha=0.25)
                if arm != "A":
                    conf = np.array([q["conf"] for q in log])
                    inter = np.array([q["inter"] for q in log])
                    ax[0, 1].plot(st, smooth(conf), lw=1.4, color="#444444", label="confinement mean ||x0 - x_hat||^beta")
                    ax[0, 1].plot(st, smooth(inter), lw=1.4, ls="--", color="#999999", label="interaction mean ||x_hat - x_hat'||^beta")
                    ax[0, 1].set_xlabel("step")
                    ax[0, 1].set_title(f"{run}: terms (unweighted, 2k moving avg)", fontsize=9)
                    ax[0, 1].legend(fontsize=7)
                    ax[0, 1].grid(alpha=0.25)
                fig.tight_layout()
                save(fig, f"loss/loss_{run}")
                axes_o[0, j].plot(st, smooth(lo), lw=1.2, color=COL[arm], alpha=[1, 0.7, 0.45][s], label=f"seed {s}")
                last = lo[st > 90_000]
                summary.append({"run": run, "loss_mean_90k_100k": float(last.mean()), "loss_first_logged": float(lo[0]),
                                "any_nonfinite": not all(q["finite"] for q in log), "gnorm_max_overall": float(max(q["gnorm_max"] for q in log))})
            axes_o[0, j].set_title(f"{ds} arm {arm}", fontsize=10)
            axes_o[0, j].set_xlabel("step")
            axes_o[0, j].legend(fontsize=7)
            axes_o[0, j].grid(alpha=0.25)
            vals = np.concatenate([smooth(np.array([q["loss"] for q in load_log(f"{ds}_{arm}_s{s}")]))[40:] for s in SEEDS])
            axes_o[0, j].set_ylim(np.nanpercentile(vals, 0.5) * 0.98, np.nanpercentile(vals, 99.5) * 1.02)
        axes_o[0, 0].set_ylabel("weighted training loss (2k-step moving avg)")
        fig_o.suptitle(f"{ds}: training loss of the 9 paper runs (y-range clipped to the 0.5-99.5 percentile after 4k steps)", fontsize=10)
        fig_o.tight_layout()
        save(fig_o, f"loss_overview_{ds}")
    (EV / "loss_summary.json").write_text(json.dumps(summary, indent=0) + "\n")


def samples():
    from ddm2d import data as D
    ref = D.eval_set("gmm", 4096, 0)
    rows = {r["run"] + f"_{r['nfe']}": r for run in [f"gmm_{a}_s{s}" for a in "ABC" for s in SEEDS]
            for r in json.loads((EV / f"mmd_{run}.json").read_text()) if r["repeat"] == 0}
    show = [5, 10, 20, 100]
    for s in SEEDS:
        fig, axes = plt.subplots(3, len(show) + 1, figsize=(2.9 * (len(show) + 1), 2.2 * 3), squeeze=False)
        for i, arm in enumerate("ABC"):
            run = f"gmm_{arm}_s{s}"
            smp = np.load(EV / f"samples_{run}.npz")
            axes[i, 0].scatter(ref[:, 0], ref[:, 1], s=0.6, alpha=0.35, c="#555555", rasterized=True)
            axes[i, 0].set_title("reference p_0 (eval set 0)", fontsize=8)
            axes[i, 0].set_ylabel(LAB[arm], fontsize=7)
            for j, nfe in enumerate(show):
                x = smp[f"nfe{nfe}"]
                axes[i, j + 1].scatter(x[:, 0], x[:, 1], s=0.6, alpha=0.35, c=COL[arm], rasterized=True)
                axes[i, j + 1].set_title(f"{run}  NFE {nfe}\nMMD²_u = {rows[f'{run}_{nfe}']['mmd2_u']:.1e}", fontsize=7)
            for ax in axes[i]:
                ax.set_xlim(-6, 6)
                ax.set_ylim(0, 6)
                ax.set_aspect("equal")
                ax.tick_params(labelsize=6)
        fig.suptitle(f"GMM: 4096 samples per panel, EMA weights, churn 1, sampler seeds shared by all arms (training seed {s})", fontsize=9)
        fig.tight_layout()
        save(fig, f"samples_seed{s}")


def _mmd_table():
    import csv
    recs = []
    for arm in "ABC":
        for s in SEEDS:
            recs += json.loads((EV / f"mmd_gmm_{arm}_s{s}.json").read_text())
    EV.mkdir(exist_ok=True)
    keys = ["run", "nfe", "repeat", "mmd2_u", "mmd2_b", "d_rho_u", "d_rho_b", "kxx_u", "kyy_u", "kxy", "n", "m", "sigma", "churn", "n_finite"]
    with open(EV / "fig3_left_mmd2_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
    return recs


def fig3left():
    recs = _mmd_table()
    paper = json.loads((ROOT / "docs" / "paper_values" / "fig3_paper_values.json").read_text())["fig3_left_mmd2"]["series"]
    null = json.loads((EV / "mmd_null.json").read_text())
    nfes = sorted({r["nfe"] for r in recs})
    agg = {}
    for arm in "ABC":
        for s in SEEDS:
            for nfe in nfes:
                v = [r["mmd2_u"] for r in recs if r["run"] == f"gmm_{arm}_s{s}" and r["nfe"] == nfe]
                vb = [r["mmd2_b"] for r in recs if r["run"] == f"gmm_{arm}_s{s}" and r["nfe"] == nfe]
                agg[(arm, s, nfe)] = (np.mean(v), np.std(v, ddof=1), np.mean(vb), len(v))
    # summary table (per seed = mean over repeats; across seeds = mean and min/max of the per-seed means)
    out = []
    for arm in "ABC":
        for nfe in nfes:
            per_seed = [agg[(arm, s, nfe)][0] for s in SEEDS]
            per_seed_b = [agg[(arm, s, nfe)][2] for s in SEEDS]
            pv = dict((int(x), y) for x, y in paper[PAPER_KEY[arm]])
            out.append({"arm": arm, "nfe": nfe, **{f"mmd2_u_seed{s}": per_seed[s] for s in SEEDS},
                        **{f"mc_sd_seed{s}": agg[(arm, s, nfe)][1] for s in SEEDS},
                        "mmd2_u_mean_over_seeds": float(np.mean(per_seed)), "mmd2_u_seed_sd": float(np.std(per_seed, ddof=1)),
                        "mmd2_u_seed_min": float(min(per_seed)), "mmd2_u_seed_max": float(max(per_seed)),
                        "mmd2_b_mean_over_seeds": float(np.mean(per_seed_b)),
                        "paper_read_off_fig3": pv.get(nfe), "repeats_per_seed": agg[(arm, 0, nfe)][3]})
    (EV / "fig3_left_summary.json").write_text(json.dumps({"rows": out, "null_p0_vs_p0": {
        "mmd2_u": [q["mmd2_u"] for q in null], "mmd2_b": [q["mmd2_b"] for q in null]},
        "note": "mmd2_u_seed<s> = mean over repeats (sampler seed 1e6+1000r+nfe, reference eval set r) of the unbiased MMD^2 with RBF sigma=1, "
                "n=m=4096; mc_sd = sd over repeats; paper_read_off_fig3 = paper value read off the vector figure (not ours)"}, indent=1) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for arm in "CBA":
        m = np.array([np.mean([agg[(arm, s, n)][0] for s in SEEDS]) for n in nfes])
        lo = np.array([min(agg[(arm, s, n)][0] for s in SEEDS) for n in nfes])
        hi = np.array([max(agg[(arm, s, n)][0] for s in SEEDS) for n in nfes])
        axes[0].plot(nfes, m, "-o", color=COL[arm], ms=5, lw=1.8, label=LAB[arm] + " (mean of 3 seeds)")
        axes[0].fill_between(nfes, lo, hi, color=COL[arm], alpha=0.18, lw=0)
        axes[1].plot(nfes, m, "-o", color=COL[arm], ms=5, lw=1.8, label=LAB[arm])
        axes[1].fill_between(nfes, lo, hi, color=COL[arm], alpha=0.18, lw=0)
        px, py = zip(*paper[PAPER_KEY[arm]])
        axes[2].plot(px, py, "-o", color=COL[arm], ms=5, lw=1.8, label="paper: " + LAB[arm])
    orc_path = EV / "oracle_mmd.json"
    if orc_path.exists():
        orc = json.loads(orc_path.read_text())
        for kind, ls, lab in [("oracle_mean", "--", "oracle: exact E[X0|x_t] in the same sampler (ideal A/B)"),
                              ("oracle_draw", ":", "oracle: exact draw from p(X0|x_t) (ideal C)")]:
            y = [np.mean([r["mmd2_u"] for r in orc if r["who"] == kind and r["churn"] == 1.0 and r["nfe"] == n]) for n in nfes]
            for ax in axes[:2]:
                ax.plot(nfes, y, ls, color="#333333", lw=1.2, label=lab)
    nu = [q["mmd2_u"] for q in null]
    for ax in axes[:2]:
        ax.axhspan(min(nu), max(nu), color="#888888", alpha=0.25, lw=0, label=f"p0 vs p0 null, {len(nu)} draws (MMD²_u)")
    axes[0].set_title("Ours: MMD²_u vs sampling steps (band = min-max over 3 seeds)", fontsize=10)
    axes[0].set_ylim(-0.0008, 0.017)
    axes[1].set_title("Ours, zoom (y in [-5e-4, 3e-3])", fontsize=10)
    axes[1].set_ylim(-0.0005, 0.003)
    axes[2].set_title("Paper Fig. 3 left (values read off the vector figure)", fontsize=10)
    axes[2].set_ylim(-0.0008, 0.017)
    for ax in axes:
        ax.set_xticks(nfes)
        ax.set_xlabel("Num sampling steps (= NFE)")
        ax.axhline(0, color="#bbbbbb", lw=0.8)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Squared MMD [generated | target], RBF σ=1, n=m=4096")
    axes[0].legend(fontsize=6.5)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig3_left")
    # per-seed figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for k, arm in enumerate("CBA"):
        for s in SEEDS:
            m = [agg[(arm, s, n)][0] for n in nfes]
            e = [agg[(arm, s, n)][1] for n in nfes]
            axes[k].errorbar(nfes, m, yerr=e, fmt="-o", ms=4, lw=1.3, capsize=2, color=COL[arm], alpha=[1, 0.7, 0.45][s], label=f"seed {s}")
        axes[k].set_title(LAB[arm] + " — per seed (error bar = sd over repeats)", fontsize=9)
        axes[k].set_xticks(nfes)
        axes[k].set_xlabel("NFE")
        axes[k].grid(alpha=0.25)
        axes[k].legend(fontsize=7)
        axes[k].set_yscale("symlog", linthresh=1e-4)
    axes[0].set_ylabel("MMD²_u (symlog)")
    fig.tight_layout()
    save(fig, "fig3_left_per_seed")


def fig3right():
    res = json.loads((EV / "posterior_std.json").read_text())
    paper = json.loads((ROOT / "docs" / "paper_values" / "fig3_paper_values.json").read_text())["fig3_right_posterior_std"]["series"]
    cur = res["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    ax = axes[0]
    ft = [p["t"] for p in cur["exact"]["fine"]]
    ax.plot(ft, [p["std_mean"] for p in cur["exact"]["fine"]], color=COL["exact"], lw=1.8, label="Posterior p(X_0 | X_t) (8 exact draws)")
    ax.plot(ft, [p["exact_std_mean"] for p in cur["exact"]["fine"]], color=COL["exact"], lw=1.0, ls=":", label="Posterior, exact std (no sampling)")
    for arm in "CB":
        m = np.mean([[p["std_mean"] for p in cur[f"gmm_{arm}_s{s}"]["fine"]] for s in SEEDS], 0)
        lo = np.min([[p["std_mean"] for p in cur[f"gmm_{arm}_s{s}"]["fine"]] for s in SEEDS], 0)
        hi = np.max([[p["std_mean"] for p in cur[f"gmm_{arm}_s{s}"]["fine"]] for s in SEEDS], 0)
        ax.plot(ft, m, color=COL[arm], lw=1.8, label=LAB[arm] + " (mean of 3 seeds)")
        ax.fill_between(ft, lo, hi, color=COL[arm], alpha=0.2, lw=0)
    ax.set_title("Ours: average (on x_t) std of x̂ over ξ (band = min-max over seeds)\nArm A is deterministic: std = 0 exactly, not drawn", fontsize=9)
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("mean per-coordinate std over 8 draws")
    for arm, key in [("C", "C (beta=0.1, lambda=1)"), ("B", "B (beta=2, lambda=0)")]:
        x, y = zip(*sorted(paper[key]))
        axes[1].plot(x, y, color=COL[arm], lw=1.8, label="paper: " + LAB[arm])
    x, y = zip(*sorted(paper["exact posterior"]))
    axes[1].plot(x, y, color=COL["exact"], lw=1.8, label="paper: Posterior p(X_0 | X_t)")
    axes[1].set_title("Paper Fig. 3 right (values read off the vector figure; t grid = linspace(0.01, 0.99, 10))", fontsize=9)
    axes[1].set_xlabel("Timestep t")
    for a in axes:
        a.grid(alpha=0.25)
        a.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "fig3_right")
    # table on the paper grid
    rows = []
    pt = [p["t"] for p in cur["exact"]["paper"]]
    for i, t in enumerate(pt):
        r = {"t": round(t, 4), "exact_8draws": cur["exact"]["paper"][i]["std_mean"], "exact_std": cur["exact"]["paper"][i]["exact_std_mean"]}
        for arm in "ABC":
            for s in SEEDS:
                r[f"{arm}_s{s}"] = cur[f"gmm_{arm}_s{s}"]["paper"][i]["std_mean"]
            v = [r[f"{arm}_s{s}"] for s in SEEDS]
            r[f"{arm}_mean"], r[f"{arm}_seed_sd"] = float(np.mean(v)), float(np.std(v, ddof=1))
        for key, nm in [("C (beta=0.1, lambda=1)", "paper_C"), ("B (beta=2, lambda=0)", "paper_B"), ("exact posterior", "paper_exact")]:
            r[nm] = dict((round(a, 2), b) for a, b in paper[key]).get(round(t, 2))
        r["ratio_C_over_exact8_ours"] = r["C_mean"] / r["exact_8draws"]
        r["ratio_C_over_exact_paper"] = r["paper_C"] / r["paper_exact"] if r["paper_C"] is not None else None
        rows.append(r)
    (EV / "fig3_right_summary.json").write_text(json.dumps(rows, indent=1) + "\n")


def fig4(run_suffix="s0"):
    d = np.load(EV / "fig4_draws.npz")
    summ = json.loads((EV / "fig4_summary.json").read_text())
    for s in SEEDS:
        fig, axes = plt.subplots(2, 5, figsize=(17, 7.4))
        for i in range(2):
            for j in range(5):
                ax = axes[i, j]
                ax.scatter(*d["posterior"][i, j].T, s=9, color="#00bfff", alpha=0.2, lw=0, rasterized=True)
                ax.scatter(*d[f"gmm_C_s{s}"][i, j].T, s=9, color="green", alpha=0.2, lw=0, rasterized=True)
                ax.scatter(*d[f"gmm_B_s{s}"][i, j].T, s=9, color="red", alpha=0.9, lw=0, rasterized=True)
                ax.scatter(*d["x0"][i], marker="x", s=160, color="orange", lw=2, zorder=5)
                ax.scatter(*d["xt"][i, j], marker="x", s=160, color="black", lw=2, zorder=5)
                ax.set_xlim(-6, 6)
                ax.set_ylim(min(-1.5, float(d["xt"][..., 1].min()) - 0.6), 6.5)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"Noise level t = {d['t'][j]:.2f}", fontsize=14)
                if j == 0:
                    ax.set_ylabel(f"x0 = ({d['x0'][i][0]:.0f}, {d['x0'][i][1]:.0f})", fontsize=12)
                e = [q for q in summ if q["row"] == i and abs(q["t"] - float(d["t"][j])) < 1e-6 and q["who"] == f"gmm_C_s{s}"][0]
                ax.text(0.02, 0.97, f"C: MMD²_u vs post. {e['mmd2_u_vs_posterior']:.1e}", transform=ax.transAxes, fontsize=7, va="top")
        h = [plt.Line2D([], [], ls="", marker="o", color="#00bfff", alpha=0.5, label=r"Posterior $p_{0|t}(\cdot|x_t)$ (exact, 4096)"),
             plt.Line2D([], [], ls="", marker="o", color="green", alpha=0.4, label=r"$\hat x_\theta(t,x_t,\cdot)$, $\beta=0.1,\lambda=1$ (C, 4096)"),
             plt.Line2D([], [], ls="", marker="o", color="red", label=r"$\hat x_\theta(t,x_t,\cdot)$, $\beta=2,\lambda=0$ (B, 4096)"),
             plt.Line2D([], [], ls="", marker="x", color="orange", ms=10, mew=2, label=r"$x_0$"),
             plt.Line2D([], [], ls="", marker="x", color="black", ms=10, mew=2, label=r"$x_t|x_0$")]
        fig.legend(handles=h, loc="lower center", ncol=5, fontsize=12, frameon=True)
        fig.suptitle(f"Fig. 4 reproduction, training seed {s} (EMA weights). Same x_t for every arm and the same axes in every panel; "
                     "rows follow the task spec (top: x0 = (3, 3)).", fontsize=10)
        fig.tight_layout(rect=(0, 0.07, 1, 0.97))
        save(fig, f"fig4_seed{s}")


def snaps():
    import csv
    recs = []
    for arm in "ABC":
        for s in SEEDS:
            recs += json.loads((EV / f"snaps_gmm_{arm}_s{s}.json").read_text())
    with open(EV / "snapshots_mmd2_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "step", "nfe", "repeat", "mmd2_u", "mmd2_b"], extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
    nfes = sorted({r["nfe"] for r in recs})
    steps = sorted({r["step"] for r in recs})
    fig, axes = plt.subplots(1, len(nfes), figsize=(3.4 * len(nfes), 3.4), sharey=True)
    for k, nfe in enumerate(nfes):
        for arm in "ABC":
            for s in SEEDS:
                y = [np.mean([r["mmd2_u"] for r in recs if r["run"] == f"gmm_{arm}_s{s}" and r["step"] == st and r["nfe"] == nfe]) for st in steps]
                axes[k].plot(np.array(steps) / 1000, y, "-o", ms=3, lw=1.1, color=COL[arm], alpha=[1, 0.7, 0.45][s],
                             label=f"{arm} s{s}" if k == 0 else None)
        axes[k].set_title(f"NFE {nfe}", fontsize=10)
        axes[k].set_xlabel("training step (k), EMA snapshot")
        axes[k].set_yscale("symlog", linthresh=1e-4)
        axes[k].grid(alpha=0.25)
    axes[0].set_ylabel("MMD²_u (mean of 2 repeats, symlog)")
    axes[0].legend(fontsize=6, ncol=3)
    fig.suptitle("GMM: MMD² of EMA snapshots during training", fontsize=10)
    fig.tight_layout()
    save(fig, "snapshots_mmd2")


if __name__ == "__main__":
    what = sys.argv[1]
    fns = {"loss": loss, "samples": samples, "fig3left": fig3left, "fig3right": fig3right, "fig4": fig4, "snaps": snaps}
    for k in (fns if what == "all" else [what]):
        fns[k]()
