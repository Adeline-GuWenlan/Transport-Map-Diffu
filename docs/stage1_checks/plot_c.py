"""Figures for check (c), from results/check_c.json and results/check_c_samples_seed0.npz.
Usage: python plot_c.py   (also called at the end of check_c_toy2d.py)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from es_core import BLUE, ORANGE, AQUA, YELLOW, MAGENTA, RED, VIOLET, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "results")
LABELS = {
    "es_beta1": r"corrected, $\beta=1$",
    "es_beta1.5": r"corrected, $\beta=1.5$",
    "literal_rawdiff": r"literal: $-\frac{1}{2}E[F(x)-F(x')]$",
    "data_pairs": r"$x,x'$ from data: $-\frac{1}{2}E\|y-y'\|$",
    "beta2_squared": r"squared norms ($\beta=2$)",
    "coef1_no_half": r"coefficient 1 instead of $\frac{1}{2}$",
    "mixed_norm_sq": r"$\|\cdot\|$ then $-\frac{1}{2}\|\cdot\|^2$",
}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    J = json.load(open(os.path.join(RES, "check_c.json")))
    S = np.load(os.path.join(RES, "check_c_samples_seed0.npz"))
    R, steps = J["results"], J["config"]["steps"]
    tgt = S["target"]
    plt.rcParams.update({"font.size": 8})

    # ---------------- samples
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.4), constrained_layout=True)
    panels = [("target", r"target $Q$ (8 Gaussians)", tgt)] + [(k, LABELS[k], S[k]) for k in LABELS]
    for ax, (k, lab, xs) in zip(axes.ravel(), panels):
        fin = np.isfinite(xs).all(1)
        if k == "target":
            ax.scatter(xs[:2000, 0], xs[:2000, 1], s=1.5, color="#52514e", rasterized=True, lw=0)
            sub = ""
        else:
            col = BLUE if k.startswith("es_") else ORANGE
            ax.scatter(tgt[:1500, 0], tgt[:1500, 1], s=1.0, color="#d9d8d3", rasterized=True, lw=0)
            ax.scatter(xs[fin][:2000, 0], xs[fin][:2000, 1], s=2.0, color=col, rasterized=True, lw=0)
            st = R[k]["0"]
            sub = f"\n$D_1$ = {R[k]['summary']['D1_mean']:.3g}"
            std = max(st["sample_std"])
            if st["frac_outside_4"] > 0.01:
                ax.text(0, 0, f"{100 * st['frac_outside_4']:.0f}% of samples\noutside window\n(std {std:.1e})",
                        ha="center", va="center", fontsize=7, color="#52514e")
            elif std < 0.1:
                ax.annotate(f"all samples here\n(std {std:.3f})", xy=st["sample_mean"], xytext=(0, -2.4),
                            ha="center", fontsize=7, color="#52514e",
                            arrowprops=dict(arrowstyle="->", color="#52514e", lw=0.7))
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(lab + sub, fontsize=7.5)
        for s in ax.spines.values():
            s.set_color("#c3c2b7")
    fig.savefig(os.path.join(FIG, "check_c_samples.pdf"), dpi=200)

    # ---------------- curves
    fig, ax = plt.subplots(figsize=(7.0, 3.0), constrained_layout=True)
    cols = dict(zip(LABELS, [BLUE, AQUA, ORANGE, YELLOW, MAGENTA, RED, VIOLET]))
    styles = dict(zip(LABELS, ["-", "-", "-", "--", ":", "-", "--"]))
    top = 1e2
    for k in LABELS:
        c = np.array(R[k]["0"]["curve"], dtype=float)
        y = np.clip(c[:, 1], 1e-3, None)
        ax.semilogy(c[:, 0], np.minimum(y, top), styles[k], color=cols[k], lw=2, label=LABELS[k])
    ax.axhline(abs(J["baselines"]["target_vs_target"]) + 1e-3, color=MUTED, ls=":", lw=1)
    ax.text(steps, (abs(J["baselines"]["target_vs_target"]) + 1e-3) * 1.2, "target vs target (sampling noise)",
            ha="right", va="bottom", fontsize=7, color="#52514e")
    ax.axhline(J["baselines"]["point_mass_origin_vs_target"], color=MUTED, ls="--", lw=1)
    ax.text(steps, J["baselines"]["point_mass_origin_vs_target"] * 1.25, "point mass at origin", ha="right",
            va="bottom", fontsize=7, color="#52514e")
    ax.text(300, top * 0.6, r"diverging runs leave the plot (they reach $10^{8}$ to $10^{9}$)", fontsize=7, color="#52514e",
            va="top")
    ax.set_ylim(5e-4, top * 1.2)
    ax.set_xlabel("training step")
    ax.set_ylabel(r"held-out $\hat D_1$ (1500 vs 1500)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(os.path.join(FIG, "check_c_curves.pdf"))


if __name__ == "__main__":
    main()
