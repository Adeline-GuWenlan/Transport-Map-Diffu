"""ddm2dx X1: ground-truth plots of the E2 datasets gmm2w and gmm9w (train-set scatter + component means/weights).
Usage (from ddm2d/): PYTHONPATH=src python scripts/ext_plot_datasets.py   -> figures/ext/datasets_gt.{png,pdf}"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddm2d import data as D

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
for ax, name in zip(axes, ["gmm", "gmm2w", "gmm9w"]):
    x = D.train_set(name)
    means, stds, w = D.GMM_SPECS[name]
    ax.scatter(x[:20000, 0], x[:20000, 1], s=0.5, alpha=0.3, color="#4c72b0", rasterized=True)
    for k in range(len(w)):
        ax.add_patch(plt.Circle(means[k], 2 * stds[k], fill=False, color="#c44e52", lw=0.8))
        ax.annotate(f"k={k}\nπ={w[k]:.2f}\ns={stds[k]:.1f}", means[k] + np.array([0.0, 2 * stds[k] + 0.15]), ha="center",
                    va="bottom", fontsize=7)
    ax.set_title(f"{name} (train set, first 20k of {len(x)}; circles = 2 s_k)", fontsize=9)
    ax.set_aspect("equal")
    lim = 6.5 if name == "gmm9w" else 5.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim) if name == "gmm9w" else ax.set_ylim(-1, 6.5)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(f"figures/ext/datasets_gt.{ext}", dpi=150)
print({n: D.fingerprint(D.train_set(n)) for n in ["gmm2w", "gmm9w"]})
