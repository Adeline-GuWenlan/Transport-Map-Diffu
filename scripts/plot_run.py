"""Figures for a run directory: loss curve, grad norm, held-out energy distance (with baselines), sample montage.

    python scripts/plot_run.py runs/<run> [--baselines runs/<run>/eval_final.json]

Writes <run>/fig_curves.png and <run>/fig_samples_over_steps.png.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# dataviz reference palette (light mode, fixed order), as docs/stage1_checks/es_core.py
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#a3a29c"


def read_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def smooth(v, k):
    if len(v) < k or k <= 1:
        return np.asarray(v)
    c = np.cumsum(np.insert(np.asarray(v, float), 0, 0))
    out = (c[k:] - c[:-k]) / k
    return np.concatenate([np.full(k - 1, np.nan), out])


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("training step", color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--baselines", default=None)
    a = ap.parse_args()
    m = read_jsonl(os.path.join(a.run, "metrics.jsonl"))
    e = read_jsonl(os.path.join(a.run, "eval.jsonl"))
    bl_path = a.baselines or os.path.join(a.run, "eval_final.json")
    bl = json.load(open(bl_path))["baselines"] if os.path.exists(bl_path) else None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    st = np.array([r["step"] for r in m])
    ax = axes[0]
    loss = [r["loss"] for r in m]
    ax.plot(st, loss, color=BLUE, linewidth=0.8, alpha=0.35)
    ax.plot(st, smooth(loss, 10), color=BLUE, linewidth=2)
    fin = os.path.join(a.run, "eval_final.json")
    if os.path.exists(fin):
        # optimum of the expected loss: 1/2 sum_c pi_c E||Y - Y'|| (classes equally weighted by the sampler)
        wy = json.load(open(fin))["ema"]["energy_distance"]["within_y"]
        floor = 0.5 * float(np.mean(list(wy.values())))
        ax.axhline(floor, color=MUTED, linestyle="--", linewidth=1)
        ax.text(st[-1], floor, f"optimum ½E‖Y−Y′‖ = {floor:.2f} ", color=INK2, fontsize=8, va="top", ha="right")
        ax.set_ylim(floor - 0.3, floor + 3.0)
        ax.text(st[-1], floor + 2.9, f"step 0 loss {loss[0]:.1f} (off scale) ", color=INK2, fontsize=8, va="top", ha="right")
    style(ax, "Training loss (per 50 steps; bold: 500-step mean)", "loss")
    ax = axes[1]
    ax.plot(st, [r["grad_norm"] for r in m], color=BLUE, linewidth=1.5)
    ax.axhline(2.0, color=MUTED, linestyle="--", linewidth=1)
    ax.text(st[-1] if len(st) else 0, 2.0, " clip 2.0", color=INK2, fontsize=8, va="bottom", ha="right")
    ax.set_yscale("log")
    style(ax, "Gradient norm before clipping", "‖g‖ (log)")
    ax = axes[2]
    es = [r["step"] for r in e]
    ax.plot(es, [r["raw_ed_mean"] for r in e], color=BLUE, linewidth=2, marker="o", markersize=4, label="raw weights")
    ax.plot(es, [r["ema_ed_mean"] for r in e], color=ORANGE, linewidth=2, marker="o", markersize=4, label="EMA weights")
    if e:
        for key, col in (("raw_ed_mean", BLUE), ("ema_ed_mean", ORANGE)):
            ax.annotate(f"{e[-1][key]:.3f}", (es[-1], e[-1][key]), xytext=(4, 0), textcoords="offset points",
                        color=INK, fontsize=8, va="center")
    if bl:
        names = {"noise": "N(0, I) noise", "zero": "zero image (step 0)", "class_mean": "class-mean point mass",
                 "other_class": "wrong-class train images", "train": "train images (floor)"}
        for k, lab in names.items():
            if k in bl:
                v = max(bl[k]["mean"], 1e-4)
                ax.axhline(v, color=MUTED, linestyle=":", linewidth=1)
                ax.text(1.0, v, f"{lab}: {bl[k]['mean']:.3g} ", color=INK2, fontsize=7.5, va="bottom", ha="right",
                        transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.08))
    n = e[-1].get("n_per_class") if e else "?"
    style(ax, f"Held-out energy distance D̂₁ ({n}/class, class mean)", "D̂₁ (log)")
    p = os.path.join(a.run, "fig_curves.png")
    fig.savefig(p, dpi=130)
    print("wrote", p)

    # sample montage: EMA grids at up to 6 steps (rows = classes, 8 fixed z per class)
    for w in ("ema", "raw"):
        files = sorted(glob.glob(os.path.join(a.run, "samples", f"step*_{w}.png")))
        if not files:
            continue
        idx = np.unique(np.linspace(0, len(files) - 1, min(6, len(files))).round().astype(int))
        files = [files[i] for i in idx]
        fig, axes = plt.subplots(1, len(files), figsize=(3.2 * len(files), 4.0), constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, f in zip(axes, files):
            img = plt.imread(f)
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None, interpolation="nearest")
            ax.set_title(f"step {int(os.path.basename(f)[4:11])}", fontsize=9, color=INK)
            ax.axis("off")
        fig.suptitle(f"Samples, {w} weights, fixed z (rows = classes, 8 samples each)", fontsize=10, color=INK)
        p = os.path.join(a.run, f"fig_samples_over_steps_{w}.png")
        fig.savefig(p, dpi=130)
        print("wrote", p)


if __name__ == "__main__":
    main()
