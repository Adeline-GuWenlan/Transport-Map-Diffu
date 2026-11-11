"""Experiments A / B / C summary (docs/experiments_AB.md): table, metric-vs-step figure, sample montage, and the
Experiment C rule applied to the numbers.

    python scripts/summarize_ab.py [--runs runs/expA_s0 runs/expB_s0 ...] [--weights ema|raw] [--out runs/expAB]

Reads <run>/eval/step*.json (written by scripts/exp_gpu.slurm after training). A run is "A" or "B" by the prefix of
its directory name (expA_*, expB_*); its seed is read from <run>/config.yaml. Writes <out>_summary.md,
<out>_metrics_<weights>.png, <out>_samples_<weights>.png.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# dataviz reference palette, light mode: categorical slots 1-2 in fixed order (A = slot 1, B = slot 2)
COLOR = {"A": "#2a78d6", "B": "#eb6834"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0"
MARKERS = ["o", "s", "^", "D"]
LABEL = {"A": "A: pixel ES", "B": "B: pixel + DINOv2 ES"}
# Experiment C thresholds (assumptions, stated in docs/experiments_AB.md): relative change between 20k and 30k
IMPROVING, STALLED = 0.10, 0.05


def load_run(d):
    exp = "A" if os.path.basename(d).startswith("expA") else "B" if os.path.basename(d).startswith("expB") else None
    cfg = yaml.safe_load(open(os.path.join(d, "config.yaml"))) if os.path.exists(os.path.join(d, "config.yaml")) else {}
    evals = {}
    for f in sorted(glob.glob(os.path.join(d, "eval", "step*.json"))):
        r = json.load(open(f))
        evals[r["step"]] = r
    return {"dir": d, "name": os.path.basename(d), "exp": exp, "seed": cfg.get("seed"), "evals": evals}


def metrics(r, w):
    """One snapshot's headline numbers for weights w."""
    m = r[w]
    fe = m.get("feature_ed", {})
    return {"fid": m.get("fid"), "pixel": m["energy_distance"]["mean"],
            "inception": fe.get("inception", {}).get("mean"), "dinov2": fe.get("dinov2", {}).get("mean"),
            "blockiness": m["blockiness"], "nn_acc": m.get("nn_classifier_accuracy"),
            "conf_acc": m.get("class_confusion_argmin_accuracy"), "diversity": m["diversity_ratio_mean"]}


def fmt(v, nd=3):
    return "–" if v is None else f"{v:.{nd}f}"


def rel_drop(a, b):
    """(a - b) / a: positive = improved (lower is better for every metric here)."""
    return None if a is None or b is None or a == 0 else (a - b) / abs(a)


def table(runs, w):
    rows = ["| run | seed | step | FID-10k | pixel D1 | Inception D1 | DINOv2 D1 | blockiness | 1-NN class acc | "
            "D1-confusion acc | diversity |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        for s, r in sorted(run["evals"].items()):
            m = metrics(r, w)
            rows.append(f"| {run['name']} | {run['seed']} | {s} | {fmt(m['fid'], 1)} | {fmt(m['pixel'])} | "
                        f"{fmt(m['inception'])} | {fmt(m['dinov2'])} | {fmt(m['blockiness'], 2)} | {fmt(m['nn_acc'], 3)} | "
                        f"{fmt(m['conf_acc'], 2)} | {fmt(m['diversity'], 2)} |")
    return "\n".join(rows)


def references(runs):
    for run in runs:
        for r in run["evals"].values():
            b, fb = r["baselines"], r.get("feature_baselines", {})
            return {"pixel": (b["train"]["mean"], b["other_class"]["mean"]),
                    "inception": (fb.get("inception", {}).get("train"), fb.get("inception", {}).get("other_class")),
                    "dinov2": (fb.get("dinov2", {}).get("train"), fb.get("dinov2", {}).get("other_class")),
                    "real_nn_acc": r.get("real_heldout_nn_classifier_accuracy"),
                    "ref_blockiness": r.get("reference_blockiness")}
    return None


def decide_c(runs, w):
    """Experiment C rule on A's numbers (20k -> 30k); the visual check of the grids is still required."""
    out = []
    for run in [r for r in runs if r["exp"] == "A"]:
        ev = run["evals"]
        if not {20000, 30000} <= set(ev):
            out.append(f"- {run['name']}: needs the 20k and 30k evaluations")
            continue
        m20, m30 = metrics(ev[20000], w), metrics(ev[30000], w)
        d = {k: rel_drop(m20[k], m30[k]) for k in ("fid", "inception", "pixel")}
        mono = all(metrics(ev[a], w)["fid"] is not None and metrics(ev[b], w)["fid"] is not None
                   and metrics(ev[b], w)["fid"] < metrics(ev[a], w)["fid"]
                   for a, b in ((10000, 20000), (20000, 30000)) if a in ev and b in ev)
        if mono and d["fid"] is not None and d["fid"] >= IMPROVING and (d["inception"] or 0) >= IMPROVING:
            verdict = "quality still improving -> candidate for the 200k run (if the grids show structure)"
        elif (d["pixel"] or 0) >= STALLED and abs(d["fid"] or 0) < STALLED and abs(d["inception"] or 0) < STALLED:
            verdict = "pixel D1 still falling but FID / Inception D1 stalled -> test new losses first"
        else:
            verdict = "in between -> decide from the grids and from B"
        out.append(f"- {run['name']} (seed {run['seed']}), 20k -> 30k relative improvement: FID {fmt(d['fid'], 3)}, "
                   f"Inception D1 {fmt(d['inception'], 3)}, pixel D1 {fmt(d['pixel'], 3)}; FID monotone 10k->20k->30k: "
                   f"{mono}. **{verdict}**")
    pairs = {}
    for run in runs:
        if 30000 in run["evals"]:
            pairs.setdefault(run["seed"], {})[run["exp"]] = metrics(run["evals"][30000], w)
    for seed, p in sorted(pairs.items(), key=lambda kv: str(kv[0])):
        if {"A", "B"} <= set(p):
            out.append(f"- seed {seed} at 30k, B vs A: FID {fmt(p['B']['fid'], 1)} vs {fmt(p['A']['fid'], 1)}; Inception D1 "
                       f"{fmt(p['B']['inception'])} vs {fmt(p['A']['inception'])}; pixel D1 {fmt(p['B']['pixel'])} vs "
                       f"{fmt(p['A']['pixel'])}")
    return "\n".join(out) if out else "- no A run with evaluations yet"


def figure(runs, w, refs, path):
    panels = [("fid", "FID-10k (Inception, vs CIFAR-10 train)", None),
              ("pixel", "held-out pixel energy distance D1", "pixel"),
              ("inception", "Inception-pool3 energy distance D1", "inception"),
              ("dinov2", "DINOv2 ViT-S/14 energy distance D1", "dinov2")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (key, title, refkey) in zip(axes.flat, panels):
        ax.set_facecolor("#fcfcfb")
        for run in runs:
            st = sorted(s for s in run["evals"] if metrics(run["evals"][s], w)[key] is not None)
            if not st:
                continue
            v = [metrics(run["evals"][s], w)[key] for s in st]
            k = sorted({r["seed"] for r in runs if r["exp"] == run["exp"]}, key=str).index(run["seed"])
            ax.plot(np.array(st) / 1000, v, color=COLOR[run["exp"]], lw=2, marker=MARKERS[k % 4], ms=6,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=f"{LABEL[run['exp']]}, seed {run['seed']}")
        if refkey and refs and refs[refkey][0] is not None:
            for val, txt, ls in ((refs[refkey][0], "right-class real (floor)", ":"),
                                 (refs[refkey][1], "wrong-class real", "--")):
                ax.axhline(val, color=MUTED, lw=1, ls=ls, zorder=0)
                ax.annotate(txt, (1.0, val), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                            textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=INK2)
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("training step (thousands)", color=INK2, fontsize=9)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
    h, lab = axes.flat[0].get_legend_handles_labels()
    if h:
        fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=min(len(h), 2), frameon=False,
                   fontsize=8.5, labelcolor=INK)
    fig.suptitle(f"Experiments A / B: {w.upper()} weights, lower is better on every panel (log scale)",
                 x=0.01, ha="left", y=0.995, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def montage(runs, w, path):
    """Rows = evaluated steps; columns = runs; each cell the 10 classes x 10 grid of that snapshot."""
    cols = [r for r in runs if r["evals"]]
    steps = sorted({s for r in cols for s in r["evals"]})
    if not cols or not steps:
        return False
    fig, axes = plt.subplots(len(steps), len(cols), figsize=(3.2 * len(cols), 3.3 * len(steps)), dpi=150,
                             squeeze=False)
    for i, s in enumerate(steps):
        for j, r in enumerate(cols):
            ax = axes[i][j]
            ax.axis("off")
            f = os.path.join(r["dir"], "eval", f"step{s:07d}_samples_{w}.png")
            if os.path.exists(f):
                ax.imshow(plt.imread(f))
            if i == 0:
                ax.set_title(f"{LABEL[r['exp']]}\nseed {r['seed']}", fontsize=9, color=INK)
            if j == 0:
                ax.text(-0.04, 0.5, f"{s // 1000}k", transform=ax.transAxes, ha="right", va="center", fontsize=10,
                        color=INK)
    fig.suptitle(f"Samples ({w.upper()}), rows = training step, 10 classes x 10 per cell (fixed noise)", fontsize=10,
                 color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--weights", default="ema", choices=["ema", "raw"])
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "expAB"))
    a = ap.parse_args()
    dirs = a.runs or sorted(glob.glob(os.path.join(ROOT, "runs", "exp[AB]_*")))
    runs = [load_run(d) for d in dirs if os.path.isdir(d)]
    runs = sorted([r for r in runs if r["exp"]], key=lambda r: (r["exp"], str(r["seed"])))
    refs = references(runs)
    md = [f"# Experiments A / B summary ({a.weights.upper()} weights)", "", table(runs, a.weights), ""]
    if refs:
        md += ["References (held-out test set, 1000/class): pixel D1 right-class real "
               f"{fmt(refs['pixel'][0])}, wrong-class real {fmt(refs['pixel'][1])}; stage-4 block-shuffle thresholds "
               "(CIFAR-10): 16x16 0.214, 4x4 0.397, pixels 0.431. Inception D1 right/wrong-class real "
               f"{fmt(refs['inception'][0])} / {fmt(refs['inception'][1])}; DINOv2 D1 {fmt(refs['dinov2'][0])} / "
               f"{fmt(refs['dinov2'][1])}. Real test images: 1-NN class acc {fmt(refs['real_nn_acc'])}, blockiness "
               f"{fmt(refs['ref_blockiness'], 2)}. DINOv2 D1 is the training space of B (circular for B); FID and "
               "Inception D1 are not.", ""]
    md += ["## Experiment C rule (numbers only; look at the sample montage too)", "", decide_c(runs, a.weights), ""]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    figure(runs, a.weights, refs, f"{a.out}_metrics_{a.weights}.png")
    if montage(runs, a.weights, f"{a.out}_samples_{a.weights}.png"):
        md += [f"Figures: `{os.path.relpath(a.out, ROOT)}_metrics_{a.weights}.png`, "
               f"`{os.path.relpath(a.out, ROOT)}_samples_{a.weights}.png`."]
    text = "\n".join(md)
    open(f"{a.out}_summary_{a.weights}.md", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
