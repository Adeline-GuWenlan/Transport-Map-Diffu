"""UNet vs DiT backbone comparison (docs/experiments_unet.md): per-metric tables, cost table, metric-vs-step figure,
30k sample montage.

    python scripts/summarize_backbones.py [--weights ema|raw]

Reads the 8 runs runs/{expA,expB,unetA,unetB}_s{0,1}: <run>/eval/step*.json, <run>/config.yaml, <run>/metrics.jsonl,
the slurm .out of each run (runs/slurm-tmap_*.out), and runs/cost_unet_S1.json (params and GMACs, one counter for both
backbones). Reuses the readers of scripts/summarize_ab.py (imported, not changed). Writes
runs/backbone_summary_<weights>.md, runs/backbone_metrics_<weights>.png, runs/backbone_samples_<weights>.png.
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import summarize_ab as sab  # noqa: E402  (readers: load_run, metrics, references, fmt; palette constants)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = sab.ROOT
RUNS = os.path.join(ROOT, "runs")
STEPS = [5000, 10000, 20000, 30000]
# arm = (backbone, loss). Order fixes table columns and montage columns.
ARMS = [("dit", "A"), ("unet", "A"), ("dit", "B"), ("unet", "B")]
PREFIX = {("dit", "A"): "expA", ("dit", "B"): "expB", ("unet", "A"): "unetA", ("unet", "B"): "unetB"}
BB_NAME = {"dit": "DiT-B/2", "unet": "UNet"}
LOSS_NAME = {"A": "pixel ES", "B": "pixel + DINOv2 ES"}
LS = {"dit": "-", "unet": "--"}
SEEDS = [0, 1]
METRICS = [("fid", "FID-10k (vs CIFAR-10 train)", 1, True),
           ("pixel", "held-out pixel energy distance D1", 3, True),
           ("inception", "Inception-pool3 D1", 3, True),
           ("dinov2", "DINOv2 ViT-S/14 D1 (training space of B: circular for B)", 3, True),
           ("blockiness", "blockiness (real test images about 1.0)", 2, False),
           ("nn_acc", "class 1-NN agreement (real test images 0.288)", 3, False),
           ("diversity", "diversity ratio (1 = as diverse as real)", 2, False)]


def arm_label(arm):
    return f"{BB_NAME[arm[0]]} {arm[1]} ({LOSS_NAME[arm[1]]})"


def load_all():
    runs = {}
    for arm in ARMS:
        for s in SEEDS:
            d = os.path.join(RUNS, f"{PREFIX[arm]}_s{s}")
            r = sab.load_run(d)
            if r["seed"] != s:
                raise SystemExit(f"{d}: config seed {r['seed']} != directory seed {s}")
            r["arm"], r["exp"] = arm, arm[1]
            r["cfg"] = yaml.safe_load(open(os.path.join(d, "config.yaml")))
            runs[(arm, s)] = r
    return runs


def half_range(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None, None
    return sum(vals) / len(vals), (max(vals) - min(vals)) / 2


def metric_table(runs, w, key, nd):
    head = "| step | " + " | ".join(f"{BB_NAME[a[0]]} {a[1]} s0 | {BB_NAME[a[0]]} {a[1]} s1 | "
                                     f"**{BB_NAME[a[0]]} {a[1]} mean ± hr**" for a in ARMS) + " |"
    rows = [head, "|---|" + "---|" * (3 * len(ARMS))]
    for st in STEPS:
        cells = []
        for a in ARMS:
            v = [sab.metrics(runs[(a, s)]["evals"][st], w)[key] if st in runs[(a, s)]["evals"] else None for s in SEEDS]
            m, h = half_range(v)
            cells += [sab.fmt(v[0], nd), sab.fmt(v[1], nd),
                      "–" if m is None else f"**{m:.{nd}f} ± {h:.{nd}f}**"]
        rows.append(f"| {st // 1000}k | " + " | ".join(cells) + " |")
    return "\n".join(rows)


TS1 = re.compile(r"^\[(\w{3} \w{3} +\d+ \d\d:\d\d:\d\d) \w+ (\d{4})\]")
TS2 = re.compile(r"^\[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)")


def parse_ts(line):
    m = TS1.match(line)
    if m:
        return dt.datetime.strptime(f"{m.group(1)} {m.group(2)}", "%a %b %d %H:%M:%S %Y")
    m = TS2.match(line)
    return dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S") if m else None


def slurm_out(run_dir):
    tag = f"run {run_dir};"
    hits = [f for f in sorted(glob.glob(os.path.join(RUNS, "slurm-tmap_*.out"))) if tag in open(f).read()]
    return hits[0] if len(hits) == 1 else None


def gpu_hours(path):
    """Wall time of each allocation segment in the run's slurm .out: from a 'restart k' line to the last timestamp
    before the next restart (or EOF). Split into training and the post-training evaluation ('tmap.train exited with 0'
    -> 'done'). One GPU per job, so wall hours = GPU-hours. Gaps between requeues are not counted."""
    segs, train_end, done, restarts = [], None, None, 0
    cur = None
    for line in open(path):
        t = parse_ts(line)
        if t is None:
            continue
        if "restart" in line and "job" in line:
            if cur:
                segs.append(cur)
            cur = [t, t]
            restarts = max(restarts, int(re.search(r"restart (\d+)", line).group(1)))
        elif cur:
            cur[1] = t
        if "tmap.train exited with 0" in line:
            train_end = t
        if line.rstrip().endswith("] done"):
            done = t
    if cur:
        segs.append(cur)
    total = sum((b - a).total_seconds() for a, b in segs) / 3600
    ev = (done - train_end).total_seconds() / 3600 if done and train_end else None
    return {"total": total, "eval": ev, "train": None if ev is None else total - ev, "restarts": restarts}


def cost_rows(runs):
    cost = json.load(open(os.path.join(RUNS, "cost_unet_S1.json")))
    rows = ["| run | grad_chunk | s/step (median) | train GPU-h | eval GPU-h | total GPU-h | requeues | peak mem GiB | "
            "params M | GMACs/sample |", "|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for a in ARMS:
        for s in SEEDS:
            r = runs[(a, s)]
            ml = [json.loads(x) for x in open(os.path.join(r["dir"], "metrics.jsonl"))]
            sps = statistics.median(1 / x["steps_per_s"] for x in ml if x.get("steps_per_s"))
            mem = max(x.get("max_mem_gib", 0) for x in ml)
            so = slurm_out(r["dir"])
            g = gpu_hours(so) if so else {"total": None, "eval": None, "train": None, "restarts": None}
            c = cost["unet" if a[0] == "unet" else "dit-b2"]
            out[(a, s)] = {"s_per_step": sps, **g}
            rows.append(f"| {r['name']} | {r['cfg']['train']['grad_chunk']} | {sps:.3f} | {sab.fmt(g['train'], 2)} | "
                        f"{sab.fmt(g['eval'], 2)} | {sab.fmt(g['total'], 2)} | {g['restarts']} | {mem:.1f} | "
                        f"{c['params_M']:.1f} | {c['gmacs_per_sample']:.2f} |")
    return "\n".join(rows), out


def figure(runs, w, refs, path):
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.8), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    refkey = {"pixel": "pixel", "inception": "inception", "dinov2": "dinov2"}
    for ax, (key, title, _, logy) in zip(axes.flat, METRICS):
        ax.set_facecolor("#fcfcfb")
        for a in ARMS:
            for k, s in enumerate(SEEDS):
                r = runs[(a, s)]
                st = [x for x in STEPS if x in r["evals"] and sab.metrics(r["evals"][x], w)[key] is not None]
                v = [sab.metrics(r["evals"][x], w)[key] for x in st]
                ax.plot(np.array(st) / 1000, v, color=sab.COLOR[a[1]], ls=LS[a[0]], lw=2, marker=sab.MARKERS[k], ms=7,
                        markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=f"{arm_label(a)}, seed {s}")
        if key in refkey and refs and refs[refkey[key]][0] is not None:
            for val, txt, ls in ((refs[refkey[key]][0], "right-class real (floor)", ":"),
                                 (refs[refkey[key]][1], "wrong-class real", "-.")):
                ax.axhline(val, color=sab.MUTED, lw=1, ls=ls, zorder=0)
                ax.annotate(txt, (1.0, val), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                            textcoords="offset points", ha="right", va="bottom", fontsize=7.5, color=sab.INK2)
        if key == "blockiness" and refs and refs.get("ref_blockiness"):
            ax.axhline(refs["ref_blockiness"], color=sab.MUTED, lw=1, ls=":", zorder=0)
        if key == "nn_acc" and refs and refs.get("real_nn_acc"):
            ax.axhline(refs["real_nn_acc"], color=sab.MUTED, lw=1, ls=":", zorder=0)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=9.5, color=sab.INK)
        ax.set_xlabel("training step (thousands)", color=sab.INK2, fontsize=8.5)
        ax.set_xticks([5, 10, 20, 30])
        ax.grid(True, color=sab.GRID, lw=0.6)
        ax.tick_params(colors=sab.INK2, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(sab.MUTED)
    leg = axes.flat[-1]
    leg.axis("off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    leg.legend(h, lab, loc="center", frameon=False, fontsize=8.5, labelcolor=sab.INK,
               title="solid = DiT-B/2, dashed = UNet\nblue = A (pixel ES), orange = B (+DINOv2 ES)\n"
                     "marker: circle seed 0, square seed 1", title_fontsize=8.5)
    fig.suptitle(f"UNet vs DiT-B/2, {w.upper()} weights. Lower is better on the log panels; the dotted line is the "
                 "real-data reference.", x=0.01, ha="left", y=0.995, fontsize=11, color=sab.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def montage(runs, w, path, step=30000):
    fig, axes = plt.subplots(len(SEEDS), len(ARMS), figsize=(3.6 * len(ARMS), 4.1 * len(SEEDS)), dpi=110,
                             squeeze=False)
    for i, s in enumerate(SEEDS):
        for j, a in enumerate(ARMS):
            ax = axes[i][j]
            ax.axis("off")
            f = os.path.join(runs[(a, s)]["dir"], "eval", f"step{step:07d}_samples_{w}.png")
            if os.path.exists(f):
                ax.imshow(plt.imread(f), interpolation="nearest")
            ax.set_title(f"{arm_label(a)}\nseed {s} ({runs[(a, s)]['name']})", fontsize=8.5, color=sab.INK)
    fig.suptitle(f"{step // 1000}k-step samples ({w.upper()}): rows = classes 0-9, fixed noise (seed 1234), same for "
                 "every panel", fontsize=10, color=sab.INK)
    fig.tight_layout(h_pad=2.0)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="ema", choices=["ema", "raw"])
    a = ap.parse_args()
    w = a.weights
    runs = load_all()
    missing = [(r["name"], st) for r in runs.values() for st in STEPS if st not in r["evals"]]
    refs = sab.references(list(runs.values()))
    md = [f"# Backbone comparison: UNet vs DiT-B/2 ({w.upper()} weights)", "",
          "Generated by `scripts/summarize_backbones.py` from the committed run files "
          "(`runs/{expA,expB,unetA,unetB}_s{0,1}/eval/step*.json`). Nothing is retyped. "
          "Arms: A = pixel energy score, B = pixel + DINOv2 energy score. hr = half-range of the two seeds "
          "((max - min) / 2).", ""]
    if missing:
        md += [f"**Missing evaluations:** {missing}", ""]
    for key, title, nd, _ in METRICS:
        md += [f"## {title}", "", metric_table(runs, w, key, nd), ""]
    if refs:
        md += ["References (held-out test set, 1000/class): pixel D1 right-class real "
               f"{sab.fmt(refs['pixel'][0])}, wrong-class real {sab.fmt(refs['pixel'][1])}. Inception D1 right/wrong-class "
               f"real {sab.fmt(refs['inception'][0])} / {sab.fmt(refs['inception'][1])}; DINOv2 D1 "
               f"{sab.fmt(refs['dinov2'][0])} / {sab.fmt(refs['dinov2'][1])}. Real test images: 1-NN class agreement "
               f"{sab.fmt(refs['real_nn_acc'])}, blockiness {sab.fmt(refs['ref_blockiness'], 2)}.", ""]
    ct, cost = cost_rows(runs)
    md += ["## Measured cost", "", ct, "",
           "s/step: median over the 300 `metrics.jsonl` lines of 1 / `steps_per_s`. GPU-h: wall time of each allocation "
           "segment in the run's slurm `.out` (1 L40S per job; gaps between requeues not counted); eval = the "
           "post-training evaluation of the 4 snapshots. Params and GMACs/sample: `runs/cost_unet_S1.json` "
           "(`torch.utils.flop_counter`, same counter for both; the DiT paper quotes 23.0 GMACs for DiT-B/2). "
           "grad_chunk only changes speed (the gradient is exact, `tests/test_train.py`).", ""]
    for arm in ARMS:
        tot = [cost[(arm, s)]["total"] for s in SEEDS]
        sps = [cost[(arm, s)]["s_per_step"] for s in SEEDS]
        m, h = half_range(sps)
        md.append(f"- {arm_label(arm)}: s/step {m:.3f} ± {h:.3f}, total GPU-h per run "
                  f"{sab.fmt(half_range(tot)[0], 2)}")
    total = sum(c["total"] for (arm, s), c in cost.items() if arm[0] == "unet" and c["total"] is not None)
    md += ["", f"UNet runs, total GPU-h (4 runs): {total:.2f}.", ""]
    figure(runs, w, refs, os.path.join(RUNS, f"backbone_metrics_{w}.png"))
    montage(runs, w, os.path.join(RUNS, f"backbone_samples_{w}.png"))
    md += [f"Figures: `runs/backbone_metrics_{w}.png` (metric vs step), `runs/backbone_samples_{w}.png` (30k grids, "
           "4 arms x 2 seeds)."]
    text = "\n".join(md)
    open(os.path.join(RUNS, f"backbone_summary_{w}.md"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
