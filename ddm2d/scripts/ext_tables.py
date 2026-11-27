"""ddm2dx X3: markdown tables for docs/beta_exploration.md, from runs/ext/eval/*.csv (ext_eval.py summary). Writes runs/ext/eval/tables.md.
Cell format: seed mean [seed min, seed max] of the per-seed means over 10 evaluation repeats; n = number of seeds."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "runs/ext/eval"
W9 = [0.30, 0.02, 0.15, 0.05, 0.20, 0.03, 0.10, 0.07, 0.08]


def rd(fn):
    return list(csv.DictReader(open(EV / fn)))


pr = rd("mmd_per_run.csv")
by = defaultdict(list)
for r in pr:
    if int(r["seed"]) in (0, 1):          # matched seeds; seed 2 in its own table below
        by[(r["group"], int(r["nfe"]))].append(r)


def cell(g, nfe, key="mmd2_u", fmt="{:.1e}"):
    rs = by.get((g, nfe), [])
    if not rs:
        return "–"
    v = [float(r[key]) for r in rs]
    if any(np.isnan(v)):
        return "NaN (" + ", ".join("NaN" if np.isnan(x) else fmt.format(x) for x in v) + ")"
    if len(v) == 1:
        return fmt.format(v[0]) + " (n=1)"
    return f"{fmt.format(np.mean(v))} [{fmt.format(min(v))}, {fmt.format(max(v))}]" + (f" n={len(v)}" if len(v) != 2 else "")


out = []
nulls = json.loads((EV / "nulls.json").read_text())
out.append("## Null floors (p0 vs p0, 4096 vs 4096)\n")
out.append("| dataset | MMD²_u mean | sd (single evaluation) | sd of a 10-repeat mean | TV mean | TV p95 | rare-mode rel. err. mean | p95 | off-mode p95 |")
out.append("|---|---|---|---|---|---|---|---|---|")
for ds, n in nulls.items():
    out.append(f"| {ds} | {n['mmd2_u_mean']:.1e} | {n['mmd2_u_sd']:.1e} | {n['mmd2_u_sd'] / np.sqrt(10):.1e} | {n['tv_mean']:.4f} | {n['tv_p95']:.4f} | "
               f"{n['rare_rel_err_mean']:.3f} | {n['rare_rel_err_p95']:.3f} | {n['off_mode_p95']:.4f} |")


def mmd_table(title, groups, nfes=(5, 10, 20, 50, 100)):
    out.append(f"\n## {title}\n")
    out.append("| run group | " + " | ".join(f"NFE {n}" for n in nfes) + " |")
    out.append("|---|" + "---|" * len(nfes))
    for g in groups:
        out.append(f"| `{g}` | " + " | ".join(cell(g, n) for n in nfes) + " |")


mmd_table("E1a: λ = 0 (B family), paper GMM, MMD²_u",
          ["gmm_A", "gmm_B_b0.1", "gmm_B_b0.5", "gmm_B_b1", "gmm_B_b1.5", "gmm_B_b1.99", "gmm_B", "gmm_B_b3", "gmm_B_b3.5"])
mmd_table("E1b: λ = 1 (C family), paper GMM, MMD²_u", ["gmm_A", "gmm_C", "gmm_C_b0.5", "gmm_C_b1", "gmm_C_b1.5", "gmm_C_b1.99", "gmm_C_b2"])
mmd_table("E2: gmm2w and gmm9w, MMD²_u", ["gmm2w_A", "gmm2w_B", "gmm2w_C", "gmm9w_A", "gmm9w_B", "gmm9w_C", "gmm9w_B_b1", "gmm9w_C_b1.5"])
mmd_table("E4: other distances, paper GMM, MMD²_u",
          ["gmm_A", "gmm_B", "gmm_B_L1b1", "gmm_B_pH0.1", "gmm_C", "gmm_C_L1b1", "gmm_C_L1b0.5", "gmm_C_pH0.1", "gmm_C_pH1", "gmm_C_imq", "gmm_C_rbf", "gmm_C_lap"],
          nfes=(5, 10, 100))
mmd_table("E4: other distances, gmm9w (ρ runs seed 0 only), MMD²_u",
          ["gmm9w_A", "gmm9w_B", "gmm9w_B_b1", "gmm9w_B_L1b1", "gmm9w_B_pH0.1", "gmm9w_C", "gmm9w_C_b1.5", "gmm9w_C_L1b1", "gmm9w_C_L1b0.5", "gmm9w_C_pH0.1",
           "gmm9w_C_pH1", "gmm9w_C_imq", "gmm9w_C_rbf", "gmm9w_C_lap"], nfes=(5, 10, 100))

# seed-2 check (X3 extras + paper seed-2 references)
s2 = defaultdict(dict)
for r in pr:
    s2[r["group"]][(int(r["seed"]), int(r["nfe"]))] = float(r["mmd2_u"])
g2 = [g for g in s2 if any(k[0] == 2 for k in s2[g])]
if g2:
    out.append("\n## Seed 2 (X3 extras and paper seed-2 references), MMD²_u per seed: s0 / s1 / s2\n")
    out.append("| run group | NFE 5 | NFE 10 | NFE 100 |")
    out.append("|---|---|---|---|")
    for g in sorted(g2):
        out.append(f"| `{g}` | " + " | ".join(" / ".join(f"{s2[g][(sd, n)]:.1e}" if (sd, n) in s2[g] else "–" for sd in (0, 1, 2)) for n in (5, 10, 100)) + " |")

# mode weights (final samples)
out.append("\n## Final-sample mode weights (E2, E4 on gmm9w): TV ½Σ|p̂_k − π_k|, rare-mode max relative error, off-mode fraction\n")
out.append("Per-seed means over 10 repeats of 4096 samples; cell = seed mean [min, max]. Null: see the null-floor table.\n")
out.append("| run group | TV NFE 5 | TV NFE 10 | TV NFE 100 | rare rel. err. NFE 10 | rare rel. err. NFE 100 | off-mode NFE 10 | off-mode NFE 100 |")
out.append("|---|---|---|---|---|---|---|---|")
for g in ["gmm_A", "gmm_B", "gmm_C", "gmm2w_A", "gmm2w_B", "gmm2w_C", "gmm9w_A", "gmm9w_B", "gmm9w_C", "gmm9w_B_b1", "gmm9w_C_b1.5", "gmm9w_B_L1b1",
          "gmm9w_B_pH0.1", "gmm9w_C_L1b1", "gmm9w_C_L1b0.5", "gmm9w_C_pH0.1", "gmm9w_C_pH1", "gmm9w_C_imq", "gmm9w_C_rbf", "gmm9w_C_lap"]:
    if (g, 10) not in by:
        continue
    out.append(f"| `{g}` | {cell(g, 5, 'tv', '{:.3f}')} | {cell(g, 10, 'tv', '{:.3f}')} | {cell(g, 100, 'tv', '{:.3f}')} | "
               f"{cell(g, 10, 'rare_max_rel_err', '{:.2f}')} | {cell(g, 100, 'rare_max_rel_err', '{:.2f}')} | "
               f"{cell(g, 10, 'off_mode', '{:.3f}')} | {cell(g, 100, 'off_mode', '{:.3f}')} |")

# per-mode fractions on gmm9w at NFE 10 and 100
raw = rd("mmd_raw.csv")
fr = defaultdict(list)
for r in raw:
    if int(r["seed"]) in (0, 1):
        fr[(r["group"], int(r["nfe"]))].append([float(v) for v in r["frac"].split()])
for nfe in (10, 100):
    out.append(f"\n## gmm9w per-mode fractions at NFE {nfe} (mean over seeds × repeats)\n")
    out.append("| run group | " + " | ".join(f"k={k} (π={w})" for k, w in enumerate(W9)) + " |")
    out.append("|---|" + "---|" * 9)
    for g in ["gmm9w_A", "gmm9w_B", "gmm9w_C", "gmm9w_B_b1", "gmm9w_C_b1.5", "gmm9w_B_L1b1", "gmm9w_B_pH0.1", "gmm9w_C_L1b1", "gmm9w_C_L1b0.5",
              "gmm9w_C_pH0.1", "gmm9w_C_pH1", "gmm9w_C_imq", "gmm9w_C_rbf", "gmm9w_C_lap"]:
        if (g, nfe) in fr:
            m = np.mean(fr[(g, nfe)], 0)
            out.append(f"| `{g}` | " + " | ".join(f"{v:.3f}" for v in m) + " |")
for ds, w in [("gmm2w", [0.8, 0.2])]:
    out.append(f"\n## gmm2w mode fractions (minor mode π = 0.2), mean over seeds × repeats\n")
    out.append("| run group | NFE 5 | NFE 10 | NFE 100 |")
    out.append("|---|---|---|---|")
    for g in ["gmm2w_A", "gmm2w_B", "gmm2w_C"]:
        out.append(f"| `{g}` | " + " | ".join(f"{np.mean(fr[(g, n)], 0)[1]:.3f}" for n in (5, 10, 100)) + " |")

# conditional mode weights
cr = rd("condw.csv")
cw = defaultdict(list)
for r in cr:
    if int(r["seed"]) in (0, 1):
        cw[(r["group"], float(r["t"]))].append(r)
out.append("\n## Conditional mode weights at high t (256 x_t, 4096 draws each)\n")
out.append("Cells: mean TV between the draws' mode fractions and the exact responsibilities r_k(x_t) (seed mean; n seeds); in brackets the "
           "mass outside the dominant mode, model / exact. Null TV = exact posterior draws through the same assignment.\n")
out.append("| run group | t = 0.99 | t = 0.88 | t = 0.7 | t = 0.45 |")
out.append("|---|---|---|---|---|")
groups = sorted({k[0] for k in cw}, key=lambda g: (g.split("_")[0], g))
for g in groups:
    cells = []
    for t in (0.99, 0.88, 0.7, 0.45):
        rs = cw.get((g, t), [])
        if not rs:
            cells.append("–")
            continue
        tv = np.mean([float(r["mean_tv"]) for r in rs])
        nm = np.mean([float(r["nondominant_mass_model"]) for r in rs])
        cells.append(f"{tv:.3f} ({nm:.2f} / {float(rs[0]['nondominant_mass_exact']):.2f})")
    out.append(f"| `{g}` (n={len(cw[(g, 0.99)])}) | " + " | ".join(cells) + " |")
out.append("| null (exact draws) | " + " | ".join(
    "/".join(f"{np.mean([float(r['null_mean_tv']) for r in cr if r['dataset'] == ds and float(r['t']) == t]):.3f}" for ds in ("gmm", "gmm2w", "gmm9w"))
    for t in (0.99, 0.88, 0.7, 0.45)) + " |  (gmm/gmm2w/gmm9w)")

# denoiser location
dr = rd("denoiser.csv")
dd = defaultdict(list)
for r in dr:
    if int(r["seed"]) in (0, 1):
        dd[(r["group"], float(r["t"]))].append(r)
out.append("\n## E1a: where the λ = 0 denoiser sits (paper GMM; mean distance of x̂ over 256 x_t; seed mean)\n")
out.append("x̂ = mean of 8 ξ draws. Columns: distance to the exact posterior mean / geometric median / coordinate-wise median / nearest component "
           "mean, and the share of x_t for which each is the closest of the four.\n")
out.append("| run group | t | d(mean) | d(geo. median) | d(coord. median) | d(nearest mode) | closest: mean / gmed / cmed / mode | ξ spread |")
out.append("|---|---|---|---|---|---|---|---|")
for g in ["gmm_A", "gmm_B_b0.1", "gmm_B_b0.5", "gmm_B_b1", "gmm_B_L1b1", "gmm_B_pH0.1", "gmm_B_b1.5", "gmm_B_b1.99", "gmm_B", "gmm_B_b3", "gmm_B_b3.5", "gmm_C"]:
    for t in (0.99, 0.88, 0.45):
        rs = dd.get((g, t), [])
        if not rs:
            continue
        a = lambda k: np.mean([float(r[k]) for r in rs])
        out.append(f"| `{g}` | {t} | {a('d_mean'):.2f} | {a('d_gmed'):.2f} | {a('d_cmed'):.2f} | {a('d_nearest_mode'):.2f} | "
                   f"{a('closest_mean'):.2f} / {a('closest_gmed'):.2f} / {a('closest_cmed'):.2f} / {a('closest_mode'):.2f} | {a('xi_spread'):.3f} |")
    rs = dd.get(("gmm_A", 0.99), [])
out.append("\nReference separation (exact quantities): " + "; ".join(
    f"t = {t}: d(mean, geo. median) = {np.mean([float(r['ref_d_mean_gmed']) for r in dd[('gmm_A', t)]]):.2f}" for t in (0.99, 0.88, 0.45)))

# posterior std at selected t
sr = [r for r in rd("posterior_std.csv") if r["grid"] == "paper"]
st = defaultdict(list)
ex = {}
for r in sr:
    if int(r["seed"]) not in (0, 1):
        continue
    st[(r["group"], round(float(r["t"]), 3))].append(float(r["std_mean"]))
    ex[(r["dataset"], round(float(r["t"]), 3))] = float(r["exact_std_mean"])
ts = sorted({k[1] for k in st})
pick = [ts[0], ts[2], ts[4], ts[6], ts[8], ts[9]]
out.append("\n## Posterior std of x̂ vs t (paper grid; 4096 x_t × 8 draws; seed mean)\n")
out.append("| run group | " + " | ".join(f"t = {t:.2f}" for t in pick) + " |")
out.append("|---|" + "---|" * len(pick))
for ds in ("gmm", "gmm2w", "gmm9w"):
    out.append(f"| **exact ({ds})** | " + " | ".join(f"{ex[(ds, t)]:.3f}" for t in pick) + " |")
for g in sorted({k[0] for k in st}, key=lambda g: (g.split("_")[0], g)):
    out.append(f"| `{g}` | " + " | ".join(f"{np.mean(st[(g, t)]):.3g}" for t in pick) + " |")

# DIAGNOSTIC: EMA snapshots (not the protocol)
snaps = sorted((EV / "snaps").glob("*.json")) if (EV / "snaps").exists() else []
if snaps:
    out.append("\n## DIAGNOSTIC (not the protocol): MMD²_u of EMA snapshots, mean of 2 repeats, NFE 5 / NFE 100; n_finite < 4096 marked *\n")
    steps = list(range(10_000, 100_001, 10_000))
    out.append("| run | " + " | ".join(f"{k // 1000}k" for k in steps) + " |")
    out.append("|---|" + "---|" * len(steps))
    for f in snaps:
        d = json.loads(f.read_text())
        cells = []
        for k in steps:
            c = []
            for nfe in (5, 100):
                rs = [r for r in d["rows"] if r["step"] == k and r["nfe"] == nfe]
                v = np.mean([r["mmd2_u"] for r in rs])
                c.append(("NaN" if not np.isfinite(v) else f"{v:.0e}") + ("*" if any(r["n_finite"] < 4096 for r in rs) else ""))
            cells.append(" / ".join(c))
        out.append(f"| `{d['run']}` | " + " | ".join(cells) + " |")

(EV / "tables.md").write_text("# ddm2dx X3 tables (generated by scripts/ext_tables.py from runs/ext/eval/*.csv)\n\n" + "\n".join(out) + "\n")
print("wrote", EV / "tables.md")
