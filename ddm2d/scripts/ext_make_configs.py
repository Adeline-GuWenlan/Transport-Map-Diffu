"""ddm2dx X1: write configs/ext/*.json for E1a, E1b, E2, E4 (+ one extra) and the run table runs/ext/runs.tsv.

Every config is a copy of configs/paper/gmm_{A,B,C}.json with only name, _comment, dataset, beta, lam, rho, rho_c changed
(spec: claude_jobs/ddm2dx/prompts/_task.md). Usage (from ddm2d/): PYTHONPATH=src python scripts/ext_make_configs.py
"""
import json
from pathlib import Path

from ddm2d import data as D
from ddm2d.losses import median_dist, median_sq_dist

ROOT = Path(__file__).resolve().parents[1]
BASE = {a: json.loads((ROOT / f"configs/paper/gmm_{a}.json").read_text()) for a in "ABC"}
OUT = ROOT / "configs/ext"
SPEC = "claude_jobs/ddm2dx/prompts/_task.md"


def fmt(b):
    return f"{b:g}"


def make(name, arm, dataset, comment, **chg):
    cfg = dict(BASE[arm])
    cfg["name"] = name
    cfg["_comment"] = f"ddm2dx {comment}. Copy of configs/paper/gmm_{arm}.json except {', '.join(['dataset'] + sorted(chg))}. Spec: {SPEC}."
    cfg["dataset"] = dataset
    cfg.update(chg)
    keys = ["name", "_comment", "dataset", "arm", "m", "beta", "lam"] + [k for k in ("rho", "rho_c") if k in cfg]
    ordered = {k: cfg[k] for k in keys}
    ordered.update({k: v for k, v in cfg.items() if k not in ordered})
    return ordered


runs = []        # (experiment, config name, seed)
cfgs = {}

# E1a: arm B family (lam = 0), beta sweep on the paper GMM; E1b: arm C family (lam = 1)
for b in [0.1, 0.5, 1.0, 1.5, 1.99, 3.0, 3.5]:
    n = f"gmm_B_b{fmt(b)}"
    cfgs[n] = make(n, "B", "gmm", f"E1a: arm B family, beta = {b}, lam = 0", beta=b, lam=0.0)
    runs += [("E1a", n, s) for s in (0, 1)]
for b in [0.5, 1.0, 1.5, 1.99, 2.0]:
    n = f"gmm_C_b{fmt(b)}"
    cfgs[n] = make(n, "C", "gmm", f"E1b: arm C family, beta = {b}, lam = 1", beta=b, lam=1.0)
    runs += [("E1b", n, s) for s in (0, 1)]

# E2: unequal mode weights, arms A, B (beta 2, lam 0), C (beta 0.1, lam 1)
for ds in ["gmm2w", "gmm9w"]:
    for a in "ABC":
        n = f"{ds}_{a}"
        cfgs[n] = make(n, a, ds, f"E2: dataset {ds}, arm {a} (paper beta / lam)")
        runs += [("E2", n, s) for s in (0, 1)]

# E4: other distances; paper GMM seeds 0, 1 and gmm9w seed 0
med = {}
for ds in ["gmm", "gmm9w"]:
    x = D.train_set(ds)
    med[ds] = {"M2": median_sq_dist(x), "M": median_dist(x)}
    m2, m1 = med[ds]["M2"], med[ds]["M"]
    e4 = [
        ("C_L1b1", "C", dict(beta=1.0, lam=1.0, rho="l1"), "rho = ||d||_1, lam = 1"),
        ("C_L1b0.5", "C", dict(beta=0.5, lam=1.0, rho="l1"), "rho = ||d||_1^0.5, lam = 1"),
        ("B_L1b1", "B", dict(beta=1.0, lam=0.0, rho="l1"), "rho = ||d||_1 (L1 regression), lam = 0"),
        ("C_pH0.1", "C", dict(beta=1.0, lam=1.0, rho="pseudo_huber", rho_c=0.1), "rho = pseudo-Huber c = 0.1, lam = 1"),
        ("C_pH1", "C", dict(beta=1.0, lam=1.0, rho="pseudo_huber", rho_c=1.0), "rho = pseudo-Huber c = 1, lam = 1"),
        ("B_pH0.1", "B", dict(beta=1.0, lam=0.0, rho="pseudo_huber", rho_c=0.1), "rho = pseudo-Huber c = 0.1, lam = 0"),
        ("C_imq", "C", dict(beta=1.0, lam=1.0, rho="imq", rho_c=1.0), "rho = -k_IMQ, c = 1, lam = 1"),
        ("C_rbf", "C", dict(beta=1.0, lam=1.0, rho="rbf", rho_c=m2),
         f"rho = -k_RBF, sigma^2 = gamma * M^2 with gamma = 1, M^2 = median ||x - x'||^2 of the {ds} training set = {m2:.6g} "
         "(losses.median_sq_dist: 100 batches of 512, seed 0), lam = 1"),
        ("C_lap", "C", dict(beta=1.0, lam=1.0, rho="laplace", rho_c=m1),
         f"rho = -k_exp, sigma = gamma * M with gamma = 1, M = median ||x - x'|| of the {ds} training set = {m1:.6g} "
         "(losses.median_dist: 100 batches of 512, seed 0), lam = 1"),
    ]
    for tag, a, chg, desc in e4:
        n = f"{ds}_{tag}"
        cfgs[n] = make(n, a, ds, f"E4: {desc}", **chg)
        runs += [("E4", n, s) for s in ((0, 1) if ds == "gmm" else (0,))]

# extra (E2, "if the CPU budget allows"): B with beta = 1 on gmm9w, seed 0
n = "gmm9w_B_b1"
cfgs[n] = make(n, "B", "gmm9w", "E2 extra: dataset gmm9w, arm B family with beta = 1, lam = 0", beta=1.0, lam=0.0)
runs.append(("E2x", n, 0))

OUT.mkdir(parents=True, exist_ok=True)
for n, c in cfgs.items():
    (OUT / f"{n}.json").write_text(json.dumps(c, indent=1) + "\n")
(ROOT / "runs/ext").mkdir(parents=True, exist_ok=True)
with open(ROOT / "runs/ext/runs.tsv", "w") as f:
    f.write("experiment\tconfig\tseed\tout_dir\n")
    for e, n, s in runs:
        f.write(f"{e}\tconfigs/ext/{n}.json\t{s}\truns/ext/{n}_s{s}\n")
(ROOT / "runs/ext/median_heuristic.json").write_text(json.dumps(med, indent=1) + "\n")
print(len(cfgs), "configs,", len(runs), "runs;", {e: sum(r[0] == e for r in runs) for e in ["E1a", "E1b", "E2", "E4", "E2x"]})
print(json.dumps(med))
