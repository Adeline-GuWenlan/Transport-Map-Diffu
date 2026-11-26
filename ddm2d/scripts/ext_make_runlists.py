"""ddm2dx X1: split runs/ext/runs.tsv (E1 first, then E2, then E4 and the E2 extra) into run lists of 6 for scripts/train_job.sbatch,
and write runs/ext/PLAN.md. Usage (from ddm2d/): python scripts/ext_make_runlists.py [--s-per-step-json runs/ext/smoke/summary.json]"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K = 6

ap = argparse.ArgumentParser()
ap.add_argument("--s-per-step-json", default=str(ROOT / "runs/ext/smoke/summary.json"))
a = ap.parse_args()
rows = [l.split("\t") for l in (ROOT / "runs/ext/runs.tsv").read_text().strip().splitlines()[1:]]
order = {"E1a": 0, "E1b": 1, "E2": 2, "E4": 3, "E2x": 4}
rows.sort(key=lambda r: order[r[0]])          # stable: keeps the generation order inside an experiment
lists = [rows[i:i + K] for i in range(0, len(rows), K)]
rl = ROOT / "runs/ext/runlists"
rl.mkdir(parents=True, exist_ok=True)
for i, chunk in enumerate(lists, 1):
    (rl / f"list{i:02d}.txt").write_text("# config out_dir seed (ddm2dx X1; scripts/ext_make_runlists.py)\n"
                                         + "".join(f"{c} {o} {s}\n" for _, c, s, o in [(r[0], r[1], r[2], r[3].strip()) for r in chunk]))

sps = {}
p = Path(a.s_per_step_json)
if p.exists():
    summ = json.loads(p.read_text())
    sps = {n: s.get("s_per_step") for n, s in summ.items() if not n.startswith("_") and not n.startswith("long_")}

cfgs = {r[1]: json.loads((ROOT / r[1]).read_text()) for r in rows}
L = ["# runs/ext PLAN (ddm2dx X1)", "",
     "Spec: `/scratch/wg2381/claude_jobs/ddm2dx/prompts/_task.md` (E1, E2, E4). Every config is `configs/paper/gmm_{A,B,C}.json` with only",
     "dataset / β / λ / ρ / ρ-parameter changed (`tests/test_ext_configs.py`). 100k steps, batch 128, m = 32 (A: m = 1). Run seed = training",
     "seed (init, data order, t, noise, ξ); data seed 0 for every run. Evaluation (X3) uses the ddm2d protocol for every run.",
     "",
     "ρ parameters: pseudo-Huber `rho_c` = c; IMQ `rho_c` = c (k = (‖d‖² + c)^(-1/2), main.tex Eq. imq_k); RBF `rho_c` = σ² = M² (γ = 1);",
     "Laplace `rho_c` = σ = M (γ = 1). M², M: median of ‖x − x'‖², ‖x − x'‖ over 100 batches of 512 training points, seed 0",
     "(`losses.median_sq_dist`, main.tex line 1747); values in `median_heuristic.json`. **Caveat [E]:** on the paper GMM the median falls",
     "in the gap between within-mode and cross-mode distances, so it moves with the batch seed (M² = 12.5 for seed 0; 9.8 / 13.6 / 8.1 for",
     "seeds 1–3). On `gmm9w` it is stable (34.6 ± 0.05).",
     "",
     "s/step: smoke measurement (`smoke/summary.json`, 1 thread, 7 processes side by side on one node; steps 100→400). Smoke is not a",
     "result. CPU-h estimate per run = 100k × s/step / 3600 × 1.3 (the ddm2d paper runs were 1.0–1.6× slower in packed jobs than alone).",
     "",
     "| # | exp | run dir | dataset | arm | β | λ | ρ | ρ param | seed | list | smoke s/step | est. CPU-h |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
tot = 0.0
idx = 0
for li, chunk in enumerate(lists, 1):
    for r in chunk:
        e, cp, s, o = r[0], r[1], r[2], r[3].strip()
        c = cfgs[cp]
        n = Path(cp).stem
        v = sps.get(n)
        est = 100_000 * v / 3600 * 1.3 if v else None
        tot += est or 0
        rho = c.get("rho", "l2")
        par = f"{c['rho_c']:.4g}" if "rho_c" in c else "–"
        beta = c["beta"] if (c["arm"] != "A" and rho in ("l2", "l1")) else "–"
        lam = c["lam"] if c["arm"] != "A" else "–"
        idx += 1
        L.append(f"| {idx} | {e} | `{o}` | {c['dataset']} | {c['arm']} | {beta} | {lam} | {rho if c['arm'] != 'A' else '– (Eq. 6)'} | "
                 f"{par} | {s} | {li:02d} | {f'{v:.3f}' if v else '?'} | {f'{est:.1f}' if est else '?'} |")
njobs = len(lists)
L += ["", f"Runs: {len(rows)}; run lists: {njobs} (`runlists/list01..{njobs:02d}.txt`, {K} runs each except the last).",
      f"Sum of per-run estimates: **{tot:.0f} CPU-h**. But a `train_job.sbatch` job holds 6 CPUs until its slowest run ends, so the",
      f"charge is ≈ 6 × (slowest run) per job: {njobs} jobs × 6 CPUs × 4.5–5.8 h ≈ {njobs * 6 * 4.5:.0f}–{njobs * 6 * 5.8:.0f} CPU-h",
      "(budget ≤ 400 CPU-h of 2D training for the chain).",
      "",
      "Not launched: the second E2 extra (C on `gmm9w` with the best E1b β). It needs the E1b evaluation (X3). Budget left for it is in",
      "`JOBS.md`.",
      ]
(ROOT / "runs/ext/PLAN.md").write_text("\n".join(L) + "\n")
print(njobs, "lists;", len(rows), "runs; est sum", round(tot))
