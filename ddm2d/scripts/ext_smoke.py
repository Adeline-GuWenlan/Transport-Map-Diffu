"""ddm2dx X1 smoke (never used in results): every configs/ext config trains 200 steps, then resumes to 400; a few longer runs check the
gradient norm for beta < 1 and L1 against the paper C config. 1 thread per process, P processes in parallel.
Usage (from ddm2d/): PYTHONPATH=src python scripts/ext_smoke.py [--procs 7] -> runs/ext/smoke/{<name>/, summary.json, SMOKE.md}"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SM = ROOT / "runs/ext/smoke"
PY = sys.executable
LONG = {"gmm_B_b0.1": 3000, "gmm_C_L1b0.5": 3000, "gmm_C_L1b1": 3000, "gmm_B_L1b1": 3000, "gmm_C_b0.5": 3000}
REF = ROOT / "configs/paper/gmm_C.json"


def train(cfg, out, max_steps):
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    r = subprocess.run([PY, "-m", "ddm2d.train", "--config", str(cfg), "--out", str(out), "--seed", "0", "--threads", "1",
                        "--max-steps", str(max_steps)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=3600)
    return r.returncode, r.stdout + r.stderr, time.time() - t0


def job(item):
    name, cfg, steps = item
    out = SM / name
    if out.exists():
        subprocess.run(["rm", "-rf", str(out)])
    if steps:                       # long gradient-norm run, no resume
        rc, log, wall = train(cfg, out, steps)
        return name, {"rc": rc, "wall": wall, "resumed": None, "tail": log[-300:]}
    rc1, log1, w1 = train(cfg, out, 200)
    rc2, log2, w2 = train(cfg, out, 400)
    return name, {"rc": rc1 or rc2, "wall": w1 + w2, "resumed": "resumed from step 200" in log2, "tail": (log1 + log2)[-300:]}


def summarize(name, res):
    out = SM / name
    recs = [json.loads(l) for l in open(out / "log.jsonl")] if (out / "log.jsonl").exists() else []
    s = dict(res)
    s.pop("tail")
    if recs:
        s.update(steps=recs[-1]["step"], finite=all(r["finite"] for r in recs), loss_first=recs[0]["loss"], loss_last=recs[-1]["loss"],
                 gnorm_mean_first=recs[0]["gnorm_mean"], gnorm_mean_last=recs[-1]["gnorm_mean"],
                 gnorm_max_all=max(r["gnorm_max"] for r in recs),
                 gnorm_max_last1000=max(r["gnorm_max"] for r in recs if r["step"] > recs[-1]["step"] - 1000),
                 # s/step from the segment that ran steps 200 -> 400 (or the whole run), excluding startup
                 s_per_step=(recs[-1]["wall_s"] - recs[1]["wall_s"]) / (recs[-1]["step"] - recs[1]["step"]) if len(recs) > 2 else None)
        s["all_finite_numbers"] = all(math.isfinite(r[k]) for r in recs for k in ("loss", "gnorm_mean", "gnorm_max"))
    else:
        s["error"] = res["tail"]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=7)
    a = ap.parse_args()
    SM.mkdir(parents=True, exist_ok=True)
    items = [(f"long_{n}", ROOT / f"configs/ext/{n}.json", s) for n, s in LONG.items()] + [("long_paper_gmm_C", REF, 3000)]
    items += [(p.stem, p, 0) for p in sorted((ROOT / "configs/ext").glob("*.json"))]
    t0 = time.time()
    with ThreadPoolExecutor(a.procs) as ex:
        results = dict(ex.map(job, items))
    summ = {n: summarize(n, r) for n, r in results.items()}
    summ["_meta"] = {"procs": a.procs, "wall_s": round(time.time() - t0, 1), "host": os.uname().nodename,
                     "cpus": len(os.sched_getaffinity(0)), "date": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (SM / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    bad = [n for n, s in summ.items() if n != "_meta" and (s["rc"] != 0 or not s.get("finite") or not s.get("all_finite_numbers")
                                                         or s["resumed"] is False)]
    print("BAD:", bad)
    print(json.dumps(summ["_meta"]))


if __name__ == "__main__":
    main()
