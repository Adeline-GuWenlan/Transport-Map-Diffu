"""Packed-process benchmark: K concurrent training processes x T threads each (K*T = 8 = the allocation), paper config, arm mix.

Usage: python scripts/bench_packed.py --out docs/benchmark_packed_s1.json [--steps 150]
Each process: scripts/time_step.py --child <arm> T steps (5 warm-up steps, then timed steps). Arms cycle C, A, B, ...
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--layouts", nargs="+", default=["8x1", "4x2", "2x4", "1x8"])
    a = ap.parse_args()
    res = []
    for lay in a.layouts:
        k, t = map(int, lay.split("x"))
        arms = ["C", "A", "B"] * 3
        procs = []
        t0 = time.time()
        for i in range(k):
            env = dict(os.environ, OMP_NUM_THREADS=str(t), MKL_NUM_THREADS=str(t))
            procs.append((arms[i], subprocess.Popen([sys.executable, str(HERE / "time_step.py"), "--child", arms[i], str(t), str(a.steps)],
                                                    stdout=subprocess.PIPE, text=True, env=env)))
        rows = []
        for arm, p in procs:
            out, _ = p.communicate(timeout=3600)
            rows.append(json.loads(out.strip().splitlines()[-1]))
        wall = time.time() - t0
        med = sorted(r["median_s"] for r in rows)
        worst = max(r["median_s"] for r in rows)
        r = {"layout": lay, "procs": k, "threads_each": t, "per_proc_median_s": [round(x["median_s"], 4) for x in rows],
             "arms": [x["arm"] for x in rows], "worst_median_s": worst, "wall_s": round(wall, 1),
             "h_per_100k_run_worst": worst * 1e5 / 3600, "runs_per_8cpu_hour": k * 3600 / (worst * 1e5),
             "cpu_h_per_run": 8 * worst * 1e5 / 3600 / k}
        print(json.dumps(r), flush=True)
        res.append(r)
    Path(a.out).write_text(json.dumps({"meta": {"host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"),
                                                "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "steps_timed": a.steps},
                                       "results": res}, indent=1) + "\n")


if __name__ == "__main__":
    main()
