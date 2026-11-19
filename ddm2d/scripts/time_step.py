"""Time paper-config training steps (batch 128, m = 32) per arm and thread count. One fresh process per (arm, threads).

Usage: python scripts/time_step.py --out docs/timing_s1.json [--threads 1 2 4 8] [--arms A B C] [--steps 40]
Child mode (internal): python scripts/time_step.py --child ARM THREADS STEPS
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def child(arm, threads, steps):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    from ddm2d.train import Trainer
    cfg = json.loads((ROOT / "configs" / "paper" / f"gmm_{arm}.json").read_text())
    tr = Trainer(cfg, seed=0)
    for _ in range(5):
        tr.train_step()
    ts = []
    for _ in range(steps):
        t0 = time.perf_counter()
        tr.train_step()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    import resource
    print(json.dumps({"arm": arm, "threads": threads, "steps_timed": steps, "median_s": ts[len(ts) // 2], "mean_s": sum(ts) / len(ts),
                      "p10_s": ts[len(ts) // 10], "p90_s": ts[(9 * len(ts)) // 10],
                      "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "torch_threads": torch.get_num_threads()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=3)
    ap.add_argument("--out")
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--steps", type=int, default=40)
    a = ap.parse_args()
    if a.child:
        return child(a.child[0], int(a.child[1]), int(a.child[2]))
    res = []
    for arm in a.arms:
        for th in a.threads:
            env = dict(os.environ, OMP_NUM_THREADS=str(th), MKL_NUM_THREADS=str(th))
            p = subprocess.run([sys.executable, __file__, "--child", arm, str(th), str(a.steps)], capture_output=True, text=True, env=env,
                               timeout=1800)
            r = json.loads(p.stdout.strip().splitlines()[-1])
            r["est_h_100k"] = r["median_s"] * 1e5 / 3600
            r["est_cpu_h_100k"] = r["est_h_100k"] * th
            print(json.dumps(r), flush=True)
            res.append(r)
    meta = {"host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"), "affinity_cpus": len(os.sched_getaffinity(0)),
            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "note": "one process at a time, nothing else running in the allocation; "
            "paper config gmm_<arm>.json (batch 128, m 32 for B/C); 5 warm-up steps then timed steps"}
    if a.out:
        Path(a.out).write_text(json.dumps({"meta": meta, "results": res}, indent=1) + "\n")


if __name__ == "__main__":
    main()
