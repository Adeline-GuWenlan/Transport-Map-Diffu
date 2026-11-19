"""Health check of running paper-scale runs (S2 step 4).

Usage: PYTHONPATH=src python scripts/health_check.py --runs runs/paper --out runs/paper/health/<name>.json [--bench 0.114]

Per run dir (<dataset>_<arm>_s<seed>): log lines, last step, all windows finite, s/step from log wall times (first window excluded),
ratio to the benchmark, and, if present, that ckpt/latest.pt loads into a fresh Trainer (model, EMA, optimizer, RNG, data order) and that
the EMA weights and the latest EMA snapshot give finite outputs on a fixed probe batch.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from ddm2d import train as T


def probe(model, arm):
    g = torch.Generator().manual_seed(0)
    xt = torch.randn(16, 2, generator=g)
    t = torch.rand(16, generator=g) * 0.98 + 0.01
    with torch.no_grad():
        y = model(t, xt) if arm == "A" else model(t, xt, torch.randn(16, 4, 2, generator=g))
    return bool(torch.isfinite(y).all())


def check(run: Path, bench: float) -> dict:
    r = {"run": run.name}
    logf = run / "log.jsonl"
    recs = [json.loads(l) for l in open(logf)] if logf.exists() else []
    r["log_lines"] = len(recs)
    r["last_step"] = recs[-1]["step"] if recs else 0
    r["all_finite"] = all(x["finite"] for x in recs) if recs else None
    if recs:
        r["last_loss"] = recs[-1]["loss"]
        r["gnorm_max_overall"] = max(x["gnorm_max"] for x in recs)
    if len(recs) >= 3:
        a, b = recs[1], recs[-1]
        r["s_per_step"] = round((b["wall_s"] - a["wall_s"]) / (b["step"] - a["step"]), 4)
        r["ratio_to_bench"] = round(r["s_per_step"] / bench, 3)
        r["eta_h_100k"] = round((100_000 - b["step"]) * r["s_per_step"] / 3600, 2)
    ck = run / "ckpt" / "latest.pt"
    if ck.exists():
        t0 = time.time()
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        tr = T.Trainer(sd["cfg"], sd["seed"])
        tr.load_state_dict(sd)                       # asserts cfg, seed, data fingerprint
        r["ckpt_step"] = tr.step
        r["ckpt_loads"] = True
        ema = T.build_model(sd["cfg"]["arm"])
        ema.load_state_dict(sd["ema"])
        ema.eval()
        r["ckpt_ema_probe_finite"] = probe(ema, sd["cfg"]["arm"])
        r["ckpt_load_s"] = round(time.time() - t0, 1)
        del sd, tr
    snaps = sorted((run / "ckpt").glob("ema_*.pt")) if (run / "ckpt").exists() else []
    if snaps:
        step = int(snaps[-1].stem.split("_")[1])
        m = T.load_ema_model(run, step=step)
        r["ema_snapshot"] = snaps[-1].name
        r["ema_snapshot_probe_finite"] = probe(m, "A" if not m.stochastic else "C")
    r["done"] = (run / "DONE.json").exists()
    r["failed"] = (run / "FAILED.txt").exists()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/paper")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", type=float, default=0.114, help="s/step expected (docs/benchmark.md, 6 x 1 on cs601)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    runs = sorted(p for p in Path(a.runs).iterdir() if p.is_dir() and "_s" in p.name and (p / "config.json").exists())
    res = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "bench_s_per_step": a.bench, "runs": [check(p, a.bench) for p in runs]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    keys = ["run", "last_step", "all_finite", "last_loss", "s_per_step", "ratio_to_bench", "eta_h_100k", "ckpt_step", "ckpt_loads",
            "ckpt_ema_probe_finite", "ema_snapshot", "ema_snapshot_probe_finite"]
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for r in res["runs"]:
        print("| " + " | ".join(str(round(r[k], 4) if isinstance(r.get(k), float) else r.get(k, "")) for k in keys) + " |")


if __name__ == "__main__":
    main()
