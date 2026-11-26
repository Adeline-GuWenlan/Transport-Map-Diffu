"""ddm2dx: health of the runs/ext training runs from their log.jsonl (finite losses, speed, gradient norms, progress).
Usage (from ddm2d/): python scripts/ext_health.py [--md runs/ext/HEALTH_table.md]   (no torch needed)"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--md", default=None)
a = ap.parse_args()
rows = [l.split("\t") for l in (ROOT / "runs/ext/runs.tsv").read_text().strip().splitlines()[1:]]
out = ["| run | state | step | s/step (last 1k) | ETA h (at that speed) | loss (last window) | gnorm mean / max (last 1k) | finite | exit / FAILED |",
       "|---|---|---|---|---|---|---|---|---|"]
bad = []
for e, cfg, seed, od in rows:
    d = ROOT / od.strip()
    name = d.name
    if not (d / "log.jsonl").exists():
        out.append(f"| `{name}` | not started | – | – | – | – | – | – | – |")
        continue
    recs = [json.loads(l) for l in open(d / "log.jsonl")]
    if not recs:
        out.append(f"| `{name}` | started, no log yet | – | – | – | – | – | – | – |")
        continue
    last = recs[-1]
    tail = [r for r in recs if r["step"] > last["step"] - 1000]
    sps = None
    if len(tail) >= 2 and tail[-1]["wall_s"] > tail[0]["wall_s"]:
        sps = (tail[-1]["wall_s"] - tail[0]["wall_s"]) / (tail[-1]["step"] - tail[0]["step"])
    steps = json.loads((ROOT / cfg).read_text())["steps"]
    eta = (steps - last["step"]) * sps / 3600 if sps else None
    fin = all(r["finite"] and math.isfinite(r["loss"]) and math.isfinite(r["gnorm_mean"]) for r in recs)
    state = "DONE" if (d / "DONE.json").exists() else "running/paused"
    failed = (d / "FAILED.txt").read_text().strip().replace("\n", "; ") if (d / "FAILED.txt").exists() else ""
    if not fin or failed:
        bad.append(name)
    gm = sum(r["gnorm_mean"] for r in tail) / len(tail)
    gx = max(r["gnorm_max"] for r in tail)
    out.append(f"| `{name}` | {state} | {last['step']} | {f'{sps:.3f}' if sps else '–'} | {f'{eta:.1f}' if eta is not None else '–'} | "
               f"{last['loss']:.4f} | {gm:.2f} / {gx:.1f} | {fin} | {failed or '–'} |")
txt = "\n".join(out) + "\n"
print(txt)
print("NOT FINITE / FAILED:", bad)
if a.md:
    Path(a.md).write_text(txt)
