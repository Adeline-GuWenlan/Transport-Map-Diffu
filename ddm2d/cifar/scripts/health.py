"""Health summary of the E3 runs from cifar/runs/<run>/log.jsonl: steps, loss trend, grad norm, clip fraction, s/step, memory,
finiteness. Usage (from ddm2d/): python cifar/scripts/health.py [run ...]"""
import json
import sys
from pathlib import Path

RUNS = ["cifar_B", "cifar_C", "cifar_A", "cifar_B_b1", "cifar_B_pH", "cifar_C_dino"]


def summary(run):
    p = Path("cifar/runs") / run / "log.jsonl"
    if not p.exists():
        return {"run": run, "status": "no log yet"}
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if not recs:
        return {"run": run, "status": "empty log"}
    last = recs[-1]
    sps = [r["s_per_step"] for r in recs[2:]] or [recs[-1]["s_per_step"]]
    at = lambda s: next((r for r in recs if r["step"] >= s), None)
    return {"run": run, "step": last["step"], "all_finite": all(r["finite"] for r in recs),
            "loss@100": (at(100) or {}).get("loss"), "loss@1000": (at(1000) or {}).get("loss"), "loss_last": last["loss"],
            "conf_last": last.get("conf"), "inter_last": last.get("inter"), "gnorm_last": last["gnorm_mean"],
            "frac_clipped_last": last["frac_clipped"], "median_s_per_step": sorted(sps)[len(sps) // 2],
            "max_mem_gib": last.get("max_mem_gib"), "feat_loss_last": last.get("feat_loss")}


if __name__ == "__main__":
    for r in sys.argv[1:] or RUNS:
        print(json.dumps(summary(r)))
