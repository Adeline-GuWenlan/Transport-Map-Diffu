"""X4: sampler-seed spread of FID-10k and a real-data FID floor (E3). Same protocol as ddmcifar.evaluate, other sampler seeds.

ddmcifar.evaluate names its outputs without the seed (<ckpt>_<weights>_nfe<k>.json), so seeds other than 0 would collide with the
protocol files. This script calls the same functions (load_model, sample, to_uint8, fid_of_uint8) and writes seed-tagged JSON to
cifar/runs/eval/seeds/<run>_<ckpt>_<weights>_nfe<k>_seed<s>.json. Skips files that exist (requeue-safe).

    python cifar/scripts/eval_seeds.py --runs cifar_A,cifar_B --ckpt snap_0030000 --nfe 2,4,10,100 --seeds 1,2
    python cifar/scripts/eval_seeds.py --real          # FID-10k of real images (test set; 3 disjoint train subsets) vs train stats
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ddmcifar.data import load_cifar, to_uint8
from ddmcifar.evaluate import fid_of_uint8
from ddmcifar.sampler import sample
from ddmcifar.train import load_model

ROOT = Path(__file__).resolve().parents[1]          # ddm2d/cifar
OUT = ROOT / "runs" / "eval" / "seeds"


def real_floor(device):
    out = ROOT / "runs" / "eval" / "real_floor.json"
    if out.exists():
        return
    recs = []
    xt, _ = load_cifar("test")
    recs.append({"set": "test[0:10000]", **fid_of_uint8(to_uint8(xt[:10_000]), device)})
    xtr, _ = load_cifar("train")
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(xtr), generator=g)
    for i in range(3):
        idx = perm[i * 10_000:(i + 1) * 10_000]
        recs.append({"set": f"train randperm(seed 0)[{i * 10000}:{(i + 1) * 10000}]", **fid_of_uint8(to_uint8(xtr[idx]), device)})
    for r in recs:
        print(json.dumps({"set": r["set"], "fid": r["fid"]}), flush=True)
    out.write_text(json.dumps(recs, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="")
    ap.add_argument("--ckpt", default="snap_0030000")
    ap.add_argument("--nfe", default="2,4,10,100")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--out", default=str(OUT), help="output dir (smoke tests: somewhere else)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.real:
        real_floor(a.device)
    labels = torch.arange(a.n) % 10
    for run in [r for r in a.runs.split(",") if r]:
        rdir = ROOT / "runs" / run
        path = rdir / "ckpt" / f"{a.ckpt}.pt"
        model, cfg, step = load_model(path, a.weights, a.device)
        bf16 = bool(cfg.get("bf16", True)) and torch.device(a.device).type == "cuda"
        for s in [int(v) for v in a.seeds.split(",")]:
            for k in [int(v) for v in a.nfe.split(",")]:
                js = out / f"{run}_{a.ckpt}_{a.weights}_nfe{k}_seed{s}.json"
                if js.exists():
                    continue
                t0 = time.time()
                x = sample(model, labels, k, seed=s, device=a.device, batch=a.batch, bf16=bf16)
                rec = {"run": run, "ckpt": a.ckpt, "ckpt_path": str(path), "step": step, "weights": a.weights, "nfe": k, "seed": s,
                       "n": a.n, "arm": cfg["arm"], "beta": cfg["beta"], "lam": cfg["lam"], "rho": cfg["rho"], "bf16": bf16,
                       "sample_seconds": time.time() - t0, "pixel_mean": float(x.mean()), "pixel_std": float(x.std()),
                       "frac_outside_pm1": float((x.abs() > 1).float().mean())}
                rec.update(fid_of_uint8(to_uint8(x), a.device))
                js.write_text(json.dumps(rec, indent=1) + "\n")
                print(json.dumps({kk: rec[kk] for kk in ("run", "nfe", "seed", "fid", "sample_seconds")}), flush=True)
        del model
        torch.cuda.empty_cache() if a.device == "cuda" else None


if __name__ == "__main__":
    main()
