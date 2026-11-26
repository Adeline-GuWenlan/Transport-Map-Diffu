"""Health peek (not an evaluation): 10 x 10 grid (row = class) from a snapshot's EMA weights on CPU, NFE 10, sampler seed 0.
Usage (from ddm2d/): python cifar/scripts/snapshot_peek.py cifar_A snap_0010000 -> cifar/runs/health/<run>_<snap>_nfe10.png"""
import json
import sys
from pathlib import Path

import torch

from ddmcifar.evaluate import save_grid
from ddmcifar.sampler import sample
from ddmcifar.train import load_model

run, snap = sys.argv[1], sys.argv[2]
nfe = int(sys.argv[3]) if len(sys.argv) > 3 else 10
torch.set_num_threads(7)
model, cfg, step = load_model(Path("cifar/runs") / run / "ckpt" / f"{snap}.pt", "ema", "cpu")
labels = torch.arange(10).repeat_interleave(10)
x = sample(model, labels, nfe, seed=0, device="cpu", batch=100)
out = Path("cifar/runs/health")
out.mkdir(parents=True, exist_ok=True)
save_grid(x, out / f"{run}_{snap}_nfe{nfe}.png")
st = {"run": run, "snap": snap, "step": step, "nfe": nfe, "pixel_mean": float(x.mean()), "pixel_std": float(x.std()),
      "frac_outside_pm1": float((x.abs() > 1).float().mean()), "finite": bool(torch.isfinite(x).all())}
(out / f"{run}_{snap}_nfe{nfe}.json").write_text(json.dumps(st, indent=1) + "\n")
print(json.dumps(st))
