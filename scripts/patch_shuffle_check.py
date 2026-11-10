"""How much does the pixel energy distance care about dependence between patches?

For each class c, build "patch-shuffled" images: every 4x4 (or p x p) patch position is filled from an independently
drawn train image of class c. This keeps every patch's marginal distribution exactly and destroys the dependence
between patches (image structure). Also pixel-shuffled (p = 1). Compare their held-out D1 with the baselines.

    python scripts/patch_shuffle_check.py --dataset fashion_mnist --out runs/2026-11-06_loss_blindspot/fashion_mnist.json
"""
import argparse
import json
import os

import numpy as np
import torch

from tmap.data import ImageArray
from tmap.energy_score import per_class_energy_distance
from tmap.evaluate import baselines, reference_set


def patch_shuffle(x, p, rng):
    """x: (n, C, H, W) of one class -> same shape; each p x p block from an independent random image."""
    n, C, H, W = x.shape
    out = torch.empty_like(x)
    for i in range(0, H, p):
        for j in range(0, W, p):
            src = torch.from_numpy(rng.integers(0, n, size=n))
            out[:, :, i:i + p, j:j + p] = x[src, :, i:i + p, j:j + p]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    train = ImageArray(f"data/{a.dataset}", "train", 32)
    test = ImageArray(f"data/{a.dataset}", "test", 32)
    ref_x, ref_l = reference_set(test, a.n)
    yy = {}
    res = {"baselines": {k: v["mean"] for k, v in baselines(train, ref_x, ref_l, a.n, yy_cache=yy).items()}}
    rng = np.random.default_rng(0)
    for p in (16, 8, 4, 2, 1):
        xs, ls = [], []
        for c in train.classes:
            pool = train.get(train.by_class[c][:5000])[0]
            xs.append(patch_shuffle(pool, p, rng)[:a.n])
            ls.append(torch.full((a.n,), c, dtype=torch.long))
        r = per_class_energy_distance(torch.cat(xs), torch.cat(ls), ref_x, ref_l, yy_cache=yy)
        res[f"patch_shuffled_{p}"] = r["mean"]
        print(f"{a.dataset}: patch-shuffled p={p}: D1 {r['mean']:.4f}", flush=True)
    print(json.dumps(res["baselines"]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
