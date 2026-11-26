"""Evaluation for the CIFAR-10 trial (identical protocol for every run; prompts/_task.md E3).

FID-10k: clean-fid `compute_fid(gen=...)` in `legacy_tensorflow` mode against the CIFAR-10 *train* statistics (cifar10, res 32,
split train), as transport_map's tmap.evaluate.fid_cleanfid does. The stats file is clean-fid's bundled
`cifar10_legacy_tensorflow_train_32.npz` (the same file as transport_map/data/fid/stats/; sha256 of both are recorded in the output);
the Inception weights are copied from transport_map/data/fid/inception/ to /tmp, where clean-fid looks for them.
Samples: n images with class-balanced labels c_i = i mod 10 (1000 per class for n = 10k), sampler seed `seed` (default 0, the same for
every run), EMA weights unless --weights raw. The 10 x 10 grid (one row per class) uses the first 100 of these samples.

    python -m ddmcifar.evaluate --run cifar/runs/cifar_B --ckpt final --nfe 2,4,8,10,15,30,100 --grid-nfe 10,100 --spread
    --ckpt final | snap_0010000 | ... (ckpt/<name>.pt; "final" = ckpt/latest.pt of a DONE run)
Outputs in <run>/eval/: <ckpt>_<weights>_nfe<k>.json, grid_<ckpt>_<weights>_nfe<k>.png, xi_spread_<ckpt>_<weights>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_cifar, to_uint8
from .sampler import sample, xi_spread
from .train import load_model

INCEPTION_SRC = "/scratch/wg2381/transport_map/data/fid/inception/inception-2015-12-05.pt"
STATS_TMAP = "/scratch/wg2381/transport_map/data/fid/stats/cifar10_legacy_tensorflow_train_32.npz"
SPREAD_TS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fid_of_uint8(imgs: torch.Tensor, device="cuda", batch: int = 100) -> dict:
    """clean-fid legacy_tensorflow FID of uint8 images (N, 3, 32, 32) vs CIFAR-10 train stats."""
    import cleanfid
    from cleanfid import fid
    if not os.path.exists("/tmp/inception-2015-12-05.pt") and os.path.exists(INCEPTION_SRC):
        shutil.copy(INCEPTION_SRC, "/tmp/inception-2015-12-05.pt")
    n = len(imgs)
    assert n % batch == 0
    state = {"i": 0}

    def gen(_z):
        b = _z.shape[0]
        out = imgs[state["i"]:state["i"] + b]
        state["i"] += b
        return out

    v = fid.compute_fid(gen=gen, dataset_name="cifar10", dataset_res=32, num_gen=n, dataset_split="train",
                        mode="legacy_tensorflow", batch_size=batch, device=torch.device(device), use_dataparallel=False,
                        num_workers=0, verbose=False)
    stats = os.path.join(os.path.dirname(cleanfid.__file__), "stats", "cifar10_legacy_tensorflow_train_32.npz")
    return {"fid": float(v), "n": n, "mode": "legacy_tensorflow", "ref": "cifar10 train 32",
            "stats_sha256": sha256(stats), "stats_tmap_sha256": sha256(STATS_TMAP) if os.path.exists(STATS_TMAP) else None}


def save_grid(x: torch.Tensor, path, nrow: int = 10):
    from PIL import Image
    u = to_uint8(x).permute(0, 2, 3, 1).numpy()
    n, h, w, _ = u.shape
    rows = (n + nrow - 1) // nrow
    canvas = np.full((rows * (h + 2) + 2, nrow * (w + 2) + 2, 3), 255, np.uint8)
    for i in range(n):
        r, c = divmod(i, nrow)
        canvas[2 + r * (h + 2):2 + r * (h + 2) + h, 2 + c * (w + 2):2 + c * (w + 2) + w] = u[i]
    Image.fromarray(canvas).save(path)


def grid_order(x: torch.Tensor, labels: torch.Tensor, per_class: int = 10) -> torch.Tensor:
    """Rows = classes 0..9, the first `per_class` samples of each class."""
    return torch.cat([x[labels == c][:per_class] for c in range(10)])


def evaluate_ckpt(run: Path, ckpt: str, nfes, grid_nfes=(), n: int = 10_000, seed: int = 0, device="cuda", weights="ema",
                  spread: bool = False, batch: int = 1000, skip_done: bool = True) -> list:
    path = run / "ckpt" / ("latest.pt" if ckpt == "final" else f"{ckpt}.pt")
    model, cfg, step = load_model(path, weights, device)
    bf16 = bool(cfg.get("bf16", True)) and torch.device(device).type == "cuda"
    ev = run / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    labels = torch.arange(n) % 10
    res = []
    for k in sorted(set(nfes) | set(grid_nfes)):
        js = ev / f"{ckpt}_{weights}_nfe{k}.json"
        if skip_done and js.exists():
            res.append(json.loads(js.read_text()))
            continue
        t0 = time.time()
        x = sample(model, labels, k, seed=seed, device=device, batch=batch, bf16=bf16)
        t_sample = time.time() - t0
        rec = {"run": run.name, "ckpt": ckpt, "ckpt_path": str(path), "step": step, "weights": weights, "nfe": k, "seed": seed,
               "n": n, "arm": cfg["arm"], "beta": cfg["beta"], "lam": cfg["lam"], "rho": cfg["rho"], "bf16": bf16,
               "sample_seconds": t_sample, "pixel_mean": float(x.mean()), "pixel_std": float(x.std()),
               "frac_outside_pm1": float((x.abs() > 1).float().mean())}
        if k in grid_nfes:
            save_grid(grid_order(x, labels), ev / f"grid_{ckpt}_{weights}_nfe{k}.png")
        if k in nfes:
            t1 = time.time()
            rec.update(fid_of_uint8(to_uint8(x), device))
            rec["fid_seconds"] = time.time() - t1
        js.write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps({kk: rec[kk] for kk in ("run", "ckpt", "nfe", "fid", "sample_seconds") if kk in rec}), flush=True)
        res.append(rec)
    if spread:
        js = ev / f"xi_spread_{ckpt}_{weights}.json"
        if not (skip_done and js.exists()):
            xt, yt = load_cifar("test", n=None)
            gsel = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(xt), generator=gsel)[:256]
            sp = xi_spread(model, xt[idx], yt[idx], SPREAD_TS, n_draws=8, seed=seed, device=device, bf16=bf16)
            js.write_text(json.dumps({"run": run.name, "ckpt": ckpt, "step": step, "weights": weights, "n_images": 256,
                                      "n_draws": 8, "spread": sp}, indent=1) + "\n")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="final")
    ap.add_argument("--nfe", default="10")
    ap.add_argument("--grid-nfe", default="")
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    ap.add_argument("--spread", action="store_true")
    ap.add_argument("--batch", type=int, default=1000)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", len(os.sched_getaffinity(0)))))
    ints = lambda s: [int(v) for v in s.split(",") if v]
    evaluate_ckpt(Path(a.run), a.ckpt, ints(a.nfe), ints(a.grid_nfe), a.n, a.seed, a.device, a.weights, a.spread, a.batch)


if __name__ == "__main__":
    main()
