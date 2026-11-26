"""GPU bench for the E3 plan (one L40S job, < 1 h): s/step and peak memory of the full-size training step for every distinct cost
class (A; B / C at m = 4, batch 128, with and without grad_chunk; B_pH; C_dino), then the wall time of one FID-10k at NFE 10 and at
NFE 100 (untrained B weights: the FID value is meaningless, only the time counts).

    python cifar/scripts/bench.py --out cifar/runs/bench_<jobid>   (from ddm2d/, PYTHONPATH=src:cifar/src)
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
from ddmcifar.train import Trainer

CASES = [("cifar_A", 0), ("cifar_B", 0), ("cifar_B", 256), ("cifar_C", 0), ("cifar_B_pH", 0),
         ("cifar_C_dino", 0), ("cifar_C_dino", 256), ("cifar_C_dino", 128)]


def bench_case(name, chunk, data, n_warm, n_time):
    cfg = json.loads(Path(f"cifar/configs/{name}.json").read_text())
    cfg["grad_chunk"] = chunk
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rec = {"run": name, "grad_chunk": chunk}
    tr = None
    try:
        tr = Trainer(cfg, None, data)
        tr.init_feature_scale()
        for _ in range(n_warm):
            tr.train_step()
        torch.cuda.synchronize()
        t0 = time.time()
        losses = [tr.train_step() for _ in range(n_time)]
        torch.cuda.synchronize()
        rec.update(s_per_step=(time.time() - t0) / n_time, peak_mem_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30, losses=losses, status="ok",
                   feat=tr.feat)
    except torch.OutOfMemoryError as e:
        rec.update(status="OOM", error=str(e)[:300], peak_mem_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    model = tr.model if (tr is not None and rec["status"] == "ok") else None
    del tr
    torch.cuda.empty_cache()
    return rec, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "train": [], "fid": []}
    data = load_cifar("train")
    keep = None
    for name, chunk in CASES:
        rec, model = bench_case(name, chunk, data, a.warm, a.steps)
        print(json.dumps({k: v for k, v in rec.items() if k != "losses"}), flush=True)
        res["train"].append(rec)
        if name == "cifar_B" and model is not None and keep is None:
            keep = model.eval()
        else:
            del model
        (out / "bench.json").write_text(json.dumps(res, indent=1) + "\n")
    labels = torch.arange(10_000) % 10
    for nfe in (10, 100):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        x = sample(keep, labels, nfe, seed=0, device="cuda", batch=1000, bf16=True)
        t1 = time.time()
        f = fid_of_uint8(to_uint8(x), "cuda")
        rec = {"nfe": nfe, "n": 10_000, "sample_s": t1 - t0, "fid_call_s": time.time() - t1, "total_s": time.time() - t0,
               "fid_untrained_meaningless": f["fid"], "peak_mem_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        print(json.dumps(rec), flush=True)
        res["fid"].append(rec)
        (out / "bench.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
