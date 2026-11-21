"""Size and cost of the UNet (configs/expA_cifar10_unet_pixel.yaml) next to DiT-B/2 (configs/expA_cifar10_dit-b2_pixel.yaml),
at 3x32x32, on CPU (unet chain S1; docs/experiments_unet.md "Size and cost").

    python scripts/cost_unet.py --out runs/cost_unet_S1.json [--threads 7] [--steps 2] [--dinov2]

- params; forward GMACs/sample (torch.utils.flop_counter, MACs = FLOPs / 2);
- CPU s/step at batch 64: the real tmap.train.forward_backward (fp32, pixel loss) + clip + AdamW step, with the exact
  two-pass gradient at chunk 16 (one pass of DiT-B/2 at batch 64 in fp32 needs ~14 GiB and is OOM-killed in 16 GB);
- saved-activation memory per sample under bf16 autocast: bytes of the tensors packed for backward
  (torch.autograd.graph.saved_tensors_hooks, deduplicated by storage) for batch b1 and b2, slope (b2 - b1);
  the slope removes per-call constants such as bf16 copies of the weights;
- grad_chunk for a 48 GB L40S: largest power of two with chunk * activations/sample <= 30 GiB, capped at 512.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from tmap.model import build_model, count_params
from tmap.train import forward_backward, load_config

CFGS = {"unet": "configs/expA_cifar10_unet_pixel.yaml", "dit-b2": "configs/expA_cifar10_dit-b2_pixel.yaml"}
LIMIT_GIB, CAP = 30.0, 512


def gmacs(model):
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(torch.randn(1, 3, 32, 32), torch.tensor([0]))
    return fc.get_total_flops() / 2 / 1e9


def saved_bytes(fn, b):
    """Bytes of distinct storages saved for backward by fn(b) under bf16 autocast (CPU)."""
    seen = {}

    def pack(t):
        try:
            s = t.untyped_storage()
            seen[s.data_ptr()] = max(seen.get(s.data_ptr(), 0), s.nbytes())
        except (RuntimeError, NotImplementedError):
            pass
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = fn(b)
    del out
    return sum(seen.values())


def act_per_sample(fn, b1=2, b2=6):
    m1, m2 = saved_bytes(fn, b1), saved_bytes(fn, b2)
    return (m2 - m1) / (b2 - b1), m1, m2


def chunk_for(mib_per_sample):
    c = 1
    while c * 2 <= CAP and c * 2 * mib_per_sample / 1024 <= LIMIT_GIB:
        c *= 2
    return c


def cpu_step_time(model, steps, batch=64, chunk=16):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(0)
    xl = torch.arange(8).repeat_interleave(batch // 8)
    y = torch.rand(batch, 3, 32, 32, generator=g) * 2 - 1
    lc = {"beta": 1.0, "interaction_weight": 0.5}
    ts = []
    for i in range(steps + 1):                      # first step is warm-up
        z = torch.randn(batch, 3, 32, 32, generator=g)
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        forward_backward(model, None, z, xl, y, xl, lc, 0.0, False, chunk)
        torch.nn.utils.clip_grad_norm_(params, 2.0)
        opt.step()
        if i:
            ts.append(time.time() - t0)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=7)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--dinov2", action="store_true", help="also DINOv2-S/14 at 224 px (exp B's feature encoder)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    res = {"threads": a.threads, "torch": torch.__version__, "limit_gib": LIMIT_GIB, "cap": CAP,
           "cpu": open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": "),
           "slurm_job": os.environ.get("SLURM_JOB_ID")}
    for name, path in CFGS.items():
        cfg = load_config(path)
        torch.manual_seed(0)
        m = build_model(cfg)
        with torch.no_grad():                       # non-zero output layer: realistic activations and timing
            for p in m.parameters():
                p.add_(1e-3 * torch.randn_like(p))
        r = {"config": path, "params_M": count_params(m) / 1e6, "gmacs_per_sample": gmacs(m)}
        x0 = torch.randn(8, 3, 32, 32)

        def fn(b, m=m):
            return m(x0[:b].clone().requires_grad_(False), torch.arange(b) % 10)
        per, m1, m2 = act_per_sample(fn)
        r["act_mib_per_sample_bf16"] = per / 2 ** 20
        r["act_raw_mib_b2_b6"] = [m1 / 2 ** 20, m2 / 2 ** 20]
        ts = cpu_step_time(m, a.steps)
        r["cpu_s_per_step_b64"], r["cpu_grad_chunk"] = ts, 16
        r["cpu_s_per_step_b64_mean"] = sum(ts) / len(ts)
        res[name] = r
        print(name, json.dumps(r), flush=True)
        del m, fn
        import gc; gc.collect()
    if a.dinov2:
        from tmap.features import build_feature_encoder
        enc = build_feature_encoder({"encoder": "dinov2_vits14", "img_size": 224}, 3)
        x0 = torch.randn(8, 3, 32, 32)
        per, m1, m2 = act_per_sample(lambda b: enc(x0[:b].clone().requires_grad_(True)))
        res["dinov2_vits14_224"] = {"act_mib_per_sample_bf16": per / 2 ** 20, "act_raw_mib_b2_b6": [m1 / 2 ** 20,
                                                                                                    m2 / 2 ** 20]}
        print("dinov2", json.dumps(res["dinov2_vits14_224"]), flush=True)
    for name in CFGS:
        a_mib = res[name]["act_mib_per_sample_bf16"]
        res[name]["grad_chunk_A"] = chunk_for(a_mib)
        if "dinov2_vits14_224" in res:
            b_mib = a_mib + res["dinov2_vits14_224"]["act_mib_per_sample_bf16"]
            res[name]["act_mib_per_sample_bf16_B"] = b_mib
            res[name]["grad_chunk_B"] = chunk_for(b_mib)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if "chunk" in kk or "mib_per" in kk}
                      for k, v in res.items() if isinstance(v, dict)}))


if __name__ == "__main__":
    main()
