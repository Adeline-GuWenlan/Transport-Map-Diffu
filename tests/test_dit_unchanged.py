"""Golden values: the DiT training path must not change when the UNet backbone is added (unet chain S1).

A tiny DiT, built from configs/expA_cifar10_dit-b2_pixel.yaml with small overrides, trains 3 steps through
tmap.train.train() itself (sampler, z draws, chunked two-pass gradient, clipping, AdamW, EMA). The per-step losses
and checksums of the final raw and EMA parameters were recorded on main @ 8a15563, before any UNet edit
(2026-11-20, torch 2.14.1+cu126, CPU, 1 thread). Two cases: pixel loss only (A) and pixel + random-conv feature
term (the B code path, without downloading DINOv2).

Regenerate (only if the DiT path is changed on purpose):  python tests/test_dit_unchanged.py
"""
import json
import os
import sys

import torch

from tmap.train import load_config, train

CONFIG = "configs/expA_cifar10_dit-b2_pixel.yaml"
OVR = ["device=cpu", "num_threads=1", "seed=0",
       "model.depth=2", "model.hidden_size=32", "model.num_heads=2", "model.patch_size=8",
       "batch.classes_per_batch=2", "batch.n_per_class=4",
       "data.subset={classes: [0, 1, 2], n_per_class: 32}",
       "train.max_steps=3", "train.grad_chunk=3", "train.log_every=1", "train.ckpt_every=1000",
       "train.snapshot_steps=[]", "train.sample_every=1000", "eval.every=1000",
       "eval.energy_distance.n_per_class=8", "optim.warmup_steps=2"]
CASES = {
    "A_pixel": [],
    "B_feature": ["loss.feature={encoder: random_conv, weight: 1.0, scale: auto, scale_n_per_class: 8}"],
}

# Recorded on main @ 8a15563 before any UNet edit. Do not edit by hand.
GOLDEN = {
    "A_pixel": {"losses": [26.997547149658203, 24.627769470214844, 25.19733428955078], "grad_norms": [2.6244101524353027, 3.5740630626678467, 4.674740791320801], "step": 3, "model_sum": 262.4966898521235, "model_sq": 683.7431791220893, "model_wsum": 2.4311685560988123, "ema_sum": 262.4556743638588, "ema_sq": 683.738381655847, "ema_wsum": 2.4527327656353393, "delta_sq": 0.0004840352122179805},
    "B_feature": {"losses": [58.859352111816406, 52.20798110961914, 52.482181549072266], "grad_norms": [40.71659469604492, 25.383760452270508, 44.23339080810547], "step": 3, "model_sum": 262.476900168132, "model_sq": 683.7346126704236, "model_wsum": 2.4501958333549823, "ema_sum": 262.455669594865, "ema_sq": 683.7383805240049, "ema_wsum": 2.452735633860675, "delta_sq": 0.0004569805350178264},
}


def run_case(name, out_dir):
    cfg = load_config(CONFIG, OVR + CASES[name])
    train(cfg, str(out_dir), resume=False)
    recs = [json.loads(l) for l in open(os.path.join(out_dir, "metrics.jsonl"))]
    ck = torch.load(os.path.join(out_dir, "ckpt", "latest.pt"), weights_only=False)
    res = {"losses": [r["loss"] for r in recs], "grad_norms": [r["grad_norm"] for r in recs], "step": ck["step"]}
    for key in ("model", "ema"):
        sd = ck[key]
        res[f"{key}_sum"] = float(sum(v.double().sum() for v in sd.values()))
        res[f"{key}_sq"] = float(sum((v.double() ** 2).sum() for v in sd.values()))
        res[f"{key}_wsum"] = float(sum((v.double().flatten() * torch.linspace(-1, 1, v.numel(), dtype=torch.float64)
                                        ).sum() for v in sd.values()))
    res["delta_sq"] = float(sum(((ck["model"][k].double() - ck["ema"][k].double()) ** 2).sum() for k in ck["model"]))
    return res


def _close(a, b, rtol=1e-5, atol=1e-7):
    return abs(a - b) <= atol + rtol * abs(b)


def _check(name, tmp_path):
    g = GOLDEN[name]
    assert g is not None, "golden value not recorded"
    r = run_case(name, tmp_path / name)
    assert r["step"] == g["step"] == 3
    assert len(r["losses"]) == len(g["losses"]) == 3
    for a, b in zip(r["losses"] + r["grad_norms"], g["losses"] + g["grad_norms"]):
        assert _close(a, b), (name, r, g)
    for k in g:
        if k == "delta_sq":                    # sum (raw - EMA)^2: sensitive to the 3 updates themselves
            assert _close(r[k], g[k], rtol=1e-4, atol=1e-12), (name, k, r[k], g[k])
        elif k.endswith(("_sum", "_sq", "_wsum")):
            assert _close(r[k], g[k], rtol=1e-6, atol=1e-6), (name, k, r[k], g[k])


def test_dit_pixel_golden(tmp_path):
    _check("A_pixel", tmp_path)


def test_dit_feature_golden(tmp_path):
    _check("B_feature", tmp_path)


if __name__ == "__main__":
    import tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as d:
        print(json.dumps({n: run_case(n, os.path.join(d, n)) for n in CASES}, indent=1))
