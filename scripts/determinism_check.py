"""Do the DiT and the UNet runs see the same batches, flips and noise for the same seed? (unet chain S1)

    python scripts/determinism_check.py --out runs/determinism_unet_S1.json [--seed 0] [--steps 3]

Runs the real tmap.train.train() for `steps` steps on CPU from configs/expA_cifar10_{dit-b2,unet}_pixel.yaml with
small models and n_per_class cut down (the sampler, z draws and data are untouched by those overrides, apart from
the batch shape, which is the same for both backbones). It records, per step:
- the sampler's class choice, image indices and hflip mask (replayed from the sampler's numpy RNG state right before
  each real draw, then checked against the real draw: same images, same RNG state afterwards);
- checksums of the batch y, labels and of the training noise z as passed to tmap.train.forward_backward;
- a checksum of the in-loop fixed sample-grid noise (tmap.sample.fixed_noise, seed 1234).
Nothing in tmap.train is changed; the hooks wrap the real objects.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile

import numpy as np
import torch

import tmap.train as T
from tmap.data import ClassStratifiedSampler

CFGS = {"dit": "configs/expA_cifar10_dit-b2_pixel.yaml", "unet": "configs/expA_cifar10_unet_pixel.yaml"}
COMMON = ["device=cpu", "num_threads=4", "batch.n_per_class=4", "train.log_every=1", "train.ckpt_every=1000",
          "train.snapshot_steps=[]", "train.sample_every=1000", "eval.every=1000",
          "eval.energy_distance.n_per_class=4"]
SMALL = {"dit": ["model.depth=1", "model.hidden_size=32", "model.num_heads=2", "model.patch_size=8"],
         "unet": ["model.base_channels=16", "model.channel_mult=[1, 2]", "model.num_res_blocks=1",
                  "model.num_heads=2", "model.emb_dim=32"]}


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def run(backbone: str, seed: int, steps: int) -> dict:
    log = {"draws": [], "fb": [], "fixed_noise": None}

    class LoggingSampler(ClassStratifiedSampler):
        def next(self):
            rep = np.random.default_rng()
            rep.bit_generator.state = copy.deepcopy(self.rng.bit_generator.state)
            x, y = super().next()
            # replay the same calls as ClassStratifiedSampler.next on a copy of the pre-draw state
            cls = rep.choice(np.asarray(self.data.classes), self.k, replace=False)
            idx = np.concatenate([rep.choice(self.data.by_class[int(c)], self.n, replace=False) for c in cls])
            flip = rep.random(len(idx)) < 0.5 if self.hflip else np.zeros(len(idx), bool)
            xr, yr = self.data.get(idx)
            xr = torch.where(torch.from_numpy(flip)[:, None, None, None], xr.flip(-1), xr)
            assert torch.equal(xr, x) and torch.equal(yr, y), "replay does not reproduce the real draw"
            assert rep.bit_generator.state == self.rng.bit_generator.state, "replay consumed a different stream"
            log["draws"].append({"classes": cls.tolist(), "indices": idx.tolist(), "flip": flip.astype(int).tolist(),
                                 "y_sha": digest(x), "labels": y.tolist()})
            return x, y

    orig_fb, orig_fn = T.forward_backward, T.fixed_noise

    def fb(model, enc, z, xl, y, yl, *a, **k):
        log["fb"].append({"z_sha": digest(z), "z_sum": float(z.double().sum()), "z_sq": float((z.double() ** 2).sum()),
                          "z_shape": list(z.shape), "xl": xl.tolist(), "y_sha": digest(y)})
        return orig_fb(model, enc, z, xl, y, yl, *a, **k)

    def fn(*a, **k):
        z, c = orig_fn(*a, **k)
        log["fixed_noise"] = {"z_sha": digest(z), "c": c.tolist(), "shape": list(z.shape)}
        return z, c

    T.ClassStratifiedSampler, T.forward_backward, T.fixed_noise = LoggingSampler, fb, fn
    try:
        cfg = T.load_config(CFGS[backbone], COMMON + SMALL[backbone] + [f"seed={seed}", f"train.max_steps={steps}"])
        with tempfile.TemporaryDirectory() as d:
            T.train(cfg, d, resume=False)
    finally:
        T.ClassStratifiedSampler, T.forward_backward, T.fixed_noise = ClassStratifiedSampler, orig_fb, orig_fn
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args()
    logs = {b: run(b, a.seed, a.steps) for b in CFGS}
    same = {k: logs["dit"][k] == logs["unet"][k] for k in ("draws", "fb", "fixed_noise")}
    res = {"seed": a.seed, "steps": a.steps, "configs": CFGS, "overrides_common": COMMON, "overrides_small": SMALL,
           "identical": same, "all_identical": all(same.values()), "logs": logs}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"identical": same, "all_identical": res["all_identical"],
                      "z_sha_per_step": [r["z_sha"] for r in logs["dit"]["fb"]],
                      "first_indices": logs["dit"]["draws"][0]["indices"][:8]}))


if __name__ == "__main__":
    main()
