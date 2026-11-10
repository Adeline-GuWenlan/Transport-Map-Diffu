"""Does an energy distance on frozen Inception features see what the pixel one misses?

Same sets as scripts/patch_shuffle_check.py (right-class train, wrong-class train, 4x4- and 1x1-block-shuffled
right-class images; the reference is the test set), plus the CIFAR-10 CPU model's samples if --ckpt is given.
D1 is computed per class in pixel space and in Inception pool3 space (clean-fid legacy_tensorflow extractor,
2048-d, uint8 input resized inside the network).

    python scripts/feature_space_check.py --n 100 --ckpt runs/2026-11-06_smoke_cifar10_dit-t4/ckpt/latest.pt \
        --out runs/2026-11-06_loss_blindspot/cifar10_feature_space.json
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch_shuffle_check import patch_shuffle  # noqa: E402

from tmap.data import ImageArray, resolve  # noqa: E402
from tmap.energy_score import per_class_energy_distance  # noqa: E402
from tmap.evaluate import reference_set  # noqa: E402
from tmap.sample import generate, load_checkpoint_model, to_uint8  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    src = resolve("data/fid/inception/inception-2015-12-05.pt")
    if not os.path.exists("/tmp/inception-2015-12-05.pt"):
        shutil.copy(src, "/tmp/inception-2015-12-05.pt")
    from cleanfid.features import build_feature_extractor
    feat = build_feature_extractor("legacy_tensorflow", torch.device("cpu"), use_dataparallel=False)

    def features(x):  # x in [-1, 1] float -> 2048-d
        out = []
        for i in range(0, len(x), 50):
            with torch.no_grad():
                out.append(feat(to_uint8(x[i:i + 50]).float()).cpu())
        return torch.cat(out).double()

    train = ImageArray("data/cifar10", "train", 32)
    test = ImageArray("data/cifar10", "test", 32)
    ref_x, ref_l = reference_set(test, a.n)
    rng = np.random.default_rng(0)
    K = len(train.classes)
    sets = {}
    sets["right_class_train"] = torch.cat([train.get(train.by_class[c][a.n:2 * a.n])[0] for c in range(K)])
    sets["wrong_class_train"] = torch.cat([train.get(train.by_class[(c + 1) % K][:a.n])[0] for c in range(K)])
    for p in (4, 1):
        sets[f"block_shuffled_{p}"] = torch.cat([patch_shuffle(train.get(train.by_class[c][:2000])[0], p, rng)[:a.n]
                                                 for c in range(K)])
    if a.ckpt:
        m, _, _ = load_checkpoint_model(a.ckpt, "ema")
        c = torch.arange(K).repeat_interleave(a.n)
        sets["model_ema"] = generate(m, c, seed=0).clamp(-1, 1)
    labels = torch.arange(K).repeat_interleave(a.n)
    ref_f = features(ref_x)
    res = {"n_per_class": a.n, "pixel": {}, "inception": {}}
    yy_p, yy_f = {}, {}
    for name, x in sets.items():
        res["pixel"][name] = per_class_energy_distance(x, labels, ref_x, ref_l, yy_cache=yy_p)["mean"]
        res["inception"][name] = per_class_energy_distance(features(x), labels, ref_f, ref_l, yy_cache=yy_f)["mean"]
        print(f"{name:20s} pixel D1 {res['pixel'][name]:8.4f}   inception D1 {res['inception'][name]:9.3f}", flush=True)
    res["seconds"] = time.time() - t0
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out} ({res['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
