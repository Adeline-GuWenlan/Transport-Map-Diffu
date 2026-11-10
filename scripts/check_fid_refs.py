"""Stage 3: end-to-end check of the FID tooling on real data (CPU, ~10 min).

    source env/activate.sh && python scripts/check_fid_refs.py [--modes legacy_tensorflow clean] [--n 10000]

1. Frechet distance between clean-fid's precomputed CIFAR-10 train stats (legacy_tensorflow and clean)
   and NVIDIA EDM's reference (data/fid/edm/cifar10-32x32.npz, 50k train images, same Inception net).
2. FID of the real CIFAR-10 *test* images (data/cifar10/test_images.npy) against the train stats, via
   cleanfid.fid.compute_fid with a "generator" that returns test images. This is the real-data floor that a
   generator's FID-10k is compared against, and it checks the whole chain (our decoding, uint8 layout,
   clean-fid, stats) at once.
Writes data/fid/check_fid_refs.json.
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="*", default=["legacy_tensorflow", "clean"])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=100)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS") or len(os.sched_getaffinity(0))))
    from cleanfid import fid

    out = {"torch_threads": torch.get_num_threads()}
    # 1. stats vs stats
    edm = np.load(os.path.join(ROOT, "data/fid/edm/cifar10-32x32.npz"), allow_pickle=False)
    for mode in ["legacy_tensorflow", "clean"]:
        st = np.load(os.path.join(ROOT, f"data/fid/stats/cifar10_{mode}_train_32.npz"), allow_pickle=False)
        d = float(fid.frechet_distance(st["mu"], st["sigma"], edm["mu"], edm["sigma"]))
        out[f"frechet(cleanfid_{mode}_train, edm_train)"] = d
        print(f"Frechet distance clean-fid {mode} train stats vs EDM train ref: {d:.4f}")

    # clean-fid loads the Inception weights from /tmp (hard-coded) and downloads them if absent.
    src = os.path.join(ROOT, "data/fid/inception/inception-2015-12-05.pt")
    dst = "/tmp/inception-2015-12-05.pt"
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)

    # 2. real test images as the "generator"
    x = np.load(os.path.join(ROOT, "data/cifar10/test_images.npy"))[: a.n]   # uint8 N,3,32,32
    for mode in a.modes:
        state = {"i": 0}

        def G(z):   # clean-fid calls G(z) with z of shape (batch, z_dim); returns uint8-range float (B,3,H,W)
            i = state["i"]
            b = x[i: i + z.shape[0]]
            state["i"] = i + z.shape[0]
            return torch.from_numpy(b.astype(np.float32))

        t0 = time.time()
        score = fid.compute_fid(gen=G, dataset_name="cifar10", dataset_res=32, dataset_split="train",
                                num_gen=a.n, batch_size=a.batch, mode=mode, z_dim=1,
                                device=torch.device("cpu"), use_dataparallel=False, verbose=False)
        assert state["i"] >= a.n, state
        dt = time.time() - t0
        out[f"fid_{mode}(cifar10_test[:{a.n}], cifar10_train_stats)"] = float(score)
        out[f"seconds_{mode}"] = round(dt, 1)
        print(f"FID {mode}: real CIFAR-10 test ({a.n}) vs train stats = {score:.4f}  ({dt:.0f} s on CPU)")
    with open(os.path.join(ROOT, "data/fid/check_fid_refs.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
