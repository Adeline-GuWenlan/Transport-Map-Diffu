"""Held-out evaluation: per-class unbiased energy distance D_1 (pixel space) against the held-out split, with
baselines; nearest-train-neighbour distances; within-class diversity; optional clean-fid FID (CIFAR-10 only;
on CPU at small sample counts it is indicative only).

    python -m tmap.evaluate --ckpt runs/<run>/ckpt/latest.pt --out runs/<run>/eval_final.json [--n-per-class 1000]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .data import load_split
from .energy_score import pairwise_dist_eval, per_class_energy_distance
from .sample import generate, to_uint8


def generate_per_class(model, classes, n_per_class: int, seed: int = 0, batch: int = 500, bf16: bool = False):
    c = torch.tensor([cl for cl in classes for _ in range(n_per_class)], dtype=torch.long)
    return generate(model, c, seed=seed, batch=batch, autocast_bf16=bf16), c


def reference_set(data, n_per_class: int | None):
    """First n_per_class images of every class (deterministic), float [-1, 1], with labels."""
    xs, ls = [], []
    for c in data.classes:
        xc = data.class_tensor(c, n_per_class)
        xs.append(xc)
        ls.append(torch.full((len(xc),), c, dtype=torch.long))
    return torch.cat(xs), torch.cat(ls)


def heldout_energy_distance(model, ref_x, ref_l, n_per_class: int, seed: int = 0, beta: float = 1.0,
                            yy_cache: dict | None = None, bf16: bool = False):
    classes = sorted(torch.unique(ref_l).tolist())
    x, lx = generate_per_class(model, classes, n_per_class, seed=seed, bf16=bf16)
    r = per_class_energy_distance(x, lx, ref_x, ref_l, beta, yy_cache)
    return r, x, lx


def baselines(train, ref_x, ref_l, n_per_class: int, seed: int = 0, beta: float = 1.0, yy_cache=None,
              which=("noise", "zero", "class_mean", "train", "other_class")):
    """Per-class D_1 of simple samplers against the held-out reference set (mean over classes reported).

    noise: N(0, I) images | zero: F == 0 (DiT at step 0) | class_mean: point mass at the class's train mean
    train: train images of the same class (floor) | other_class: train images of class (c + 1) mod K.
    """
    g = torch.Generator().manual_seed(seed)
    classes = sorted(torch.unique(ref_l).tolist())
    shape = ref_x.shape[1:]
    out = {}
    for name in which:
        xs, ls = [], []
        for i, c in enumerate(classes):
            if name == "noise":
                xc = torch.randn(n_per_class, *shape, generator=g)
            elif name == "zero":
                xc = torch.zeros(n_per_class, *shape)
            elif name == "class_mean":
                xc = train.class_tensor(c).mean(0, keepdim=True).expand(n_per_class, *shape).clone()
            elif name == "train":
                ids = train.by_class[c]
                sel = np.random.default_rng(seed + c).choice(ids, min(n_per_class, len(ids)), replace=False)
                xc = train.get(sel)[0]
            elif name == "other_class":
                c2 = classes[(i + 1) % len(classes)]
                ids = train.by_class[c2]
                sel = np.random.default_rng(seed + c).choice(ids, min(n_per_class, len(ids)), replace=False)
                xc = train.get(sel)[0]
            else:
                raise ValueError(name)
            xs.append(xc)
            ls.append(torch.full((len(xc),), c, dtype=torch.long))
        r = per_class_energy_distance(torch.cat(xs), torch.cat(ls), ref_x, ref_l, beta, yy_cache)
        out[name] = {"mean": r["mean"], "per_class": r["per_class"]}
    return out


@torch.no_grad()
def nearest_train(x, lx, train, max_train_per_class: int | None = None):
    """For each sample, the pixel L2 distance to its nearest same-class train image. Returns (dist, index into train)."""
    d_all = torch.empty(len(x), dtype=torch.float64)
    i_all = torch.empty(len(x), dtype=torch.long)
    for c in torch.unique(lx).tolist():
        mk = lx == c
        ids = train.by_class[c] if max_train_per_class is None else train.by_class[c][:max_train_per_class]
        tr = train.get(ids)[0]
        d = pairwise_dist_eval(x[mk], tr)
        v, j = d.min(1)
        d_all[mk] = v
        i_all[mk] = torch.as_tensor(ids)[j]
    return d_all, i_all


def nn_summary(x, lx, train, ref_x, ref_l):
    dg, ig = nearest_train(x, lx, train)
    dt, _ = nearest_train(ref_x, ref_l, train)
    q = lambda d: {k: float(v) for k, v in zip(["p05", "p50", "p95", "min"], [*torch.quantile(d, torch.tensor(
        [0.05, 0.5, 0.95], dtype=torch.float64)).tolist(), d.min().item()])}
    return {"generated_to_train": q(dg), "test_to_train": q(dt),
            "frac_generated_closer_than_test_p05": float((dg < torch.quantile(dt, 0.05)).double().mean())}, dg, ig


@torch.no_grad()
def class_confusion(x, lx, ref_x, ref_l, n: int = 200, beta: float = 1.0):
    """D1(generated class c, reference class c') for all pairs (first n of each). Row argmin == c means the
    samples of class c are closest in distribution to the right class. Returns (matrix list, row-argmin accuracy)."""
    classes = sorted(torch.unique(ref_l).tolist())
    gx = {c: x[lx == c][:n] for c in classes}
    rx = {c: ref_x[ref_l == c][:n] for c in classes}
    from .energy_score import _offdiag_mean
    xx = {c: float(_offdiag_mean(pairwise_dist_eval(gx[c], gx[c], beta))) for c in classes}
    yy = {c: float(_offdiag_mean(pairwise_dist_eval(rx[c], rx[c], beta))) for c in classes}
    M = [[2 * float(pairwise_dist_eval(gx[c], rx[d], beta).mean()) - xx[c] - yy[d] for d in classes] for c in classes]
    acc = float(np.mean([int(np.argmin(M[i]) == i) for i in range(len(classes))]))
    return M, acc


@torch.no_grad()
def nn_classifier_accuracy(x, lx, train, n_train_per_class: int = 1000, n_query_per_class: int = 200):
    """1-nearest-neighbour (pixel L2) classifier fitted on n_train_per_class train images per class; accuracy of its
    predicted label vs the conditioning label of the samples."""
    ids = np.concatenate([train.by_class[c][:n_train_per_class] for c in train.classes])
    tr, tl = train.get(ids)
    q = torch.cat([torch.nonzero(lx == c).squeeze(1)[:n_query_per_class] for c in torch.unique(lx).tolist()])
    pred = tl[pairwise_dist_eval(x[q], tr).argmin(1)]
    return float((pred == lx[q]).double().mean())


@torch.no_grad()
def blockiness(x, patch: int):
    """Mean |pixel step| across DiT patch boundaries divided by the mean |pixel step| inside patches (horizontal and
    vertical neighbours). About 1 for natural images; >> 1 means a visible patch grid."""
    dh = (x[..., :, 1:] - x[..., :, :-1]).abs()       # step between column j and j + 1
    dv = (x[..., 1:, :] - x[..., :-1, :]).abs()
    j = torch.arange(dh.shape[-1])
    bnd = (j + 1) % patch == 0                         # j = patch - 1, 2 patch - 1, ...: crosses a boundary
    b = torch.cat([dh[..., bnd].flatten(), dv[..., bnd, :].flatten()]).mean()
    w = torch.cat([dh[..., ~bnd].flatten(), dv[..., ~bnd, :].flatten()]).mean()
    return float(b / w)


@torch.no_grad()
def radial_power(x, bands=((1, 4), (4, 8), (8, 12), (12, 17))):
    """Mean radially binned power spectrum of (x - per-image mean), summed over channels, in frequency bands
    (cycles per image). Returns {band: power}."""
    x = x.double()
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    P = (torch.fft.fft2(x).abs() ** 2).mean(dim=(0, 1))       # (H, W)
    H, W = P.shape
    fy = torch.fft.fftfreq(H) * H
    fx = torch.fft.fftfreq(W) * W
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return {f"{lo}-{hi}": float(P[(r >= lo) & (r < hi)].mean()) for lo, hi in bands}


def fid_cleanfid(model, num_gen: int, dataset_name="cifar10", dataset_res=32, mode="legacy_tensorflow",
                 dataset_split="train", seed: int = 0, batch: int = 100, device="cpu"):
    """clean-fid FID of class-balanced samples vs precomputed train stats. Needs 3-channel data (CIFAR-10).
    Inception weights: clean-fid reads /tmp/inception-2015-12-05.pt (copied from data/fid/inception)."""
    import shutil
    from cleanfid import fid
    from .data import resolve
    src = resolve("data/fid/inception/inception-2015-12-05.pt")
    if not os.path.exists("/tmp/inception-2015-12-05.pt") and os.path.exists(src):
        shutil.copy(src, "/tmp/inception-2015-12-05.pt")
    K = model.num_classes
    state = {"i": 0}
    g = torch.Generator().manual_seed(seed)

    def gen(_z):
        b = _z.shape[0]
        c = (torch.arange(state["i"], state["i"] + b) % K).long()
        state["i"] += b
        z = torch.randn(b, model.in_channels, model.input_size, model.input_size, generator=g)
        return to_uint8(generate(model, c, z, batch=b))

    return float(fid.compute_fid(gen=gen, dataset_name=dataset_name, dataset_res=dataset_res, num_gen=num_gen,
                                 dataset_split=dataset_split, mode=mode, batch_size=batch,
                                 device=torch.device(device), use_dataparallel=False, num_workers=0, verbose=False))


def main():
    from .sample import load_checkpoint_model, save_grid
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="JSON output path")
    ap.add_argument("--n-per-class", type=int, default=None, help="default: eval.energy_distance.n_per_class_final")
    ap.add_argument("--weights", default="both", choices=["ema", "raw", "both"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fid-num-gen", type=int, default=0, help=">0: also compute clean-fid FID (CIFAR-10 only)")
    ap.add_argument("--nn-grid", default=None, help="PNG of generated samples next to their nearest train image")
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", len(os.sched_getaffinity(0)))))
    res = {"ckpt": a.ckpt}
    _, cfg, ck = load_checkpoint_model(a.ckpt, "ema", a.device)
    res["step"] = ck["step"]
    ec = cfg["eval"]["energy_distance"]
    n = a.n_per_class or ec.get("n_per_class_final", ec["n_per_class"])
    beta = ec.get("beta", 1.0)
    train = load_split(cfg, "train_split")
    held = load_split(cfg, "heldout_split")
    ref_x, ref_l = reference_set(held, n)
    n_eff = min(n, int(torch.bincount(ref_l).max()))
    res["n_per_class"] = n_eff
    yy = {}
    res["baselines"] = baselines(train, ref_x, ref_l, n_eff, seed=a.seed, beta=beta, yy_cache=yy,
                                 which=ec.get("baselines", ["noise", "zero", "class_mean", "train", "other_class"]))
    patch = cfg["model"]["patch_size"]
    res["reference_blockiness"] = blockiness(ref_x, patch)
    res["reference_radial_power"] = radial_power(ref_x)
    multi = len(train.classes) > 1
    if multi:
        # reference values for the class-identity checks: real held-out images in place of samples
        res["real_heldout_nn_classifier_accuracy"] = nn_classifier_accuracy(ref_x, ref_l, train)
    for w in (["ema", "raw"] if a.weights == "both" else [a.weights]):
        model, _, _ = load_checkpoint_model(a.ckpt, w, a.device)
        r, x, lx = heldout_energy_distance(model, ref_x, ref_l, n_eff, seed=a.seed, beta=beta, yy_cache=yy)
        nn_s, dg, ig = nn_summary(x, lx, train, ref_x, ref_l)
        res[w] = {"energy_distance": r, "nearest_neighbour": nn_s,
                  "diversity_ratio_mean": float(np.mean([r["within_x"][c] / r["within_y"][c] for c in r["within_x"]])),
                  "pixel_mean_abs_gen": float(x.abs().mean()), "frac_outside_minus1_1": float((x.abs() > 1).double().mean()),
                  "blockiness": blockiness(x, patch), "radial_power": radial_power(x)}
        res[w]["radial_power_ratio_gen_over_ref"] = {k: res[w]["radial_power"][k] / res["reference_radial_power"][k]
                                                     for k in res["reference_radial_power"]}
        if multi:
            M, acc = class_confusion(x, lx, ref_x, ref_l, n=min(200, n_eff), beta=beta)
            res[w]["class_confusion_D1"] = M
            res[w]["class_confusion_argmin_accuracy"] = acc
            res[w]["nn_classifier_accuracy"] = nn_classifier_accuracy(x, lx, train)
        if a.nn_grid and w == "ema":
            # 2 rows per class: generated (first 8) and their nearest train images
            rows = []
            for c in sorted(torch.unique(lx).tolist()):
                ids = torch.nonzero(lx == c).squeeze(1)[:8]
                rows += [x[ids], train.get(ig[ids])[0]]
            save_grid(torch.cat(rows), a.nn_grid, nrow=8)
        if a.fid_num_gen > 0:
            if model.in_channels != 3:
                res[w]["fid"] = None
                res[w]["fid_note"] = "skipped: FID needs RGB reference stats (CIFAR-10)"
            else:
                fc = cfg["eval"].get("fid", {})
                res[w]["fid"] = fid_cleanfid(model, a.fid_num_gen, fc.get("dataset_name", "cifar10"),
                                             fc.get("dataset_res", 32), fc.get("mode", "legacy_tensorflow"),
                                             fc.get("dataset_split", "train"), seed=a.seed, device=a.device)
                res[w]["fid_note"] = f"clean-fid {a.fid_num_gen} samples on {a.device}; indicative only if small"
    res["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    b = res["baselines"]
    print(f"step {res['step']}  n/class {n_eff}  held-out D1 (mean over classes):")
    for w in ("ema", "raw"):
        if w in res:
            print(f"  model[{w}] {res[w]['energy_distance']['mean']:.4f}  diversity ratio {res[w]['diversity_ratio_mean']:.3f}"
                  f"  NN gen->train p50 {res[w]['nearest_neighbour']['generated_to_train']['p50']:.3f}"
                  f" (test->train p50 {res[w]['nearest_neighbour']['test_to_train']['p50']:.3f})")
            print(f"    blockiness {res[w]['blockiness']:.3f} (reference {res['reference_blockiness']:.3f}); "
                  f"power ratio gen/ref by band {({k: round(v, 2) for k, v in res[w]['radial_power_ratio_gen_over_ref'].items()})}")
            if "nn_classifier_accuracy" in res[w]:
                print(f"    1-NN classifier agrees with the conditioning class: {res[w]['nn_classifier_accuracy']:.3f}"
                      f" (real test images: {res['real_heldout_nn_classifier_accuracy']:.3f}); "
                      f"D1-confusion row-argmin accuracy {res[w]['class_confusion_argmin_accuracy']:.2f}")
    for k, v in b.items():
        print(f"  baseline {k:12s} {v['mean']:.4f}")
    print(f"wrote {a.out} ({res['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
