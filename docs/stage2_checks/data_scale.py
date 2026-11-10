"""Stage-2 measurement: the scale of the energy-score loss in pixel space, and held-out D_1
baselines, for MNIST, Fashion-MNIST and CIFAR-10 at 32x32 in [-1, 1].

  python -I docs/stage2_checks/data_scale.py --mnist DIR --fmnist DIR --cifar10 DIR --out JSON

DIRs hold the raw downloads: idx .gz files (MNIST, Fashion-MNIST) and either the extracted
cifar-10-batches-bin/ (--cifar10) or the HF uoft-cs/cifar10 parquet files (--cifar10-parquet).
Parsed with numpy / pyarrow + PIL only (no pickle).
MNIST-style 28x28 images are zero-padded (value -1 after scaling) to 32x32, 1 channel.

Per class c (n = 1000 test images, "Q_c"), unbiased D_1 (stage-1 note, Sec. 7) of:
  noise   : N(0, I) samples (what a randomly initialised non-DiT map would give; the "ceiling")
  zero    : point mass at 0 (DiT at step 0: adaLN-Zero makes F == 0)
  mean    : point mass at the class mean of the train set
  train   : 1000 train images of class c (the floor; ~0 up to noise)
  other   : 1000 train images of class c+1 (wrong class)
Also: E||Y - Y'|| within class (twice the optimal loss value up to sign: at P = Q the
expected loss equals 1/2 E||Y - Y'||) and the RMS pixel value.
"""
import argparse, gzip, json, os
import numpy as np


def read_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_mnist_like(d):
    xtr = read_idx(os.path.join(d, "train-images-idx3-ubyte.gz"))
    ytr = read_idx(os.path.join(d, "train-labels-idx1-ubyte.gz"))
    xte = read_idx(os.path.join(d, "t10k-images-idx3-ubyte.gz"))
    yte = read_idx(os.path.join(d, "t10k-labels-idx1-ubyte.gz"))
    pad = lambda x: np.pad(x, ((0, 0), (2, 2), (2, 2)))[:, None]   # 28 -> 32, 1 channel
    return pad(xtr), ytr, pad(xte), yte


def load_cifar10_bin(d):
    d = os.path.join(d, "cifar-10-batches-bin")
    def rd(names):
        raw = np.concatenate([np.fromfile(os.path.join(d, n), dtype=np.uint8) for n in names])
        raw = raw.reshape(-1, 3073)
        return raw[:, 1:].reshape(-1, 3, 32, 32), raw[:, 0]
    xtr, ytr = rd([f"data_batch_{i}.bin" for i in range(1, 6)])
    xte, yte = rd(["test_batch.bin"])
    return xtr, ytr, xte, yte


def load_cifar10_parquet(d, pyarrow_dir=None):
    """Hugging Face uoft-cs/cifar10 parquet (PNG bytes per row), decoded with PIL."""
    import io, sys
    if pyarrow_dir:
        sys.path.insert(0, pyarrow_dir)
    import pyarrow.parquet as pq
    from PIL import Image
    def rd(fn):
        t = pq.read_table(os.path.join(d, fn)).to_pydict()
        x = np.stack([np.asarray(Image.open(io.BytesIO(r["bytes"])).convert("RGB")) for r in t["img"]])
        return x.transpose(0, 3, 1, 2), np.asarray(t["label"])
    xtr, ytr = rd("train-00000-of-00001.parquet")
    xte, yte = rd("test-00000-of-00001.parquet")
    return xtr, ytr, xte, yte


def to_pm1(x):
    return x.reshape(len(x), -1).astype(np.float64) / 127.5 - 1.0


def pdist(a, b):
    s = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.maximum(s, 0))


def offdiag_mean(K):
    m = K.shape[0]
    return (K.sum() - np.trace(K)) / (m * (m - 1))


def d1_u(x, y):
    """unbiased D_1; a point mass (all rows equal) gives E||X - X'|| = 0 exactly."""
    return 2 * pdist(x, y).mean() - offdiag_mean(pdist(x, x)) - offdiag_mean(pdist(y, y))


def analyse(name, xtr, ytr, xte, yte, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    xtr, xte = to_pm1(xtr), to_pm1(xte)
    d = xtr.shape[1]
    out = dict(dataset=name, n_train=len(xtr), n_test=len(xte), dim=d,
               train_class_counts=np.bincount(ytr).tolist(), test_class_counts=np.bincount(yte).tolist(),
               rms_pixel=float(np.sqrt((xtr ** 2).mean())), pixel_std=float(xtr.std()),
               sqrt_d=float(np.sqrt(d)), per_class={})
    K = int(ytr.max()) + 1
    acc = {k: [] for k in ["noise", "zero", "mean", "train", "other", "EYY_within", "EYY_all"]}
    for c in range(K):
        te = xte[yte == c]; te = te[rng.choice(len(te), min(n, len(te)), replace=False)]
        trc = xtr[ytr == c]; tr = trc[rng.choice(len(trc), n, replace=False)]
        oth = xtr[ytr == (c + 1) % K]; oth = oth[rng.choice(len(oth), n, replace=False)]
        noise = rng.standard_normal((n, d))
        r = dict(noise=d1_u(noise, te), zero=d1_u(np.zeros((n, d)), te),
                 mean=d1_u(np.repeat(trc.mean(0, keepdims=True), n, 0), te),
                 train=d1_u(tr, te), other=d1_u(oth, te), EYY_within=offdiag_mean(pdist(te, te)))
        allte = xte[rng.choice(len(xte), n, replace=False)]
        r["EYY_all"] = offdiag_mean(pdist(allte, allte))
        out["per_class"][c] = {k: round(float(v), 4) for k, v in r.items()}
        for k, v in r.items():
            acc[k].append(v)
    out["mean_over_classes"] = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
    print(json.dumps({k: out[k] for k in ["dataset", "dim", "rms_pixel", "pixel_std", "mean_over_classes"]}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist"); ap.add_argument("--fmnist"); ap.add_argument("--cifar10")
    ap.add_argument("--cifar10-parquet"); ap.add_argument("--pyarrow-dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = []
    if a.mnist: res.append(analyse("MNIST", *load_mnist_like(a.mnist)))
    if a.fmnist: res.append(analyse("Fashion-MNIST", *load_mnist_like(a.fmnist)))
    if a.cifar10: res.append(analyse("CIFAR-10", *load_cifar10_bin(a.cifar10)))
    if a.cifar10_parquet:
        res.append(analyse("CIFAR-10", *load_cifar10_parquet(a.cifar10_parquet, a.pyarrow_dir)))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
