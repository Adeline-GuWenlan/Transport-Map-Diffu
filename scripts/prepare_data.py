"""Stage 3: decode the raw downloads in data/<name>/raw/ into arrays stage 4 can memory-map.

    python -I scripts/prepare_data.py [--root data] [--only fashion_mnist cifar10 mnist]

Writes, per dataset:
    data/<name>/{train,test}_images.npy   uint8, N x C x H x W, native resolution (28x28 gray, 32x32 RGB)
    data/<name>/{train,test}_labels.npy   int64, N
    data/<name>/meta.json                 shapes, class counts, pixel stats, sha256 of raw and npy files
File order is the order of the source files. No padding, scaling or augmentation here: the loader pads
28 -> 32 and maps uint8 to [-1, 1] (configs/smoke_cpu.yaml). Raw files are parsed with gzip+numpy (IDX)
and pyarrow+PIL (parquet); nothing is unpickled.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import sys

import numpy as np

FMNIST_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]   # github.com/zalandoresearch/fashion-mnist README
MNIST_CLASSES = [str(i) for i in range(10)]
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]             # cs.toronto.edu/~kriz/cifar.html, batches.meta order

# sha256 measured by stage 2 (docs/stage2_choices.md sec. 3.3) and re-measured by stage 3
RAW_SHA256 = {
    "fashion_mnist": {
        "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
        "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
        "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
        "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
    },
    "mnist": {
        "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
        "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
        "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
        "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
    },
    "cifar10": {
        "train-00000-of-00001.parquet": "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883",
        "test-00000-of-00001.parquet": "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de",
    },
}
SOURCES = {
    "fashion_mnist": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/ (Zalando, MIT licence)",
    "mnist": "https://ossci-datasets.s3.amazonaws.com/mnist/ (mirror of LeCun et al.; CC BY-SA 3.0 per torchvision docs)",
    "cifar10": "https://huggingface.co/datasets/uoft-cs/cifar10 @ 0b2714987fa478483af9968de7c934580d0bb9a2 "
               "(mirror of Krizhevsky 2009, https://www.cs.toronto.edu/~kriz/cifar.html; no licence stated)",
}


def sha256(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                return h.hexdigest()
            h.update(b)


def read_idx(path):
    """IDX format (yann.lecun.com/exdb/mnist): big-endian magic 0x0000 08 <ndim>, then dims, then uint8."""
    with gzip.open(path, "rb") as f:
        buf = f.read()
    zero, dtype, ndim = buf[0:2], buf[2], buf[3]
    if zero != b"\x00\x00" or dtype != 0x08:
        raise ValueError(f"{path}: bad IDX magic {buf[:4].hex()}")
    dims = np.frombuffer(buf, dtype=">u4", count=ndim, offset=4).astype(np.int64)
    data = np.frombuffer(buf, dtype=np.uint8, offset=4 + 4 * ndim)
    if data.size != int(np.prod(dims)):
        raise ValueError(f"{path}: {data.size} bytes, header says {dims}")
    return data.reshape(dims)


def load_idx_dataset(raw):
    out = {}
    for split, prefix in (("train", "train"), ("test", "t10k")):
        x = read_idx(os.path.join(raw, f"{prefix}-images-idx3-ubyte.gz"))[:, None]   # N,1,28,28
        y = read_idx(os.path.join(raw, f"{prefix}-labels-idx1-ubyte.gz")).astype(np.int64)
        if len(x) != len(y):
            raise ValueError(f"{split}: {len(x)} images vs {len(y)} labels")
        out[split] = (np.ascontiguousarray(x), y)
    return out, None


def load_cifar10_parquet(raw):
    import pyarrow.parquet as pq
    from PIL import Image
    out, hf_names = {}, None
    for split in ("train", "test"):
        tab = pq.read_table(os.path.join(raw, f"{split}-00000-of-00001.parquet"))
        md = tab.schema.metadata or {}
        if b"huggingface" in md:
            feats = json.loads(md[b"huggingface"]).get("info", {}).get("features", {})
            hf_names = feats.get("label", {}).get("names", hf_names)
        cols = tab.to_pydict()
        imgs = []
        for r in cols["img"]:
            im = Image.open(io.BytesIO(r["bytes"]))
            if im.size != (32, 32):
                raise ValueError(f"image size {im.size}")
            imgs.append(np.asarray(im.convert("RGB")))
        x = np.stack(imgs).transpose(0, 3, 1, 2)                                     # N,3,32,32
        y = np.asarray(cols["label"], dtype=np.int64)
        out[split] = (np.ascontiguousarray(x), y)
    return out, hf_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    ap.add_argument("--only", nargs="*", default=["fashion_mnist", "mnist", "cifar10"])
    a = ap.parse_args()

    expect = {"fashion_mnist": (60000, 10000, (1, 28, 28), FMNIST_CLASSES),
              "mnist": (60000, 10000, (1, 28, 28), MNIST_CLASSES),
              "cifar10": (50000, 10000, (3, 32, 32), CIFAR10_CLASSES)}
    # Fashion-MNIST and CIFAR-10 are exactly balanced; MNIST is not (standard counts, e.g. torchvision docs / LeCun)
    want_counts = {"fashion_mnist": {"train": [6000] * 10, "test": [1000] * 10},
                   "mnist": {"train": [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
                             "test": [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]},
                   "cifar10": {"train": [5000] * 10, "test": [1000] * 10}}
    ok = True
    for name in a.only:
        d = os.path.join(a.root, name)
        raw = os.path.join(d, "raw")
        raw_sha = {f: sha256(os.path.join(raw, f)) for f in sorted(os.listdir(raw))}
        bad = {f: h for f, h in RAW_SHA256[name].items() if raw_sha.get(f) != h}
        if bad:
            print(f"[{name}] RAW SHA256 MISMATCH: {sorted(bad)}"); ok = False; continue
        data, hf_names = load_cifar10_parquet(raw) if name == "cifar10" else load_idx_dataset(raw)
        n_tr, n_te, chw, classes = expect[name]
        if hf_names is not None and list(hf_names) != classes:
            print(f"[{name}] class-name order in parquet metadata {hf_names} != {classes}"); ok = False
        meta = {"name": name, "source": SOURCES[name], "raw_sha256": raw_sha, "classes": classes,
                "layout": "uint8 N x C x H x W (images), int64 N (labels); source file order",
                "hf_label_names": hf_names, "splits": {}}
        for split, (x, y) in data.items():
            want_n = n_tr if split == "train" else n_te
            counts = np.bincount(y, minlength=len(classes)).tolist()
            checks = {"n": len(x) == want_n, "shape": tuple(x.shape[1:]) == chw, "dtype": x.dtype == np.uint8,
                      "labels_in_range": int(y.min()) >= 0 and int(y.max()) < len(classes),
                      "class_counts_as_expected": counts == want_counts[name][split]}
            if not all(checks.values()):
                print(f"[{name}/{split}] FAILED checks {checks}"); ok = False
            fx, fy = os.path.join(d, f"{split}_images.npy"), os.path.join(d, f"{split}_labels.npy")
            np.save(fx, x); np.save(fy, y)
            xf = x.astype(np.float64) / 255.0
            meta["splits"][split] = {
                "n": len(x), "image_shape": list(x.shape[1:]), "class_counts": counts,
                "pixel_mean_per_channel_01": xf.mean(axis=(0, 2, 3)).round(6).tolist(),
                "pixel_std_per_channel_01": xf.std(axis=(0, 2, 3)).round(6).tolist(),
                "n_unique_images": int(len(np.unique(x.reshape(len(x), -1), axis=0))),
                "files": {os.path.basename(fx): sha256(fx), os.path.basename(fy): sha256(fy)},
                "checks": checks,
            }
            print(f"[{name}/{split}] {x.shape} {x.dtype} counts={counts} "
                  f"mean={meta['splits'][split]['pixel_mean_per_channel_01']} "
                  f"unique={meta['splits'][split]['n_unique_images']} checks_ok={all(checks.values())}")
        # exact duplicates shared between train and test (known to exist in some of these sets)
        tr = {r.tobytes() for r in data["train"][0].reshape(len(data["train"][0]), -1)}
        meta["n_test_images_identical_to_a_train_image"] = int(
            sum(r.tobytes() in tr for r in data["test"][0].reshape(len(data["test"][0]), -1)))
        print(f"[{name}] test images byte-identical to some train image: "
              f"{meta['n_test_images_identical_to_a_train_image']}")
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
