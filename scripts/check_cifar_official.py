"""Stage 3: compare data/cifar10/*.npy (decoded from the HF parquet mirror) with the official CIFAR-10
binary release (https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz).

    python -I scripts/check_cifar_official.py /path/to/cifar-10-binary.tar.gz

Reads the tar members in memory (no extraction, no pickle). Binary format (cifar.html): each record is
1 label byte + 3072 bytes (1024 R, 1024 G, 1024 B, row-major 32x32). Train = data_batch_1..5, test = test_batch.
Writes data/cifar10/official_crosscheck.json.
"""
import hashlib
import json
import os
import sys
import tarfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"   # cs.toronto.edu/~kriz/cifar.html, "CIFAR-10 binary version"


def read_bin(tf, name):
    raw = np.frombuffer(tf.extractfile(tf.getmember(name)).read(), dtype=np.uint8).reshape(-1, 3073)
    return raw[:, 1:].reshape(-1, 3, 32, 32), raw[:, 0].astype(np.int64)


def main():
    path = sys.argv[1]
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    out = {"official_file": path, "md5": md5, "md5_matches_cifar_html": md5 == OFFICIAL_MD5}
    with tarfile.open(path, "r:gz") as tf:
        parts = [read_bin(tf, f"cifar-10-batches-bin/data_batch_{i}.bin") for i in range(1, 6)]
        xo_tr, yo_tr = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        xo_te, yo_te = read_bin(tf, "cifar-10-batches-bin/test_batch.bin")
        names = tf.extractfile(tf.getmember("cifar-10-batches-bin/batches.meta.txt")).read().decode().split()
    out["official_class_names"] = names
    ok = out["md5_matches_cifar_html"]
    for split, xo, yo in (("train", xo_tr, yo_tr), ("test", xo_te, yo_te)):
        xh = np.load(os.path.join(ROOT, "data/cifar10", f"{split}_images.npy"))
        yh = np.load(os.path.join(ROOT, "data/cifar10", f"{split}_labels.npy"))
        same_order = xh.shape == xo.shape and bool((xh == xo).all()) and bool((yh == yo).all())
        r = {"n_official": len(xo), "n_ours": len(xh), "identical_in_same_order": same_order}
        if not same_order:   # fall back to matching as multisets of (image bytes, label)
            ko = sorted(xo[i].tobytes() + bytes([yo[i]]) for i in range(len(xo)))
            kh = sorted(xh[i].tobytes() + bytes([yh[i]]) for i in range(len(xh)))
            r["identical_as_multiset"] = ko == kh
        ok = ok and (same_order or r.get("identical_as_multiset", False))
        out[split] = r
        print(split, r)
    out["ok"] = bool(ok)
    print("md5", md5, "matches cifar.html:", out["md5_matches_cifar_html"], "| class names", names, "| ok", ok)
    with open(os.path.join(ROOT, "data/cifar10/official_crosscheck.json"), "w") as f:
        json.dump(out, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
