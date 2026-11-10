"""Stage 3 environment check. Exits non-zero if any check fails.

    source env/activate.sh
    python scripts/check_env.py                                   # smoke config, CPU
    python scripts/check_env.py --config configs/smoke_cpu_cifar10.yaml
    python scripts/check_env.py --config configs/base_gpu.yaml --device cuda   # on a GPU node

Checks: package versions; third_party/DiT at the pinned commit and unmodified; import of class DiT
from third_party/DiT/models.py; the model built from the config; one forward + backward pass on a
real, class-stratified data batch (timed), with the stage-1 energy-score estimator as the loss; the
data files of data/README.md; the FID reference stats and Inception weights.

This is a check, not training code: the loss below is a minimal copy of stage 1's estimator
(docs/stage1_checks/es_core.py, torch_pow_dist / torch_es_loss_grouped, beta = 1, U-statistic).
"""
import argparse
import hashlib
import importlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIT_DIR = os.path.join(ROOT, "third_party", "DiT")
DIT_COMMIT = "ed81ce2229091fd4ecc9a223645f95cf379d582b"
EXPECTED = {"python": "3.11", "torch": "2.14.1", "torch_cuda": "12.6", "timm": "1.0.31",
            "torchvision": "0.29.1", "cleanfid": "0.1.35"}
FAILS = []


def check(name):
    """Decorator: run a check, print PASS/FAIL, record failures, never raise."""
    def deco(fn):
        def run(*a, **k):
            t0 = time.perf_counter()
            try:
                out = fn(*a, **k)
                print(f"[PASS] {name} ({time.perf_counter() - t0:.2f} s)")
                return out
            except Exception as e:  # noqa: BLE001
                FAILS.append(name)
                print(f"[FAIL] {name}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=3, file=sys.stdout)
                return None
        return run
    return deco


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@check("versions")
def versions():
    import torch
    print(f"  host {socket.gethostname()}  python {platform.python_version()} ({sys.executable})")
    print(f"  torch {torch.__version__}  cuda build {torch.version.cuda}  cudnn {torch.backends.cudnn.version()}"
          f"  cuda available {torch.cuda.is_available()}  threads {torch.get_num_threads()}")
    if torch.cuda.is_available():
        print(f"  gpu {torch.cuda.get_device_name(0)}  driver-visible devices {torch.cuda.device_count()}")
    vers = {}
    for mod in ["torchvision", "timm", "numpy", "scipy", "yaml", "PIL", "pyarrow", "matplotlib", "tqdm",
                "cleanfid", "requests", "pytest", "huggingface_hub", "safetensors"]:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", None)
        if v is None:
            from importlib.metadata import version
            v = version({"cleanfid": "clean-fid", "yaml": "pyyaml", "PIL": "pillow"}.get(mod, mod))
        vers[mod] = v
    print("  " + "  ".join(f"{k} {v}" for k, v in vers.items()))
    got = {"python": platform.python_version(), "torch": torch.__version__.split("+")[0],
           "torch_cuda": torch.version.cuda, "timm": vers["timm"], "torchvision": vers["torchvision"].split("+")[0],
           "cleanfid": vers["cleanfid"]}
    bad = {k: (got[k], v) for k, v in EXPECTED.items() if not str(got[k]).startswith(v)}
    assert not bad, f"version mismatch (got, expected): {bad}"
    want_env = os.environ.get("TMAP_ENV", "/scratch/wg2381/.conda/envs/tmap")
    assert os.path.realpath(sys.prefix) == os.path.realpath(want_env), \
        f"not running in the tmap env: {sys.prefix}"


@check("third_party/DiT pinned and unmodified")
def dit_clone():
    head = subprocess.run(["git", "-C", DIT_DIR, "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", DIT_DIR, "status", "--porcelain"], capture_output=True, text=True,
                           check=True).stdout.strip()
    print(f"  {DIT_DIR} HEAD {head}  porcelain {'clean' if not dirty else repr(dirty)}")
    assert head == DIT_COMMIT, f"HEAD {head} != {DIT_COMMIT}"
    assert not dirty, "third_party/DiT has local changes or untracked files"


@check("import DiT from third_party/DiT/models.py")
def import_dit():
    sys.dont_write_bytecode = True           # keep the clone clean even without PYTHONDONTWRITEBYTECODE
    sys.path.insert(0, DIT_DIR)
    import models as dit_models               # noqa: E402  (third_party/DiT/models.py)
    assert os.path.realpath(dit_models.__file__) == os.path.realpath(os.path.join(DIT_DIR, "models.py")), \
        f"imported the wrong `models`: {dit_models.__file__}"
    print(f"  {dit_models.DiT} from {dit_models.__file__}")
    return dit_models.DiT


def load_cfg(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


@check("build model from config")
def build_model(DiT, cfg):
    m = cfg["model"]
    assert m["backbone"] == "dit" and not m["learn_sigma"] and m["out_channels"] == m["in_channels"], m
    model = DiT(input_size=m["input_size"], patch_size=m["patch_size"], in_channels=m["in_channels"],
                hidden_size=m["hidden_size"], depth=m["depth"], num_heads=m["num_heads"],
                mlp_ratio=m["mlp_ratio"], class_dropout_prob=m["class_dropout_prob"],
                num_classes=cfg["data"]["num_classes"], learn_sigma=m["learn_sigma"])
    n = sum(p.numel() for p in model.parameters())
    print(f"  {cfg['name']}: depth {m['depth']} hidden {m['hidden_size']} heads {m['num_heads']} "
          f"patch {m['patch_size']} -> {n / 1e6:.3f} M params; out_channels {model.out_channels}")
    assert model.out_channels == m["out_channels"]
    return model


@check("data files (data/README.md)")
def data_files():
    import numpy as np
    for name, n_tr, n_te, chw in [("fashion_mnist", 60000, 10000, (1, 28, 28)), ("mnist", 60000, 10000, (1, 28, 28)),
                                  ("cifar10", 50000, 10000, (3, 32, 32))]:
        d = os.path.join(ROOT, "data", name)
        meta = json.load(open(os.path.join(d, "meta.json")))
        for split, n in (("train", n_tr), ("test", n_te)):
            x = np.load(os.path.join(d, f"{split}_images.npy"), mmap_mode="r")
            y = np.load(os.path.join(d, f"{split}_labels.npy"))
            counts = np.bincount(y, minlength=10)
            assert x.shape == (n, *chw) and x.dtype == np.uint8, (name, split, x.shape, x.dtype)
            assert y.shape == (n,) and y.dtype == np.int64, (name, split, y.shape, y.dtype)
            assert meta["splits"][split]["class_counts"] == counts.tolist() and all(meta["splits"][split]["checks"].values())
            if name != "mnist":   # MNIST is not class-balanced
                assert (counts == n // 10).all(), (name, split, counts)
        print(f"  {name}: train {n_tr} / test {n_te}, {chw}, uint8, "
              f"{'%d/%d per class' % (n_tr // 10, n_te // 10) if name != 'mnist' else 'MNIST class counts (unbalanced)'}")


def real_batch(cfg, seed):
    """Class-stratified batch as in the config: classes_per_batch classes x n_per_class images, scaled to [-1, 1]."""
    import numpy as np
    import torch
    dc, bc = cfg["data"], cfg["batch"]
    d = os.path.join(ROOT, dc["root"])
    x = np.load(os.path.join(d, f"{dc['train_split']}_images.npy"), mmap_mode="r")
    y = np.load(os.path.join(d, f"{dc['train_split']}_labels.npy"))
    rng = np.random.default_rng(seed)
    classes = np.sort(rng.choice(dc["num_classes"], bc["classes_per_batch"], replace=False))
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), bc["n_per_class"], replace=False) for c in classes])
    xb = torch.from_numpy(np.ascontiguousarray(x[idx])).float()
    pad = (dc["image_size"] - xb.shape[-1]) // 2
    if pad:
        xb = torch.nn.functional.pad(xb, (pad, pad, pad, pad), value=0.0)   # uint8 0 -> -1 after scaling
    xb = xb / 127.5 - 1.0
    assert xb.shape[1:] == (dc["channels"], dc["image_size"], dc["image_size"]), xb.shape
    return xb, torch.from_numpy(y[idx])


def pow_dist(a, b):
    """||a_i - b_j|| (beta = 1), exact, gradient 0 at a_i == b_j (double-where; stage 1 es_core.torch_pow_dist)."""
    import torch
    diff = a[:, None, :] - b[None, :, :]
    sq = (diff * diff).sum(-1)
    pos = sq > 0
    return torch.where(pos, torch.where(pos, sq, torch.ones_like(sq)).sqrt(), torch.zeros_like(sq))


def es_loss(x, y, labels):
    """Stage 1 estimator, eq. (9): sum_c n_c/N [mean ||x - y|| - 1/(2 m(m-1)) sum_{j != k} ||x_j - x_k||], m_c = n_c."""
    x, y = x.flatten(1).float(), y.flatten(1).float()
    loss = x.new_zeros(())
    for c in labels.unique():
        mk = labels == c
        xc, yc = x[mk], y[mk]
        m = xc.shape[0]
        dxx = pow_dist(xc, xc)
        off = ~torch_eye(m, x.device)
        loss = loss + (yc.shape[0] / y.shape[0]) * (pow_dist(xc, yc).mean() - 0.5 * dxx[off].sum() / (m * (m - 1)))
    return loss


def torch_eye(m, device):
    import torch
    return torch.eye(m, dtype=torch.bool, device=device)


@check("forward + backward on a real batch")
def fwd_bwd(model, cfg, device, reps):
    import torch
    torch.manual_seed(cfg["seed"])
    if device == "cpu":
        # os.cpu_count() is the whole node (128 on cs627), not the Slurm allocation: use the affinity mask
        torch.set_num_threads(int(cfg.get("num_threads") or os.environ.get("OMP_NUM_THREADS")
                                  or len(os.sched_getaffinity(0))))
    yb, cb = real_batch(cfg, cfg["seed"])
    model = model.to(device).train()
    yb, cb = yb.to(device), cb.to(device)
    oc = cfg["optim"]
    opt = torch.optim.AdamW(model.parameters(), lr=oc["lr"], betas=tuple(oc["betas"]), eps=oc["eps"],
                            weight_decay=oc["weight_decay"])
    t_val = float(cfg["model"]["timestep_value"])
    bf16 = device == "cuda" and cfg.get("train", {}).get("precision") == "bf16"
    if device == "cuda" and cfg.get("train", {}).get("tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"  batch: {tuple(yb.shape)} from {cfg['data']['root']}, classes {cb.unique().tolist()}, "
          f"range [{yb.min():.2f}, {yb.max():.2f}], device {device}, torch threads {torch.get_num_threads()}, "
          f"network autocast {'bf16' if bf16 else 'off (fp32)'}")

    def step(do_opt):
        z = torch.randn_like(yb)
        t = torch.full((yb.shape[0],), t_val, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
            x = model(z, t, cb)
        loss = es_loss(x, yb, cb)   # distances in fp32 (es_loss casts), as configs/base_gpu.yaml asks
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), oc["grad_clip_norm"])
        if do_opt:
            opt.step()
        return x, loss, gn

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    # step 0: the single checked forward + backward (adaLN-Zero => F == 0, only final_layer.linear gets gradient)
    sync(); t0 = time.perf_counter()
    x, loss, gn = step(do_opt=False)
    sync(); t_first = time.perf_counter() - t0
    assert x.shape == yb.shape, (x.shape, yb.shape)
    assert torch.isfinite(loss), loss
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]   # DiT: fixed sin-cos pos_embed
    grads = [(n, p.grad) for n, p in model.named_parameters() if p.requires_grad]
    missing = [n for n, g in grads if g is None or not torch.isfinite(g).all()]
    assert not missing, f"missing or non-finite gradient: {missing}"
    print(f"  frozen (requires_grad=False, no grad expected): {frozen}")
    nz = [n for n, g in grads if g.abs().sum() > 0]
    e_norm = yb.flatten(1).norm(dim=1).mean().item()
    print(f"  step 0: max|F(z,c)| = {x.abs().max().item():.2e} (adaLN-Zero: expect 0); loss {loss.item():.4f} "
          f"(F == 0 => loss = mean||y|| = {e_norm:.4f}); pre-clip grad norm {gn.item():.3f}; "
          f"params with non-zero grad: {len(nz)}/{len(grads)} {nz}")
    print(f"  step 0 fwd+loss+bwd+clip: {t_first:.3f} s (includes first-call overhead)")
    assert abs(loss.item() - e_norm) < 1e-3 * e_norm, "loss at F == 0 should equal mean ||y||"

    # timed train steps (fwd + loss + bwd + clip + AdamW), as stage 2 measured (0.138 s/step, smoke config, 8 threads)
    times, losses, nzs = [], [], []
    for _ in range(reps):
        sync(); t0 = time.perf_counter()
        x, loss, gn = step(do_opt=True)
        sync(); times.append(time.perf_counter() - t0)
        losses.append(loss.item())
        assert torch.isfinite(loss) and torch.isfinite(gn), (loss, gn)
    nzs = sum(int(p.grad.abs().sum() > 0) for p in model.parameters() if p.requires_grad)
    times.sort()
    print(f"  {reps} train steps (fwd+loss+bwd+clip+AdamW, batch {yb.shape[0]}): median {times[len(times) // 2]:.3f} s, "
          f"min {times[0]:.3f} s, max {times[-1]:.3f} s; losses {[round(v, 3) for v in losses]}; "
          f"params with non-zero grad after {reps} steps: {nzs}/{len(grads)}; max|F| {x.abs().max().item():.3f}")


@check("FID assets (clean-fid, reference stats, Inception)")
def fid_assets(device, run_inception):
    import numpy as np
    import torch
    import cleanfid
    from cleanfid import features
    want = {  # sha256 measured by stage 3 (data/README.md)
        "stats/cifar10_legacy_tensorflow_train_32.npz": "8e2ddeabe3515e74058f52218e93f0c87adb935cd1352f4cff883bac0c4eaa9a",
        "stats/cifar10_clean_train_32.npz": "a7b9763a8316c2b03997d92ee9c3900588b563326521fba05373f16e179b0fb9",
        "inception/inception-2015-12-05.pt": "f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4",
    }
    for rel, h in want.items():
        p = os.path.join(ROOT, "data", "fid", rel)
        assert sha256(p) == h, f"sha256 mismatch {p}"
    for rel in [k for k in want if k.endswith(".npz")]:
        st = np.load(os.path.join(ROOT, "data", "fid", rel), allow_pickle=False)
        assert st["mu"].shape == (2048,) and st["sigma"].shape == (2048, 2048), rel
        print(f"  {rel}: mu {st['mu'].shape} sigma {st['sigma'].shape}")
    # clean-fid caches stats in <package>/stats/; stage 3 copied ours there (see data/README.md)
    pkg_stats = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    for rel in [k for k in want if k.endswith(".npz")]:
        p = os.path.join(pkg_stats, os.path.basename(rel))
        assert os.path.exists(p) and sha256(p) == want[rel], f"missing or different in clean-fid cache: {p}"
    print(f"  clean-fid {cleanfid.__file__}: stats cache {pkg_stats} has both npz files")
    if run_inception:
        from cleanfid.inception_torchscript import InceptionV3W
        net = InceptionV3W(os.path.join(ROOT, "data", "fid", "inception"), download=False, resize_inside=False).to(device)
        img = torch.rand(2, 3, 299, 299, device=device) * 255
        with torch.no_grad():
            f = net(img)
        assert f.shape == (2, 2048) and torch.isfinite(f).all(), f.shape
        print(f"  InceptionV3W (torchscript) forward on 2 random 299px images -> {tuple(f.shape)}")
    _ = features  # imported to check the module loads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "smoke_cpu.yaml"))
    ap.add_argument("--device", default=None, help="default: the config's device")
    ap.add_argument("--reps", type=int, default=10, help="timed train steps after the checked step 0")
    ap.add_argument("--no-inception", action="store_true", help="skip the Inception forward pass")
    ap.add_argument("--n-per-class", type=int, default=None,
                    help="override batch.n_per_class (e.g. 2 to dry-run configs/base_gpu.yaml on a CPU node)")
    a = ap.parse_args()
    print(f"check_env.py  config {os.path.relpath(a.config, ROOT)}  {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    versions()
    dit_clone()
    DiT = import_dit()
    cfg = load_cfg(a.config)
    if a.n_per_class:
        print(f"  NOTE: batch.n_per_class overridden {cfg['batch']['n_per_class']} -> {a.n_per_class}")
        cfg["batch"]["n_per_class"] = a.n_per_class
    device = a.device or cfg.get("device", "cpu")
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            FAILS.append("cuda requested but torch.cuda.is_available() is False")
    model = build_model(DiT, cfg) if DiT is not None else None
    data_files()
    if model is not None and "cuda requested but torch.cuda.is_available() is False" not in FAILS:
        fwd_bwd(model, cfg, device, a.reps)
    import torch
    fid_assets("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu", not a.no_inception)
    dit_clone()  # again: importing must not have touched the clone
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {FAILS}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
