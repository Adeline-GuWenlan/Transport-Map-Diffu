"""Stage-2 measurement: minibatch-gradient noise of the energy-score loss at DiT scale, as the number
of generated samples per data point r (m_c = r n_c) changes, compared with data noise.

  TIMM_DIR=<dir with timm> python -I docs/stage2_checks/grad_var_m.py --fmnist DIR --out JSON

Model: the smoke DiT-T/4 (configs/smoke_cpu.yaml) on Fashion-MNIST 32x32, warmed up for --warm steps
(at step 0, adaLN-Zero gives F == 0 and the interaction gradient is exactly 0, so that point is not
representative). At the warmed-up theta:
  z-noise(r)   : fixed data batch (8 classes x 8), K fresh z draws with r samples per data point,
                 per-class pooled estimator (eq. 9).  tr Cov of the flattened parameter gradient.
  data+z(r=1)  : K fresh data batches and fresh z.
  pairs(r=2)   : engression-style per-data-point groups (n_g = 1, m_g = 2), fixed data, same cost as r=2.
Reports tr Cov / ||mean grad||^2 (noise-to-signal) and seconds per gradient.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "DiT"))
sys.path.insert(0, os.path.join(ROOT, "docs", "stage2_checks"))
if os.environ.get("TIMM_DIR"):
    sys.path.insert(0, os.environ["TIMM_DIR"])
import warnings; warnings.filterwarnings("ignore")
from models import DiT  # noqa: E402
from data_scale import load_mnist_like  # noqa: E402
from time_dit import pow_dist  # noqa: E402


def es_loss(x, y, gx, gy):
    """eq. (9): groups by id; group weight n_g / N; U-statistic interaction."""
    x, y = x.flatten(1), y.flatten(1)
    N = y.shape[0]
    loss = x.new_zeros(())
    for g in gy.unique():
        xc, yc = x[gx == g], y[gy == g]
        m = xc.shape[0]
        dxx = pow_dist(xc, xc)
        loss = loss + (yc.shape[0] / N) * (pow_dist(xc, yc).mean() - 0.5 * (dxx.sum() - dxx.diagonal().sum()) / (m * (m - 1)))
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmnist", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--warm", type=int, default=400); ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--clip", type=float, default=2.0, help="<= 0: no clipping")
    ap.add_argument("--warm-only", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0); rng = np.random.default_rng(0)
    xtr, ytr, _, _ = load_mnist_like(a.fmnist)
    X = torch.from_numpy(xtr.astype(np.float32) / 127.5 - 1); Y = torch.from_numpy(ytr.astype(np.int64))
    by_c = [np.where(ytr == c)[0] for c in range(10)]

    def batch(nc=8, npc=8):
        cs = rng.choice(10, nc, replace=False)
        idx = np.concatenate([rng.choice(by_c[c], npc, replace=False) for c in cs])
        return X[idx], Y[idx]

    model = DiT(input_size=32, patch_size=4, in_channels=1, depth=6, hidden_size=256, num_heads=4,
                num_classes=10, class_dropout_prob=0.0, learn_sigma=False)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.0)
    T = 999.0
    t0 = time.time(); hist = []; gnorms = []
    for s in range(a.warm):
        for gpar in opt.param_groups:
            gpar["lr"] = 2e-4 * min(1.0, (s + 1) / 100)
        y, c = batch()
        x = model(torch.randn_like(y), torch.full((len(y),), T), c)
        loss = es_loss(x, y, c, c)
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip if a.clip > 0 else float("inf"))
        opt.step()
        hist.append(float(loss)); gnorms.append(float(gn))
    warm_s = time.time() - t0
    print(f"warm-up {a.warm} steps in {warm_s:.0f}s, loss first10 {np.mean(hist[:10]):.3f} last10 {np.mean(hist[-10:]):.3f}", flush=True)
    q = lambda v: [round(float(np.percentile(v, p)), 2) for p in (0, 10, 50, 90, 100)]
    warm = dict(clip=a.clip, loss_by_100=[round(float(np.mean(hist[i:i + 100])), 3) for i in range(0, a.warm, 100)],
                gradnorm_pct_0_10_50_90_100_first100=q(gnorms[:100]), gradnorm_pct_last100=q(gnorms[-100:]))
    print(json.dumps(warm), flush=True)
    if a.warm_only:
        with open(a.out, "w") as f:
            json.dump(warm, f, indent=1)
        return
    params = [p for p in model.parameters() if p.requires_grad]

    def grad(y, c, r, mode="pooled"):
        yr, cr = y.repeat_interleave(r, 0), c.repeat_interleave(r, 0)
        x = model(torch.randn_like(yr), torch.full((len(yr),), T), cr)
        if mode == "pooled":
            loss = es_loss(x, y, cr, c)
        else:  # per-data-point groups: group id = data index
            gid = torch.arange(len(y))
            loss = es_loss(x, y, gid.repeat_interleave(r, 0), gid)
        g = torch.autograd.grad(loss, params)
        return torch.cat([q.flatten() for q in g])

    def stats(fn):
        t1 = time.time(); G = torch.stack([fn() for _ in range(a.K)]); dt = (time.time() - t1) / a.K
        mean = G.mean(0); trcov = float(((G - mean) ** 2).sum(1).mean() * a.K / (a.K - 1))
        return dict(trcov=trcov, mean_sq=float((mean ** 2).sum()), mean_grad_norm=float(G.norm(dim=1).mean()),
                    noise_to_signal=trcov / float((mean ** 2).sum()), s_per_grad=round(dt, 3))

    model.train()
    y0, c0 = batch()
    res = dict(warm_steps=a.warm, warm_seconds=round(warm_s, 1), K=a.K, batch="8 classes x 8 data points",
               loss_first10=float(np.mean(hist[:10])), loss_last10=float(np.mean(hist[-10:])), warm=warm, rows=[])
    for r in [1, 2, 4, 8]:
        row = dict(kind="z-noise, pooled per class", r=r, **stats(lambda: grad(y0, c0, r)))
        print(json.dumps(row), flush=True); res["rows"].append(row)
    row = dict(kind="z-noise, per-data-point pairs", r=2, **stats(lambda: grad(y0, c0, 2, mode="pairs")))
    print(json.dumps(row), flush=True); res["rows"].append(row)
    row = dict(kind="data+z noise, pooled per class", r=1, **stats(lambda: grad(*batch(), 1)))
    print(json.dumps(row), flush=True); res["rows"].append(row)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
