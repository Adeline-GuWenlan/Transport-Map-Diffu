"""Stage-2 measurement: parameter counts, forward FLOPs and CPU train-step time of DiT configs
used as a transport map F(z, c) (learn_sigma=False, class_dropout_prob=0, constant timestep).

Run from the project root:
  PYTHONPATH=<timm dir> python docs/stage2_checks/time_dit.py [--quick]
Imports the unmodified reference DiT from third_party/DiT/models.py.
FLOPs: torch.utils.flop_counter.FlopCounterMode (counts 2 FLOPs per multiply-add) for all matmuls
and convs, plus the attention matmuls 2*T^2*D per layer added analytically (timm calls
F.scaled_dot_product_attention, which the counter does not see on CPU). We report GMACs = FLOPs / 2, which is the unit of the DiT paper's "Gflops" column (sanity-checked below
against the paper's 118.6 for DiT-XL/2 at 32x32x4).
"""
import warnings; warnings.filterwarnings("ignore")
import argparse, json, math, os, sys, time, statistics
import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "DiT"))
if os.environ.get("TIMM_DIR"):
    sys.path.insert(0, os.environ["TIMM_DIR"])
from models import DiT  # noqa: E402

SIZES = {  # from third_party/DiT/models.py (commit ed81ce2), plus two tiny configs for CPU
    "XL": dict(depth=28, hidden_size=1152, num_heads=16),
    "L": dict(depth=24, hidden_size=1024, num_heads=16),
    "B": dict(depth=12, hidden_size=768, num_heads=12),
    "S": dict(depth=12, hidden_size=384, num_heads=6),
    "T": dict(depth=6, hidden_size=256, num_heads=4),   # ours, not in DiT
    "U": dict(depth=4, hidden_size=192, num_heads=3),   # ours, not in DiT
}


def build(size, patch, input_size, in_ch, num_classes=10):
    return DiT(input_size=input_size, patch_size=patch, in_channels=in_ch, num_classes=num_classes,
               class_dropout_prob=0.0, learn_sigma=False, **SIZES[size])


def pow_dist(a, b):
    """exact ||a_i - b_j|| with gradient 0 at 0 (double-where), per-class blocks are small."""
    d = a[:, None, :] - b[None, :, :]
    s = (d * d).sum(-1)
    pos = s > 0
    return torch.where(pos, torch.where(pos, s, torch.ones_like(s)).sqrt(), torch.zeros_like(s))


def es_loss(x, y, labels):
    """eq. (9) of docs/energy_score_loss.tex, beta=1, per-class groups, m_c = n_c."""
    x, y = x.flatten(1), y.flatten(1)
    N = y.shape[0]
    loss = x.new_zeros(())
    for c in labels.unique():
        mk = labels == c
        xc, yc = x[mk], y[mk]
        m = xc.shape[0]
        conf = pow_dist(xc, yc).mean()
        dxx = pow_dist(xc, xc)
        inter = (dxx.sum() - dxx.diagonal().sum()) / (m * (m - 1))
        loss = loss + (yc.shape[0] / N) * (conf - 0.5 * inter)
    return loss


def gmacs(model, input_size, in_ch):
    z = torch.randn(1, in_ch, input_size, input_size)
    t = torch.zeros(1)
    c = torch.zeros(1, dtype=torch.long)
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(z, t, c)
    return fc.get_total_flops() / 2 / 1e9


def time_step(model, input_size, in_ch, batch, reps, warmup=3, num_classes=10):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    labels = torch.arange(batch) % num_classes            # class-stratified, n_c = m_c
    y = torch.rand(batch, in_ch, input_size, input_size) * 2 - 1
    t = torch.zeros(batch)                                 # constant timestep
    times = []
    for i in range(warmup + reps):
        t0 = time.perf_counter()
        z = torch.randn(batch, in_ch, input_size, input_size)
        x = model(z, t, labels)
        loss = es_loss(x, y, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        dt = time.perf_counter() - t0
        assert torch.isfinite(loss), loss
        if i >= warmup:
            times.append(dt)
    return statistics.median(times), min(times), max(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/stage2_checks/results/dit_sizes.json"))
    args = ap.parse_args()
    torch.manual_seed(0)
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}", flush=True)
    rows = []
    # (size, patch, input_size, in_ch, time_it)
    plan = [
        # sanity check vs DiT paper Table 4 (latent 32x32x4, 1000 classes)
        ("XL", 2, 32, 4, False), ("B", 2, 32, 4, False), ("S", 2, 32, 4, False),
        # pixel 32x32x3 (CIFAR-10), the GPU candidates
        ("XL", 2, 32, 3, False), ("L", 2, 32, 3, False), ("B", 2, 32, 3, False), ("S", 2, 32, 3, False),
        ("B", 4, 32, 3, False), ("S", 4, 32, 3, True),
        # CPU smoke candidates, 32x32x1 (MNIST / Fashion-MNIST padded) and 32x32x3
        ("T", 4, 32, 1, True), ("T", 4, 32, 3, True), ("U", 4, 32, 1, True), ("U", 4, 32, 3, True),
        ("T", 2, 32, 1, True), ("S", 4, 32, 1, True), ("T", 8, 32, 1, True),
    ]
    reps = 3 if args.quick else 10
    for size, p, H, C, do_time in plan:
        nc = 1000 if C == 4 else 10
        m = build(size, p, H, C, num_classes=nc)
        n_par = sum(q.numel() for q in m.parameters())
        n_train = sum(q.numel() for q in m.parameters() if q.requires_grad)
        g = gmacs(m, H, C)
        T_, D_, L_ = (H // p) ** 2, SIZES[size]["hidden_size"], SIZES[size]["depth"]
        attn = 2 * T_ * T_ * D_ * L_ / 1e9   # QK^T and AV; SDPA is not seen by FlopCounterMode on CPU
        row = dict(model=f"DiT-{size}/{p}", input=f"{C}x{H}x{H}", tokens=T_, classes=nc,
                   params_M=round(n_par / 1e6, 3), trainable_M=round(n_train / 1e6, 3),
                   counted_GMACs=round(g, 3), attn_GMACs=round(attn, 3), fwd_GMACs=round(g + attn, 3))
        if do_time:
            med, lo, hi = time_step(m, H, C, batch=64, reps=reps, num_classes=nc)
            row.update(step_s_b64_median=round(med, 3), step_s_min=round(lo, 3), step_s_max=round(hi, 3))
        print(json.dumps(row), flush=True)
        rows.append(row)
        del m
    meta = dict(torch=torch.__version__, threads=torch.get_num_threads(), host=os.uname().nodename,
                note="step = fwd + es_loss + bwd + clip + AdamW step, batch 64, 10 classes x 6.4, fp32")
    with open(args.out, "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
