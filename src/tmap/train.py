"""Train a DiT transport map F(z, c) with the energy-score loss (docs/energy_score_loss.tex, eq. (9)).

    python -m tmap.train --config configs/smoke_cpu.yaml --out runs/<date>_<name> [--set train.time_limit_min=60 ...]

Resume: if <out>/ckpt/latest.pt exists, training continues from it (model, EMA, optimizer, step, data-sampler and
noise RNG states), so re-running the same command after a crash or Slurm requeue is safe. SIGTERM / SIGUSR1 make it
write a checkpoint and exit with code 3 (the Slurm script requeues on that).

Logs: <out>/metrics.jsonl (every log_every steps: loss, confinement, interaction, grad norm before clipping, lr,
steps/s, images/s), <out>/eval.jsonl (held-out per-class D_1, raw and EMA weights), <out>/samples/*.png.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import sys
import time

import torch
import yaml

from .data import ClassStratifiedSampler, load_split, resolve
from .energy_score import energy_score_loss
from .evaluate import heldout_energy_distance, reference_set
from .model import build_model, count_params
from .sample import fixed_noise, generate, save_grid

_STOP = {"flag": False, "sig": None}


def _on_signal(signum, _frame):
    _STOP["flag"], _STOP["sig"] = True, signum


def load_config(path: str, overrides=()) -> dict:
    with open(resolve(path)) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        key, val = ov.split("=", 1)
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)
    return cfg


def lr_at(step: int, oc: dict) -> float:
    w = oc.get("warmup_steps", 0)
    base = oc["lr"]
    f = min(1.0, (step + 1) / w) if w > 0 else 1.0
    if oc.get("schedule", "constant") == "cosine":
        T = oc["total_steps"]
        f *= 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, step - w) / max(1, T - w))))
    return base * f


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def loss_terms(x, fx, y, fy, yl, xl, lc: dict, lam: float):
    """Pixel energy-score loss (stage 1 eq. (9)) plus, if fx is given, lam * the same estimator on features."""
    beta, w = lc.get("beta", 1.0), lc.get("interaction_weight", 0.5)
    loss, parts = energy_score_loss(x, y, yl, xl, beta=beta, interaction_weight=w, return_parts=True)
    parts["pixel_loss"] = loss.detach()
    if fx is not None:
        lf, pf = energy_score_loss(fx, fy, yl, xl, beta=beta, interaction_weight=w, return_parts=True)
        loss = loss + lam * lf
        parts["feat_loss"] = lf.detach()
        parts["feat_confinement"], parts["feat_interaction"] = pf["confinement"], pf["interaction"]
    return loss, parts


def forward_backward(model, enc, z, xl, y, yl, lc: dict, lam: float, bf16: bool, chunk: int = 0):
    """Loss and parameter gradients for one batch (gradients accumulate into .grad).

    chunk <= 0: one pass (activations of all M samples are kept for backward).
    chunk > 0: exact two-pass gradient with activation memory of `chunk` samples. Pass 1 computes x = F(z, c) and
    phi(x) for the whole batch without a graph, then the loss and dL/dx, dL/dphi(x) on detached copies. Pass 2
    recomputes each chunk with a graph and backpropagates <F(z_k, c_k), dL/dx_k> + <phi(F(z_k, c_k)), dL/dphi_k>,
    whose parameter gradient is the chain rule for dL/dtheta. F has no dropout, so pass 2 reproduces pass 1's x.
    Cost: one extra forward of F (and of phi), about +1/3 compute.
    """
    dev_type = z.device.type

    def fwd(zc, cc):
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=bf16):
            return model(zc, cc)

    def enc_fwd(xc):
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=bf16):
            return enc(xc)

    M = len(z)
    cb = chunk if 0 < chunk < M else M
    fy = None
    if enc is not None:
        with torch.no_grad():
            fy = torch.cat([enc_fwd(y[i:i + cb]) for i in range(0, len(y), cb)]).float()
    if cb == M:
        x = fwd(z, xl)
        fx = enc_fwd(x) if enc is not None else None
        loss, parts = loss_terms(x, fx, y, fy, yl, xl, lc, lam)
        loss.backward()
        return loss.detach(), parts
    idx = range(0, M, cb)
    with torch.no_grad():
        x = torch.cat([fwd(z[i:i + cb], xl[i:i + cb]) for i in idx]).float()
        fx = torch.cat([enc_fwd(x[i:i + cb]) for i in idx]).float() if enc is not None else None
    xd = x.requires_grad_(True)
    fxd = fx.requires_grad_(True) if fx is not None else None
    loss, parts = loss_terms(xd, fxd, y, fy, yl, xl, lc, lam)
    loss.backward()
    gx, gf = xd.grad, (fxd.grad if fxd is not None else None)
    for i in idx:
        xc = fwd(z[i:i + cb], xl[i:i + cb])
        sur = (xc.float() * gx[i:i + cb]).sum()
        if enc is not None:
            sur = sur + (enc_fwd(xc).float() * gf[i:i + cb]).sum()
        sur.backward()
    return loss.detach(), parts


@torch.no_grad()
def feature_scale(enc, data, n_per_class: int, beta: float, device, bf16: bool) -> dict:
    """s = (mean same-class ||y - y'|| in pixels) / (the same in feature space), on the first n_per_class train
    images of every class. lam = loss.feature.weight * s puts the feature term on the pixel term's scale."""
    from .energy_score import _offdiag_mean, pairwise_dist_eval
    from .features import encode
    px, ft = [], []
    for c in data.classes:
        yc = data.class_tensor(c, n_per_class)
        fc = encode(enc, yc.to(device), bf16=bf16)
        px.append(float(_offdiag_mean(pairwise_dist_eval(yc, yc, beta))))
        ft.append(float(_offdiag_mean(pairwise_dist_eval(fc, fc, beta))))
    p, f = sum(px) / len(px), sum(ft) / len(ft)
    return {"scale": p / f, "pixel_mean_dist": p, "feature_mean_dist": f, "n_per_class": n_per_class}


def save_ckpt(path, **state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def train(cfg: dict, out: str, resume: bool = True) -> dict:
    out = resolve(out)
    os.makedirs(os.path.join(out, "samples"), exist_ok=True)
    tc, oc, bc, lc = cfg["train"], cfg["optim"], cfg["batch"], cfg["loss"]
    device = torch.device(cfg.get("device", "cpu"))
    if device.type == "cpu":
        torch.set_num_threads(int(cfg.get("num_threads") or os.environ.get("OMP_NUM_THREADS")
                                  or len(os.sched_getaffinity(0))))
    if tc.get("tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    bf16 = tc.get("precision", "fp32") == "bf16" and device.type == "cuda"
    if lc.get("grouping", "per_class") != "per_class" or lc.get("statistic", "u") != "u" \
            or lc.get("norm", "double_where") != "double_where":
        raise ValueError("only grouping=per_class, statistic=u, norm=double_where are implemented (stage 1 spec)")
    r = int(lc.get("samples_per_datapoint", 1))

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    model = build_model(cfg).to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=oc["lr"], betas=tuple(oc.get("betas", (0.9, 0.999))),
                            eps=oc.get("eps", 1e-8), weight_decay=oc.get("weight_decay", 0.0))
    train_data = load_split(cfg, "train_split")
    sampler = ClassStratifiedSampler(train_data, bc["classes_per_batch"], bc["n_per_class"], seed=seed,
                                     hflip=cfg["data"].get("hflip", False))
    zgen = torch.Generator(device=device).manual_seed(seed + 1)

    ec = cfg.get("eval", {}).get("energy_distance", {})
    held = load_split(cfg, "heldout_split")
    n_eval = int(ec.get("n_per_class", 500))
    ref_x, ref_l = reference_set(held, n_eval)
    n_eval = min(n_eval, int(torch.bincount(ref_l).max()))
    yy_cache = {}
    grid_z, grid_c = fixed_noise(model, train_data.classes, 8, seed=1234)

    fcfg = lc.get("feature") or {}
    enc, feat = None, None
    if fcfg.get("weight", 0) > 0:
        from .features import build_feature_encoder
        enc = build_feature_encoder(fcfg, cfg["data"]["channels"]).to(device)

    step = 0
    ck_path = os.path.join(out, "ckpt", "latest.pt")
    if resume and os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        feat = ck.get("feat")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sampler.load_state_dict(ck["sampler"])
        zgen.set_state(ck["zgen"].cpu())
        torch.set_rng_state(ck["torch_rng"].cpu())   # map_location may have moved it to the GPU
        step = ck["step"]
        print(f"resumed from {ck_path} at step {step}", flush=True)
    else:
        with open(os.path.join(out, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    lam = 0.0
    if enc is not None:
        if feat is None:   # fixed once per run (stored in the checkpoint, so a resumed run keeps the same lambda)
            if fcfg.get("scale", "auto") == "auto":
                feat = feature_scale(enc, train_data, int(fcfg.get("scale_n_per_class", 200)), lc.get("beta", 1.0),
                                     device, bf16)
            else:
                feat = {"scale": float(fcfg["scale"])}
            feat["weight"] = float(fcfg["weight"])
            feat["lambda"] = feat["weight"] * feat["scale"]
            jsonl(os.path.join(out, "feature_scale.jsonl"), feat)
        lam = feat["lambda"]
        print(f"feature term: encoder {fcfg.get('encoder')} ({enc.dim}-d), lambda = {lam:.4g} ({feat})", flush=True)
    chunk = int(tc.get("grad_chunk", 0))
    snap_steps = {int(s) for s in tc.get("snapshot_steps", [])}
    print(f"model {count_params(model) / 1e6:.3f} M params; train {len(train_data)} images, classes "
          f"{train_data.classes}; batch {bc['classes_per_batch']} x {bc['n_per_class']} (r = {r}); device {device}, "
          f"threads {torch.get_num_threads()}, bf16 {bf16}; out {out}", flush=True)

    def checkpoint():
        save_ckpt(ck_path, model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                  sampler=sampler.state_dict(), zgen=zgen.get_state(), torch_rng=torch.get_rng_state(),
                  step=step, config=cfg, feat=feat)

    def snapshot():
        """Weights-only copy at a fixed step (latest.pt is overwritten), for evaluation after training."""
        save_ckpt(os.path.join(out, "ckpt", f"step{step:07d}.pt"), model=model.state_dict(), ema=ema.state_dict(),
                  step=step, config=cfg, feat=feat)

    def do_eval_and_samples(tag_step, do_eval=True, do_samples=True):
        rec = {"step": tag_step, "time": time.time()}
        for name, net in (("raw", model), ("ema", ema)):
            if do_samples:
                x = generate(net, grid_c, grid_z, autocast_bf16=bf16)
                save_grid(x, os.path.join(out, "samples", f"step{tag_step:07d}_{name}.png"), nrow=8)
            if do_eval:
                res, x, _ = heldout_energy_distance(net, ref_x, ref_l, n_eval, seed=0, beta=ec.get("beta", 1.0),
                                                    yy_cache=yy_cache, bf16=bf16)
                rec[f"{name}_ed_mean"] = res["mean"]
                rec[f"{name}_ed_per_class"] = res["per_class"]
                rec[f"{name}_diversity_ratio"] = sum(res["within_x"][c] / res["within_y"][c]
                                                     for c in res["within_x"]) / len(res["within_x"])
        if do_eval:
            rec["n_per_class"] = n_eval
            jsonl(os.path.join(out, "eval.jsonl"), rec)
            print(f"[eval] step {tag_step}: held-out D1 raw {rec['raw_ed_mean']:.4f}  ema {rec['ema_ed_mean']:.4f}  "
                  f"diversity raw {rec['raw_diversity_ratio']:.3f} ema {rec['ema_diversity_ratio']:.3f}", flush=True)
        return rec

    for s in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(s, _on_signal)
    max_steps = int(tc.get("max_steps", 10 ** 9))
    t_limit = float(tc.get("time_limit_min", 0)) * 60 or float("inf")
    log_every, ck_every = int(tc.get("log_every", 50)), int(tc.get("ckpt_every", 1000))
    s_every = int(tc.get("sample_every", 2000))
    e_every = int(cfg.get("eval", {}).get("every", 2000))
    if step == 0:
        do_eval_and_samples(0)
    t_start = time.time()
    t_log, imgs_since, steps_since = time.time(), 0, 0
    acc = {"loss": 0.0, "conf": 0.0, "inter": 0.0, "gn": 0.0, "gn_max": 0.0, "pix": 0.0, "feat": 0.0}
    model.train()
    exit_code = 0
    while step < max_steps:
        if time.time() - t_start > t_limit:
            print(f"time limit {t_limit / 60:.1f} min reached at step {step}", flush=True)
            break
        if _STOP["flag"]:
            print(f"signal {_STOP['sig']}: checkpointing at step {step} and exiting", flush=True)
            exit_code = 3
            break
        lr = lr_at(step, oc)
        for g in opt.param_groups:
            g["lr"] = lr
        y, yl = sampler.next()
        y, yl = y.to(device, non_blocking=True), yl.to(device, non_blocking=True)
        xl = yl.repeat_interleave(r) if r > 1 else yl
        z = torch.randn((len(xl), *y.shape[1:]), generator=zgen, device=device)
        opt.zero_grad(set_to_none=True)
        loss, parts = forward_backward(model, enc, z, xl, y, yl, lc, lam, bf16, chunk)
        gn =torch.nn.utils.clip_grad_norm_(params, oc.get("grad_clip_norm") or float("inf"))
        if not torch.isfinite(loss) or not torch.isfinite(gn):
            raise FloatingPointError(f"non-finite loss {loss.item()} or grad norm {gn.item()} at step {step}")
        opt.step()
        ema_update(ema, model, oc.get("ema_decay", 0.999))
        step += 1
        acc["loss"] += loss.item(); acc["conf"] += parts["confinement"].item(); acc["inter"] += parts["interaction"].item()
        acc["gn"] += gn.item(); acc["gn_max"] = max(acc["gn_max"], gn.item())
        acc["pix"] += parts["pixel_loss"].item(); acc["feat"] += parts["feat_loss"].item() if "feat_loss" in parts else 0.0
        imgs_since += len(xl); steps_since += 1
        if step % log_every == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t_log
            rec = {"step": step, "loss": acc["loss"] / steps_since, "confinement": acc["conf"] / steps_since,
                   "interaction": acc["inter"] / steps_since, "grad_norm": acc["gn"] / steps_since,
                   "grad_norm_max": acc["gn_max"], "lr": lr, "steps_per_s": steps_since / dt,
                   "imgs_per_s": imgs_since / dt, "elapsed_min": (time.time() - t_start) / 60, "time": time.time()}
            if enc is not None:
                rec["pixel_loss"] = acc["pix"] / steps_since
                rec["feat_loss"] = acc["feat"] / steps_since       # feature-space ES, before the lambda weight
            if device.type == "cuda":
                rec["max_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
            jsonl(os.path.join(out, "metrics.jsonl"), rec)
            if step % (log_every * 10) == 0:
                print(f"step {step}  loss {rec['loss']:.4f}  conf {rec['confinement']:.4f}  inter {rec['interaction']:.4f}"
                      f"  |g| {rec['grad_norm']:.2f}  lr {lr:.2e}  {rec['steps_per_s']:.2f} it/s  "
                      f"{rec['elapsed_min']:.1f} min", flush=True)
            acc = {k: 0.0 for k in acc}
            t_log, imgs_since, steps_since = time.time(), 0, 0
        if step % ck_every == 0:
            checkpoint()
        if step in snap_steps:
            snapshot()
        if step % e_every == 0 or step % s_every == 0:
            do_eval_and_samples(step, do_eval=step % e_every == 0, do_samples=step % s_every == 0)
            t_log = time.time()   # do not count eval time in throughput
    checkpoint()
    already = step % e_every == 0 and step % s_every == 0
    final = do_eval_and_samples(step) if exit_code == 0 and not already else {"step": step}
    print(f"done: step {step}, {(time.time() - t_start) / 60:.1f} min; checkpoint {ck_path}", flush=True)
    final["exit_code"] = exit_code
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.sub=value (YAML values)")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    res = train(cfg, a.out, resume=not a.no_resume)
    sys.exit(res.get("exit_code", 0))


if __name__ == "__main__":
    main()
