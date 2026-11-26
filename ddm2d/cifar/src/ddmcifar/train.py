"""Training loop for the CIFAR-10 trial (ddm2dx E3), with exact checkpoint / resume.

Settings (prompts/_task.md E3; paper App. I.2 with the trial deviations marked there): Adam(lr 1e-4, 0.9, 0.999, 1e-8), global-norm
clip 1, 100-step cosine warmup then constant (ddm2d's lr_factor), flow-matching schedule alpha = 1 - t, sigma = t, t ~ U[safety,
1 - safety] with safety 1e-2, sigmoid weighting with bias 0 (ddm2d's sigmoid_weight) for every arm, batch 128 data points, m = 4
particles per data point for B / C, bf16 autocast for the networks (losses in float32), class-conditional (label always given).

EMA: decay_k = min(ema, (1 + k) / (10 + k)) at update k (the usual EMA warm-up; without it 37 % of the 10k-step EMA would still be the
random init at ema = 0.9999). Our choice, the same for every run.

Randomness: four CPU generators seeded from the run seed: data order + hflip (EpochSampler), t, forward noise X_1, xi. So every arm
with the same seed sees the same images, flips, t and X_1, step by step; only B / C draw xi. Model init uses torch.manual_seed(seed).
All generator states, the sampler state, optimizer, EMA, step and the torch CPU / CUDA RNG states are checkpointed:
a resumed run is bitwise identical to an uninterrupted one on CPU (tests/test_cifar.py::test_resume_bitwise).

Exact chunked gradient (grad_chunk > 0): pass 1 computes all M network outputs (and features) without a graph, the loss on detached
copies and its gradient w.r.t. them; pass 2 recomputes each chunk with a graph and back-propagates <out_k, dL/dout_k>
(+ <phi(out_k), dL/dphi_k>). This is the chain rule for dL/dtheta, as transport_map's tmap.train.forward_backward. No dropout, so pass 2
reproduces pass 1 (up to bf16 kernel non-determinism on GPU).

Signals: SIGUSR1 / SIGTERM -> checkpoint ckpt/latest.pt and exit 3 (the Slurm script requeues). Snapshots ckpt/snap_<step>.pt (EMA and
raw weights) at cfg["snapshots"]. Logs: log.jsonl.

CLI: python -m ddmcifar.train --config cifar/configs/cifar_B.json --out cifar/runs/cifar_B [--max-steps N]
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
from pathlib import Path

import torch

from ddm2d.schedule import sample_train_times, sigmoid_weight

from .data import EpochSampler, load_cifar
from .losses import energy_loss, velocity_loss
from .unet import build_unet, n_params

DEFAULTS = dict(arm="B", seed=0, steps=30_000, batch=128, m=4, beta=2.0, lam=0.0, rho="l2pow", ph_c=0.0, lr=1e-4, b1=0.9,
                b2=0.999, adam_eps=1e-8, warmup=100, clip=1.0, ema=0.9999, safety=0.01, w_bias=0.0, hflip=True, bf16=True,
                tf32=True, grad_chunk=0, feature=None, snapshots=[10_000, 20_000, 30_000], ckpt_every=1000, log_every=50,
                model={}, data_n=None, device="cuda")

_STOP = {"flag": False, "sig": None}


def _on_signal(signum, _frame):
    _STOP["flag"], _STOP["sig"] = True, signum


def resolve_config(cfg: dict) -> dict:
    unknown = set(cfg) - set(DEFAULTS) - {"_comment", "name"}
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}")
    out = copy.deepcopy(DEFAULTS)
    out.update(copy.deepcopy(cfg))
    if out["arm"] == "A":
        out["m"], out["beta"], out["lam"], out["rho"], out["feature"] = 1, 2.0, 0.0, "l2pow", None
    return out


def lr_factor(step: int, warmup: int) -> float:
    """Cosine warmup over the first `warmup` iterations (0-based step), then constant (copied from ddm2d.train.lr_factor)."""
    if step >= warmup:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * (step + 1) / warmup))


def ema_decay(k: int, ema: float) -> float:
    return min(ema, (1.0 + k) / (10.0 + k))


class Trainer:
    def __init__(self, cfg: dict, out: Path | None = None, data=None):
        self.cfg = c = resolve_config(cfg)
        self.out = Path(out) if out is not None else None
        self.device = torch.device(c["device"] if (c["device"] != "cuda" or torch.cuda.is_available()) else "cpu")
        self.bf16 = bool(c["bf16"]) and self.device.type == "cuda"
        if c["tf32"] and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        seed = c["seed"]
        torch.manual_seed(seed)
        self.model = build_unet(c["arm"], c["model"]).to(self.device)
        self.ema = copy.deepcopy(self.model).requires_grad_(False)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=c["lr"], betas=(c["b1"], c["b2"]), eps=c["adam_eps"])
        x, y = data if data is not None else load_cifar("train", n=c["data_n"])
        self.x, self.y = x, y
        self.sampler = EpochSampler(x, y, c["batch"], seed=seed * 1000 + 101, hflip=c["hflip"])
        self.g_t = torch.Generator().manual_seed(seed * 1000 + 102)
        self.g_noise = torch.Generator().manual_seed(seed * 1000 + 103)
        self.g_xi = torch.Generator().manual_seed(seed * 1000 + 104)
        self.step, self.wall_prev = 0, 0.0
        self.enc, self.feat = None, None
        fc = c["feature"]
        if fc:
            from .features import build_encoder
            self.enc = build_encoder(fc).to(self.device)
        self.acc = self._zero_acc()

    # ------------------------------------------------------------------ feature scale (once per run, stored in ckpt)
    def init_feature_scale(self):
        fc = self.cfg["feature"]
        if not fc or self.feat is not None:
            return
        from .features import feature_scale
        if fc.get("scale", "auto") == "auto":
            self.feat = feature_scale(self.enc, self.x, self.y, self.cfg["beta"], fc.get("beta", 1.0),
                                      int(fc.get("scale_n_per_class", 200)), self.device, self.bf16)
        else:
            self.feat = {"scale": float(fc["scale"])}
        self.feat["weight"] = float(fc.get("weight", 1.0))
        self.feat["lambda_f"] = self.feat["weight"] * self.feat["scale"]

    # ------------------------------------------------------------------ one step
    def _fwd(self, t, x, c, xi):
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            return self.model(t, x, c, xi) if xi is not None else self.model(t, x, c)

    def _enc(self, x):
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            return self.enc(x)

    def draw(self):
        """One step's data and randomness (CPU generators), moved to the device."""
        c, dev = self.cfg, self.device
        x0, lab = self.sampler.next()
        n = len(x0)
        t = sample_train_times(n, c["safety"], generator=self.g_t)
        x1 = torch.randn(x0.shape, generator=self.g_noise)
        xi = torch.randn((n * c["m"], *x0.shape[1:]), generator=self.g_xi) if c["arm"] != "A" else None
        mv = lambda v: v.to(dev, non_blocking=True) if v is not None else None
        return mv(x0), mv(lab), mv(t), mv(x1), mv(xi)

    def loss_and_grad(self, x0, lab, t, x1, xi):
        """Loss and parameter gradients (accumulated into .grad). Returns (loss, diag)."""
        c = self.cfg
        n, m = len(x0), c["m"]
        tb = t.view(-1, 1, 1, 1)
        xt = (1 - tb) * x0 + tb * x1
        w = sigmoid_weight(t, c["w_bias"])
        if c["arm"] == "A":
            inputs = (t, xt, lab, None)
        else:
            inputs = (t.repeat_interleave(m), xt.repeat_interleave(m, 0), lab.repeat_interleave(m), xi)
        M = len(inputs[0])
        f0 = None
        if self.enc is not None:
            with torch.no_grad():
                f0 = self._enc(x0).float()

        def loss_fn(out, fout):
            if c["arm"] == "A":
                return velocity_loss(x0 - x1, out, w)
            xh = out.float().view(n, m, *x0.shape[1:])
            loss, d = energy_loss(x0, xh, w, c["lam"], c["rho"], c["beta"], c["ph_c"])
            if fout is not None:
                fc = c["feature"]
                lf, df = energy_loss(f0, fout.float().view(n, m, -1), w, float(fc.get("lam", 1.0)), "l2pow",
                                     float(fc.get("beta", 1.0)))
                d["pixel_loss"], d["feat_loss"] = loss.detach(), lf.detach()
                d["feat_conf"], d["feat_inter"] = df["conf"], df["inter"]
                loss = loss + self.feat["lambda_f"] * lf
            return loss, d

        cb = c["grad_chunk"] if 0 < c["grad_chunk"] < M else M
        sl = lambda v, i: v[i:i + cb] if v is not None else None
        if cb == M:
            out = self._fwd(*inputs)
            fout = self._enc(out) if self.enc is not None else None
            loss, d = loss_fn(out, fout)
            loss.backward()
            return loss.detach(), d
        idx = range(0, M, cb)
        with torch.no_grad():
            out = torch.cat([self._fwd(*[sl(v, i) for v in inputs]) for i in idx]).float()
            fout = torch.cat([self._enc(out[i:i + cb]) for i in idx]).float() if self.enc is not None else None
        out.requires_grad_(True)
        if fout is not None:
            fout.requires_grad_(True)
        loss, d = loss_fn(out, fout)
        loss.backward()
        go, gf = out.grad, (fout.grad if fout is not None else None)
        for i in idx:
            oc = self._fwd(*[sl(v, i) for v in inputs])
            sur = (oc.float() * go[i:i + cb]).sum()
            if self.enc is not None:
                sur = sur + (self._enc(oc).float() * gf[i:i + cb]).sum()
            sur.backward()
        return loss.detach(), d

    def train_step(self):
        c = self.cfg
        self.model.train()
        batch = self.draw()
        self.opt.zero_grad(set_to_none=True)
        loss, d = self.loss_and_grad(*batch)
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), c["clip"])
        lr = c["lr"] * lr_factor(self.step, c["warmup"])
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        self.opt.step()
        dec = ema_decay(self.step, c["ema"])
        with torch.no_grad():
            for pe, p in zip(self.ema.parameters(), self.model.parameters()):
                pe.mul_(dec).add_(p.detach(), alpha=1.0 - dec)
        self.step += 1
        a = self.acc
        lv, gv = float(loss), float(gn)
        a["n"] += 1
        a["loss"] += lv
        a["gnorm"] += gv
        a["gnorm_max"] = max(a["gnorm_max"], gv)
        a["clipped"] += int(gv > c["clip"])
        a["finite"] = a["finite"] and math.isfinite(lv) and math.isfinite(gv)
        for k, v in d.items():
            a["d_" + k] = a.get("d_" + k, 0.0) + float(v)
        a["lr"] = lr
        return lv

    @staticmethod
    def _zero_acc():
        return {"n": 0, "loss": 0.0, "gnorm": 0.0, "gnorm_max": 0.0, "clipped": 0, "finite": True, "lr": 0.0}

    def flush_log(self, wall, extra=None):
        a = self.acc
        if a["n"] == 0:
            return None
        rec = {"step": self.step, "loss": a["loss"] / a["n"], "gnorm_mean": a["gnorm"] / a["n"], "gnorm_max": a["gnorm_max"],
               "frac_clipped": a["clipped"] / a["n"], "lr": a["lr"], "finite": a["finite"], "n_avg": a["n"],
               "wall_s": round(wall, 2), "epoch": self.sampler.epoch}
        rec.update({k[2:]: v / a["n"] for k, v in a.items() if k.startswith("d_")})
        if self.device.type == "cuda":
            rec["max_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        rec.update(extra or {})
        if self.out is not None:
            with open(self.out / "log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        self.acc = self._zero_acc()
        return rec

    # ------------------------------------------------------------------ checkpoint
    def state_dict(self, wall):
        sd = {"model": self.model.state_dict(), "ema": self.ema.state_dict(), "opt": self.opt.state_dict(), "step": self.step,
              "sampler": self.sampler.state_dict(), "g_t": self.g_t.get_state(), "g_noise": self.g_noise.get_state(),
              "g_xi": self.g_xi.get_state(), "torch_rng": torch.get_rng_state(), "acc": dict(self.acc), "cfg": self.cfg,
              "wall": wall, "feat": self.feat}
        if torch.cuda.is_available():
            sd["cuda_rng"] = torch.cuda.get_rng_state_all()
        return sd

    def load_state_dict(self, sd):
        assert sd["cfg"] == self.cfg, "checkpoint config differs from the run config"
        self.model.load_state_dict(sd["model"])
        self.ema.load_state_dict(sd["ema"])
        self.opt.load_state_dict(sd["opt"])
        self.step, self.acc, self.wall_prev, self.feat = sd["step"], dict(sd["acc"]), sd["wall"], sd["feat"]
        self.sampler.load_state_dict(sd["sampler"])
        self.g_t.set_state(sd["g_t"])
        self.g_noise.set_state(sd["g_noise"])
        self.g_xi.set_state(sd["g_xi"])
        torch.set_rng_state(sd["torch_rng"].cpu())
        if "cuda_rng" in sd and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in sd["cuda_rng"]])

    def save(self, path: Path, wall):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(self.state_dict(wall), tmp)
        os.replace(tmp, path)

    def save_snapshot(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save({"ema": self.ema.state_dict(), "model": self.model.state_dict(), "step": self.step, "cfg": self.cfg,
                    "feat": self.feat}, tmp)
        os.replace(tmp, path)


def load_model(path, weights: str = "ema", device="cpu"):
    """A DDMUNet with the EMA (or raw) weights of a checkpoint or snapshot. Returns (model, cfg, step)."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    m = build_unet(sd["cfg"]["arm"], sd["cfg"]["model"])
    m.load_state_dict(sd[weights])
    return m.to(device).eval(), sd["cfg"], sd["step"]


def run(cfg: dict, out: Path, max_steps: int | None = None, data=None) -> int:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ck = out / "ckpt"
    if (out / "DONE.json").exists():
        print(f"{out}: already DONE", flush=True)
        return 0
    if torch.cuda.is_available() is False or cfg.get("device") == "cpu":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", len(os.sched_getaffinity(0)))))
    tr = Trainer(cfg, out, data)
    c = tr.cfg
    latest = ck / "latest.pt"
    if latest.exists():
        tr.load_state_dict(torch.load(latest, map_location="cpu", weights_only=False))
        if (out / "log.jsonl").exists():
            keep = [l for l in open(out / "log.jsonl") if json.loads(l)["step"] <= tr.step]
            open(out / "log.jsonl", "w").writelines(keep)
        print(f"resumed from {latest} at step {tr.step}", flush=True)
    else:
        (out / "config.json").write_text(json.dumps(c, indent=1, sort_keys=True) + "\n")
    tr.init_feature_scale()
    if tr.feat is not None:
        print(f"feature term: {c['feature']} -> {tr.feat}", flush=True)
    print(f"arm {c['arm']} beta {c['beta']} lam {c['lam']} rho {c['rho']} m {c['m']}; {n_params(tr.model) / 1e6:.3f} M params; "
          f"device {tr.device} bf16 {tr.bf16} chunk {c['grad_chunk']}; train {len(tr.x)} images; out {out}", flush=True)
    for s in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(s, _on_signal)
    target = c["steps"] if max_steps is None else min(c["steps"], max_steps)
    snaps = {int(s) for s in c["snapshots"]}
    t0, t_log, s_log = time.time(), time.time(), tr.step
    code = 0
    while tr.step < target:
        if _STOP["flag"]:
            print(f"signal {_STOP['sig']}: checkpoint at step {tr.step}, exit 3", flush=True)
            code = 3
            break
        tr.train_step()
        wall = tr.wall_prev + time.time() - t0
        if tr.step % c["log_every"] == 0:
            if tr.device.type == "cuda":
                torch.cuda.synchronize()
            now = time.time()
            rec = tr.flush_log(wall, {"s_per_step": (now - t_log) / max(1, tr.step - s_log)})
            t_log, s_log = now, tr.step
            if rec["step"] % (c["log_every"] * 10) == 0 or not rec["finite"]:
                print(json.dumps(rec), flush=True)
            if not rec["finite"]:
                print(f"non-finite loss/grad at step {tr.step}", flush=True)
        if tr.step in snaps:
            tr.save_snapshot(ck / f"snap_{tr.step:07d}.pt")
        if tr.step % c["ckpt_every"] == 0 or tr.step == target:
            tr.save(latest, wall)
    wall = tr.wall_prev + time.time() - t0
    tr.save(latest, wall)
    if code == 0 and tr.step >= c["steps"]:
        (out / "DONE.json").write_text(json.dumps({"step": tr.step, "wall_s": wall,
                                                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    print(f"stopped at step {tr.step}, wall {wall / 3600:.2f} h", flush=True)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="top-level overrides key=json_value (smoke / bench only)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    for ov in a.set:
        k, v = ov.split("=", 1)
        cfg[k] = json.loads(v)
    sys.exit(run(cfg, Path(a.out), a.max_steps))


if __name__ == "__main__":
    main()
