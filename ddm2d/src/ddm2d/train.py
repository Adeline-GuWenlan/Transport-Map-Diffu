"""Training loop for the three arms (docs/plan.md), with exact checkpoint/resume.

Settings from App. I.1 of arXiv 2502.02483 v3: Adam(1e-3, 0.9, 0.999, 1e-8), 100-step cosine warmup, global-norm clipping 1, EMA 0.99,
batch 128, 100k steps, flow-matching schedule, safety 1e-2, sigmoid weighting, m = 32.

All training randomness (model init, data order, t, forward noise, xi) comes from one torch.Generator seeded with the run seed; its state, the
data permutation and position, the optimizer, EMA, step, and the global torch/numpy/python RNG states are checkpointed, so a resumed run is
identical to an uninterrupted one (tests/test_train.py).

CLI: python -m ddm2d.train --config configs/paper/gmm_C.json --out runs/paper/gmm_C_s0 --seed 0 --threads 2 [--deadline UNIX_TS]
Exit code 0 = finished (DONE.json written); 3 = stopped at the deadline after checkpointing (resubmit to continue).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .losses import diffusion_loss, energy_loss
from .model import build_model, n_params
from .runinfo import environment, git_commit, peak_rss_mb
from .schedule import forward_sample, sample_train_times, sigmoid_weight

DEFAULTS = dict(dataset="gmm", arm="C", steps=100_000, batch=128, lr=1e-3, b1=0.9, b2=0.999, adam_eps=1e-8, warmup=100, clip=1.0,
                ema=0.99, safety=0.01, w_bias=0.0, m=32, beta=0.1, lam=1.0, train_size=D.TRAIN_SIZE, data_seed=0,
                ckpt_every=5000, log_every=100, keep_ckpts=2, ema_snap_every=0)


def resolve_config(cfg: dict) -> dict:
    out = dict(DEFAULTS)
    unknown = set(cfg) - set(DEFAULTS) - {"_comment", "name"}
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}")
    out.update(cfg)
    if out["arm"] == "A":
        out["m"], out["beta"], out["lam"] = 1, 2.0, 0.0     # Eq. (6); m, beta, lam unused
    return out


def lr_factor(step: int, warmup: int) -> float:
    """Cosine warmup over the first `warmup` iterations (step is 0-based), then constant."""
    if step >= warmup:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * (step + 1) / warmup))


class Trainer:
    def __init__(self, cfg: dict, seed: int, out: Path | None = None):
        self.cfg = resolve_config(cfg)
        self.seed = seed
        self.out = Path(out) if out is not None else None
        c = self.cfg
        self.gen = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)          # global RNG is not used for training randomness, but seed it anyway
        np.random.seed(seed)
        random.seed(seed)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = build_model(c["arm"])
        self.ema = copy.deepcopy(self.model)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=c["lr"], betas=(c["b1"], c["b2"]), eps=c["adam_eps"])
        self.x = torch.from_numpy(D.train_set(c["dataset"], c["train_size"], c["data_seed"]))
        self.data_fp = D.fingerprint(self.x.numpy())
        self.step = 0
        self.perm = torch.randperm(self.x.shape[0], generator=self.gen)
        self.pos = 0
        self.epoch = 0
        self.acc = self._zero_acc()
        self.wall_prev = 0.0

    # ---------------------------------------------------------------- data
    def next_batch(self):
        b = self.cfg["batch"]
        if self.pos + b > self.perm.shape[0]:
            self.perm = torch.randperm(self.x.shape[0], generator=self.gen)
            self.pos = 0
            self.epoch += 1
        idx = self.perm[self.pos:self.pos + b]
        self.pos += b
        return self.x[idx]

    # ---------------------------------------------------------------- one step
    def loss_on(self, x0):
        c, g = self.cfg, self.gen
        n = x0.shape[0]
        t = sample_train_times(n, c["safety"], generator=g)
        z = torch.randn(x0.shape, generator=g)
        xt = forward_sample(x0, t, z)
        w = sigmoid_weight(t, c["w_bias"])
        if c["arm"] == "A":
            return diffusion_loss(x0, self.model(t, xt), w)
        xi = torch.randn((n, c["m"], x0.shape[1]), generator=g)
        return energy_loss(x0, self.model(t, xt, xi), w, c["beta"], c["lam"])

    def train_step(self):
        c = self.cfg
        self.model.train()
        x0 = self.next_batch()
        loss, diag = self.loss_on(x0)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), c["clip"])
        lr = c["lr"] * lr_factor(self.step, c["warmup"])
        for pg in self.opt.param_groups:
            pg["lr"] = lr
        self.opt.step()
        with torch.no_grad():
            for pe, p in zip(self.ema.parameters(), self.model.parameters()):
                pe.mul_(c["ema"]).add_(p.detach(), alpha=1.0 - c["ema"])
        self.step += 1
        a = self.acc
        a["n"] += 1
        a["loss"] += float(loss.detach())
        a["conf"] += float(diag["conf"])
        a["inter"] += float(diag["inter"])
        a["gnorm"] += float(gnorm)
        a["gnorm_max"] = max(a["gnorm_max"], float(gnorm))
        a["finite"] = a["finite"] and math.isfinite(float(loss.detach())) and math.isfinite(float(gnorm))
        a["lr"] = lr
        return float(loss.detach())

    @staticmethod
    def _zero_acc():
        return {"n": 0, "loss": 0.0, "conf": 0.0, "inter": 0.0, "gnorm": 0.0, "gnorm_max": 0.0, "finite": True, "lr": 0.0}

    def flush_log(self, wall):
        a = self.acc
        if a["n"] == 0:
            return None
        rec = {"step": self.step, "loss": a["loss"] / a["n"], "conf": a["conf"] / a["n"], "inter": a["inter"] / a["n"],
               "gnorm_mean": a["gnorm"] / a["n"], "gnorm_max": a["gnorm_max"], "lr": a["lr"], "finite": a["finite"],
               "n_avg": a["n"], "wall_s": round(wall, 2), "epoch": self.epoch}
        if self.out is not None:
            with open(self.out / "log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        self.acc = self._zero_acc()
        return rec

    # ---------------------------------------------------------------- checkpoint
    def state_dict(self, wall):
        return {"model": self.model.state_dict(), "ema": self.ema.state_dict(), "opt": self.opt.state_dict(), "step": self.step,
                "gen": self.gen.get_state(), "perm": self.perm, "pos": self.pos, "epoch": self.epoch, "acc": self.acc,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                "cfg": self.cfg, "seed": self.seed, "wall": wall, "data_fp": self.data_fp}

    def load_state_dict(self, sd):
        assert sd["cfg"] == self.cfg, "checkpoint config differs from the run config"
        assert sd["seed"] == self.seed and sd["data_fp"] == self.data_fp
        self.model.load_state_dict(sd["model"])
        self.ema.load_state_dict(sd["ema"])
        self.opt.load_state_dict(sd["opt"])
        self.step, self.pos, self.epoch, self.acc = sd["step"], sd["pos"], sd["epoch"], dict(sd["acc"])
        self.gen.set_state(sd["gen"])
        self.perm = sd["perm"]
        torch.set_rng_state(sd["torch_rng"])
        np.random.set_state(sd["numpy_rng"])
        random.setstate(sd["python_rng"])
        self.wall_prev = sd["wall"]

    def save(self, path: Path, wall):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(self.state_dict(wall), tmp)
        os.replace(tmp, path)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ema_model(run_dir, step: int | None = None) -> torch.nn.Module:
    """EMA weights of a finished run (final.pt), else of the latest checkpoint; with `step`, the EMA snapshot ckpt/ema_<step>.pt."""
    run_dir = Path(run_dir)
    if step is not None:
        p = run_dir / "ckpt" / f"ema_{step:07d}.pt"
    else:
        p = run_dir / "ckpt" / "final.pt"
        if not p.exists():
            p = run_dir / "ckpt" / "latest.pt"
    sd = torch.load(p, map_location="cpu", weights_only=False)
    m = build_model(sd["cfg"]["arm"])
    m.load_state_dict(sd["ema"])
    m.eval()
    return m


def run(cfg: dict, out: Path, seed: int, threads: int, deadline: float | None, max_steps: int | None, argv=None) -> int:
    out.mkdir(parents=True, exist_ok=True)
    ck = out / "ckpt"
    tr = Trainer(cfg, seed, out)
    c = tr.cfg
    if (out / "DONE.json").exists():
        print(f"{out}: already DONE", flush=True)
        return 0
    latest = ck / "latest.pt"
    if latest.exists():
        tr.load_state_dict(torch.load(latest, map_location="cpu", weights_only=False))
        # drop log lines written after the checkpoint (they will be recomputed identically)
        if (out / "log.jsonl").exists():
            keep = [l for l in open(out / "log.jsonl") if json.loads(l)["step"] <= tr.step]
            open(out / "log.jsonl", "w").writelines(keep)
        print(f"resumed from step {tr.step}", flush=True)
    else:
        (out / "config.json").write_text(json.dumps(c, indent=1, sort_keys=True) + "\n")
    seg = {"start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "start_step": tr.step,
           "command": " ".join(shlex.quote(a) for a in (argv or sys.argv)), "git": git_commit(), "env": environment(threads),
           "seed": seed, "n_params": n_params(tr.model), "data_fingerprint": tr.data_fp}
    t0 = time.time()
    target = c["steps"] if max_steps is None else min(c["steps"], max_steps)
    status = "finished"
    while tr.step < target:
        tr.train_step()
        wall = tr.wall_prev + time.time() - t0
        if tr.step % c["log_every"] == 0:
            rec = tr.flush_log(wall)
            if not rec["finite"]:
                print(f"non-finite loss/grad at step {tr.step}", flush=True)
        if tr.step % c["ckpt_every"] == 0 or tr.step == target:
            tr.save(latest, wall)
            if tr.step % c["ckpt_every"] == 0:
                tr.save(ck / f"step_{tr.step:07d}.pt", wall)
                olds = sorted(ck.glob("step_*.pt"))[:-c["keep_ckpts"]]
                for o in olds:
                    o.unlink()
        if c["ema_snap_every"] and tr.step % c["ema_snap_every"] == 0:
            # small EMA-only snapshot kept for every interval (intermediate-checkpoint evaluation, S3); does not touch any RNG
            snap = ck / f"ema_{tr.step:07d}.pt"
            tmp = snap.with_name(snap.name + ".tmp")
            torch.save({"ema": tr.ema.state_dict(), "step": tr.step, "cfg": tr.cfg, "seed": tr.seed}, tmp)
            os.replace(tmp, snap)
        if deadline is not None and time.time() > deadline and tr.step < target:
            tr.save(latest, wall)          # the partial log window travels in the checkpoint ("acc")
            status = "deadline"
            break
    wall = tr.wall_prev + time.time() - t0
    seg.update(end_step=tr.step, segment_wall_s=round(time.time() - t0, 1), peak_rss_mb=round(peak_rss_mb(), 1), status=status)
    with open(out / "segments.jsonl", "a") as f:
        f.write(json.dumps(seg) + "\n")
    if status == "deadline":
        print(f"deadline reached at step {tr.step}; checkpointed", flush=True)
        return 3
    if tr.step >= c["steps"]:
        tr.save(ck / "final.pt", wall)
        segs = [json.loads(l) for l in open(out / "segments.jsonl")]
        done = {"steps": tr.step, "wall_s_total": round(wall, 1), "peak_rss_mb": max(s["peak_rss_mb"] for s in segs),
                "git": git_commit(), "seed": seed, "n_segments": len(segs), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "final_ckpt_sha256": sha256_file(ck / "final.pt")}
        (out / "ckpt_manifest.txt").write_text("".join(f"{sha256_file(p)}  ckpt/{p.name}\n" for p in sorted(ck.glob("*.pt"))))
        (out / "DONE.json").write_text(json.dumps(done, indent=1) + "\n")
        print(f"DONE {out} steps={tr.step} wall={wall:.0f}s", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--deadline", type=float, default=None, help="unix time; checkpoint and exit 3 after it")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (benchmarks/tests); no DONE.json unless steps reached")
    a = ap.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    torch.set_num_threads(a.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cfg = json.loads(Path(a.config).read_text())
    sys.exit(run(cfg, Path(a.out), a.seed, a.threads, a.deadline, a.max_steps, argv=sys.argv if argv is None else argv))


if __name__ == "__main__":
    main()
