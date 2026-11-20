"""S4 DIAGNOSTIC (not a paper run; never in Table 2 / Fig. 3): does the low-noise accuracy floor of the 100k-step networks come from
optimisation noise (constant lr 1e-3, EMA 0.99 averages only ~100 steps)?

Starts from runs/paper/<run>/ckpt/final.pt (model, EMA, Adam state, RNG, data order) and continues the SAME training for --steps more steps
with only the learning rate changed (constant --lr; no warmup). Everything else (loss, batch, clipping, EMA 0.99, data, seeds) is unchanged.
Output: runs/diag/<run>_lr<lr>_<steps>/ckpt/final.pt (gitignored), log.jsonl, DONE.json. Apply identically to every arm compared.

Usage: python scripts/diag_finetune.py --run cb_A_s0 --lr 1e-4 --steps 10000 --threads 2
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import torch
    torch.set_num_threads(a.threads)
    from ddm2d.train import Trainer, git_commit, sha256_file

    src = ROOT / "runs" / "paper" / a.run / "ckpt" / "final.pt"
    sd = torch.load(src, map_location="cpu", weights_only=False)
    tr = Trainer(sd["cfg"], sd["seed"])
    tr.load_state_dict(sd)
    start = tr.step
    tr.cfg["lr"] = a.lr                 # the only change (lr_factor is 1 after warmup)
    out = ROOT / "runs" / "diag" / f"{a.run}_lr{a.lr:g}_{a.steps}"
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "w")
    t0 = time.time()
    while tr.step < start + a.steps:
        tr.train_step()
        if tr.step % 100 == 0:
            rec = tr.flush_log(time.time() - t0)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    tr.save(out / "ckpt" / "final.pt", time.time() - t0)
    done = {"diagnostic": True, "from": str(src.relative_to(ROOT)), "from_sha256": sha256_file(src), "start_step": start, "steps": tr.step,
            "lr": a.lr, "wall_s": round(time.time() - t0, 1), "git": git_commit(), "final_ckpt_sha256": sha256_file(out / "ckpt" / "final.pt")}
    (out / "DONE.json").write_text(json.dumps(done, indent=1) + "\n")
    print("done", out, done["wall_s"], flush=True)


if __name__ == "__main__":
    main()
