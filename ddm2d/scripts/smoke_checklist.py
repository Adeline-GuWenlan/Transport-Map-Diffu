"""Evidence for _task.md section 5 items 1, 2, 3 and 8 (smoke). Writes runs/smoke/checklist/{env.json, datasets.png/.pdf, models.json, resume.json}."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "4")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
from ddm2d import data as D  # noqa: E402
from ddm2d.model import build_model, n_params  # noqa: E402
from ddm2d.runinfo import environment, git_commit  # noqa: E402

OUT = ROOT / "runs" / "smoke" / "checklist"
OUT.mkdir(parents=True, exist_ok=True)

# item 1
env = environment(4)
env["gpu_note"] = ("No GPU in this allocation (cpu_short). GPU not used by design: the user's gpu48 QOS allows 2 GPUs and the user's other "
                   "chains hold both (_rules.md); CPU timing (docs/timing_s1.json) shows a 100k-step run takes ~0.5-2.6 h on CPU.")
env["git"] = git_commit()
(OUT / "env.json").write_text(json.dumps(env, indent=1) + "\n")

# item 2
fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
info = {}
for a, name in zip(ax, ["gmm", "checkerboard"]):
    x = D.train_set(name)
    e = D.eval_set(name, 4096, 0)
    info[name] = {"train_n": len(x), "train_sha256": D.fingerprint(x), "train_mean": x.mean(0).tolist(), "train_std": x.std(0).tolist(),
                  "eval0_sha256": D.fingerprint(e)}
    if name == "checkerboard":
        info[name]["train_frac_in_support"] = float(D.in_checkerboard(x).mean())
    a.scatter(x[:20000, 0], x[:20000, 1], s=0.5, alpha=0.3, c="tab:blue", label="train (first 20k of 102,400)")
    a.scatter(e[:, 0], e[:, 1], s=0.5, alpha=0.3, c="tab:orange", label="eval set k=0 (4096)")
    a.set_title(f"{name} ground truth", fontsize=9)
    a.set_aspect("equal")
    a.legend(fontsize=6, markerscale=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "datasets.png", dpi=120)
fig.savefig(OUT / "datasets.pdf")

# item 3
models = {}
for arm in "ABC":
    torch.manual_seed(0)
    m = build_model(arm)
    t, xt = torch.rand(4), torch.randn(4, 2)
    y = m(t, xt) if arm == "A" else m(t, xt, torch.randn(4, 32, 2))
    models[arm] = {"stochastic": m.stochastic, "n_params": n_params(m), "x_branch_in": m.x_pre[0].in_features, "out_layer": m.out.out_features,
                   "prediction_shape": list(y.shape), "finite": bool(torch.isfinite(y).all())}

# item 8: smoke cb_C, uninterrupted vs 150 steps + resume
a = torch.load(ROOT / "runs/smoke/cb_C_s0/ckpt/final.pt", weights_only=False)
b = torch.load(ROOT / "runs/smoke/cb_C_s0_resumed/ckpt/final.pt", weights_only=False)
maxdiff = {k: max(float((a[k][n] - b[k][n]).abs().max()) for n in a[k]) for k in ("model", "ema")}
la = [json.loads(l) for l in open(ROOT / "runs/smoke/cb_C_s0/log.jsonl")]
lb = [json.loads(l) for l in open(ROOT / "runs/smoke/cb_C_s0_resumed/log.jsonl")]
resume = {"run_uninterrupted": "runs/smoke/cb_C_s0", "run_interrupted": "runs/smoke/cb_C_s0_resumed (150 steps, process exit, resume to 300)",
          "max_abs_param_diff": maxdiff, "bitwise_equal": all(v == 0.0 for v in maxdiff.values()),
          "rng_state_equal": bool(torch.equal(a["gen"], b["gen"])), "loss_logs_equal": [r["loss"] for r in la] == [r["loss"] for r in lb],
          "n_log_lines": [len(la), len(lb)], "checkpoint_contents": sorted(a.keys()), "tolerance": "exact (0.0)"}
(OUT / "models.json").write_text(json.dumps(models, indent=1) + "\n")
(OUT / "datasets.json").write_text(json.dumps(info, indent=1) + "\n")
(OUT / "resume.json").write_text(json.dumps(resume, indent=1) + "\n")
print(json.dumps({"models": models, "resume": resume, "data": info}, indent=1))
