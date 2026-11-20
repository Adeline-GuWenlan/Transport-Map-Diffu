"""Investigation (S3): which vector->scalar std reduction reproduces the paper's exact-posterior curve in Fig. 3 right?
Exact GMM posterior, the paper's t grid, 4096 (x0, x_t) pairs (x0 = eval set 5), 8 draws per x_t. Candidate reductions:
  coord_ddof0  : per-coordinate std over the 8 draws (ddof 0), mean over the 2 coordinates   (our Fig. 3 right convention)
  coord_ddof1  : same with ddof 1
  exact_coord  : exact posterior per-coordinate std (no sampling), mean over coordinates
  sqrt_meanvar : sqrt of the mean over coordinates of the exact variance
  norm_ddof0   : sqrt(sum over coordinates of the ddof-0 variance)
  mean_dist    : mean over draws of ||draw - mean of the 8 draws||
  flat16       : std of the 16 numbers (8 draws x 2 coordinates) together
All averaged over the 4096 x_t. Output: runs/paper/eval_gmm/std_reduction_check.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ddm2d import data as D  # noqa: E402
from ddm2d.posterior import posterior_mean_cov, sample_posterior  # noqa: E402

pv = json.loads((ROOT / "docs/paper_values/fig3_paper_values.json").read_text())["fig3_right_posterior_std"]["series"]["exact posterior"]
rng = np.random.default_rng(0)
rows = []
for t, (tp, vp) in zip(np.linspace(0.01, 0.99, 10), sorted(pv)):
    x0 = D.eval_set("gmm", 4096, k=5)
    xt = (1 - t) * x0 + t * rng.standard_normal(x0.shape)
    dr = sample_posterior(xt, t, 8, rng)
    _, var = posterior_mean_cov(xt, t)
    r = {"t": float(t), "paper_exact_posterior": vp,
         "coord_ddof0": dr.std(1, ddof=0).mean(), "coord_ddof1": dr.std(1, ddof=1).mean(), "exact_coord": np.sqrt(var).mean(),
         "sqrt_meanvar": np.sqrt(var.mean(1)).mean(), "norm_ddof0": np.sqrt(dr.var(1, ddof=0).sum(-1)).mean(),
         "mean_dist": np.linalg.norm(dr - dr.mean(1, keepdims=True), axis=-1).mean(), "flat16": dr.reshape(4096, -1).std(1).mean()}
    rows.append({k: float(v) for k, v in r.items()})
    print({k: round(v, 4) for k, v in rows[-1].items()})
cands = [k for k in rows[0] if k not in ("t", "paper_exact_posterior")]
ratio = {k: [r["paper_exact_posterior"] / r[k] for r in rows] for k in cands}
(ROOT / "runs/paper/eval_gmm/std_reduction_check.json").write_text(json.dumps({"rows": rows, "paper_over_candidate": ratio}, indent=1) + "\n")
# the same reductions for the trained models (EMA), same x_t protocol as Fig. 3 right (ddm2d.evaluate.posterior_std_curve seeds)
import torch  # noqa: E402
torch.set_num_threads(2)
from ddm2d.samplers import posterior_draws  # noqa: E402
from ddm2d.schedule import forward_sample  # noqa: E402
from ddm2d.train import load_ema_model  # noqa: E402
models = {}
for name in ["exact"] + [f"gmm_{a}_s{s}" for a in "BC" for s in range(3)]:
    m = None if name == "exact" else load_ema_model(ROOT / "runs/paper" / name)
    out = []
    for i, t in enumerate(np.linspace(0.01, 0.99, 10)):
        x0 = torch.from_numpy(D.eval_set("gmm", 4096, k=100 + i))
        g = torch.Generator().manual_seed(2_000_000 + i)
        xt = forward_sample(x0, torch.tensor(float(t)), torch.randn(x0.shape, generator=g))
        if m is None:
            dr = sample_posterior(xt.numpy(), float(t), 8, np.random.default_rng(2_000_000 + i))
        else:
            dr = posterior_draws(m, float(t), xt, 8, generator=g).numpy().astype(np.float64)
        out.append({"t": float(t), "coord_ddof0": float(dr.std(1, ddof=0).mean()),
                    "mean_dist": float(np.linalg.norm(dr - dr.mean(1, keepdims=True), axis=-1).mean())})
    models[name] = out
    print(name, [(round(q["coord_ddof0"], 3), round(q["mean_dist"], 3)) for q in out])
(ROOT / "runs/paper/eval_gmm/std_reduction_check.json").write_text(json.dumps({"rows": rows, "paper_over_candidate": ratio, "models": models},
                                                                               indent=1) + "\n")
for k, v in ratio.items():
    print(f"{k:13s}", " ".join(f"{x:.2f}" for x in v))
