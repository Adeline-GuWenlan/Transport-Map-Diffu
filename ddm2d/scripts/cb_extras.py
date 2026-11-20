"""S4 checkerboard investigations (diagnostics; never mixed into the Table 2 numbers):

  oracle  The same sampler (ddm2d.samplers.sample: uniform grid, safety clamp, churn 1 and 0) driven by the EXACT checkerboard posterior
          (src/ddm2d/posterior_cb.py) instead of a network: 'oracle_mean' returns E[X0 | x_t] (ideal arm A / B), 'oracle_draw' a draw from
          p(X0 | x_t) (ideal arm C). Measures the sampler's own discretisation error, i.e. the floor for any trained network.
          NFE 1000 is run with clamp_low True and False.
  churn0  All 9 trained checkerboard runs with churn 0 instead of 1 at Table 2's NFEs (same seeds for every arm; sensitivity only).
  grid    Alternative time grid (same for every arm and the oracles): the N denoiser times equally spaced on the safety interval
          [0.01, 0.99] (t_N = 0.99 ... t_1 = 0.01), then the last bridge step to t_0 = 0 (= the last prediction). No time is ever clamped.
          Tests whether the NFE >= 200 degradation comes from calling the denoiser repeatedly at the clamped t = 0.01.
  scale   Data-scale hypothesis for Table 2: the oracles (exact posterior mean / draw) on a checkerboard rescaled by c in {0.25, 0.5, 1}
          (c = 0.5: support [-2, 2]^2, FFJORD without its final x2), the same sampler, RBF sigma = 1 against a reference set rescaled by c.
          The ideal arm A at NFE 5 / 10 shows which scale is consistent with the paper's Diffusion column (no training involved).
  dnerr   Denoiser accuracy vs t for all 9 runs: x0 = eval set 400+i, x_t from the forward process (seed 8e6+i); the exact posterior mean
          m(x_t) (posterior_cb). Arm A: RMS ||x_hat - m||. Arms B, C: RMS ||mean of 32 draws - m|| and the mean per-coordinate std of the
          draws (vs the exact posterior std, from 32 exact draws). Shows how accurate each network is at low noise.

Usage: python scripts/cb_extras.py oracle|churn0|grid|scale|dnerr [--repeats 5] [--threads 2]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "runs" / "paper" / "eval_cb"
CURVE_NFE = [5, 10, 20, 30, 50, 70, 100, 200, 500, 1000]
TABLE_NFE = [5, 10, 50, 100, 1000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["oracle", "churn0", "grid", "scale", "dnerr"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    from ddm2d.evaluate import mmd_vs_nfe
    from ddm2d.posterior_cb import posterior_mean, sample_posterior
    from ddm2d.train import load_ema_model

    class Oracle:
        def __init__(self, draw: bool, seed: int, scale: float = 1.0):
            self.stochastic = draw
            self.rng = np.random.default_rng(seed)
            self.scale = scale

        def eval(self):
            return self

        def __call__(self, t, xt, xi=None):
            x = xt.double().numpy()
            out = (sample_posterior(x, float(t), 1, self.rng, scale=self.scale)[:, 0] if self.stochastic
                   else posterior_mean(x, float(t), scale=self.scale))
            return torch.from_numpy(out).to(xt.dtype)

    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    if a.part == "oracle":
        for churn in (1.0, 0.0):
            for kind in ("oracle_mean", "oracle_draw"):
                for cl in (True, False):
                    nfes = CURVE_NFE if cl else [1000]
                    res = mmd_vs_nfe(Oracle(kind == "oracle_draw", seed=7 + int(churn)), "checkerboard", nfes, n=4096, repeats=a.repeats,
                                     churn=churn, clamp_low=cl)
                    for r in res:
                        r["who"] = kind
                    rows += res
                    print(kind, churn, cl, [(n, f"{np.mean([r['mmd2_u'] for r in res if r['nfe'] == n]):.2e}") for n in nfes], flush=True)
        (OUT / "oracle_mmd.json").write_text(json.dumps(rows, indent=0) + "\n")
    elif a.part == "scale":
        from ddm2d import data as D
        from ddm2d.evaluate import sampler_seed
        from ddm2d.mmd import rbf_mmd2
        from ddm2d.samplers import sample
        for c in (0.25, 0.5, 1.0):
            for kind in ("oracle_mean", "oracle_draw"):
                orc = Oracle(kind == "oracle_draw", seed=11, scale=c)
                for nfe in TABLE_NFE:
                    for r in range(a.repeats):
                        g = torch.Generator().manual_seed(sampler_seed(nfe, r))
                        x = sample(orc, 4096, nfe, churn=1.0, safety=0.01, generator=g, chunk=4096, clamp_low=nfe <= 100).numpy()
                        res = rbf_mmd2(x, c * D.eval_set("checkerboard", 4096, k=r), 1.0)
                        res.update(who=kind, scale=c, nfe=nfe, repeat=r, churn=1.0, clamp_low=nfe <= 100)
                        rows.append(res)
                print("scale", c, kind, [(n, f"{np.mean([q['mmd2_u'] for q in rows if q['nfe'] == n and q['who'] == kind and q['scale'] == c]):.2e}")
                                         for n in TABLE_NFE], flush=True)
        (OUT / "scale_oracle_mmd.json").write_text(json.dumps(rows, indent=0) + "\n")
    elif a.part == "dnerr":
        from ddm2d import data as D
        from ddm2d.samplers import posterior_draws
        from ddm2d.schedule import forward_sample
        ts = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]
        prep = []
        for i, t in enumerate(ts):
            x0 = torch.from_numpy(D.eval_set("checkerboard", 4096, k=400 + i))
            g = torch.Generator().manual_seed(8_000_000 + i)
            xt = forward_sample(x0, torch.tensor(t), torch.randn(x0.shape, generator=g))
            m = posterior_mean(xt.double().numpy(), t)
            ex = sample_posterior(xt.double().numpy(), t, 32, np.random.default_rng(9_000_000 + i))
            prep.append((t, xt, m, float(ex.std(1).mean()), float(np.sqrt(((ex.mean(1) - m) ** 2).sum(1).mean()))))
        for arm in "ABC":
            for s_ in range(3):
                run = f"cb_{arm}_s{s_}"
                model = load_ema_model(ROOT / "runs" / "paper" / run)
                for i, (t, xt, m, ex_std, ex_mc) in enumerate(prep):
                    tin = min(max(t, 0.01), 0.99)                          # the sampler's clamp (training range [0.01, 0.99])
                    for tname, tq in (("clamped", tin), ("raw", t)):
                        if tname == "raw" and t >= 0.01:
                            continue
                        d = posterior_draws(model, tq, xt, 32, generator=torch.Generator().manual_seed(8_500_000 + i)).double().numpy()
                        rows.append(dict(run=run, t=t, t_input=tq, which=tname, rms_err_mean=float(np.sqrt(((d.mean(1) - m) ** 2).sum(1).mean())),
                                         draw_std=float(d.std(1).mean()), exact_std_32=ex_std, exact_mc_err_32=ex_mc,
                                         rms_xt_over_alpha_minus_m=float(np.sqrt(((xt.double().numpy() / (1 - t) - m) ** 2).sum(1).mean()))))
                print(run, [(q["t"], f"{q['rms_err_mean']:.3f}") for q in rows if q["run"] == run and q["which"] == "clamped"], flush=True)
        (OUT / "denoiser_err.json").write_text(json.dumps(rows, indent=0) + "\n")
    elif a.part == "grid":
        from ddm2d import data as D
        from ddm2d.evaluate import sampler_seed
        from ddm2d.mmd import rbf_mmd2
        from ddm2d.samplers import denoise
        from ddm2d.schedule import bridge_coeffs
        who = {"oracle_mean": Oracle(False, 0), "oracle_draw": Oracle(True, 1)}
        for arm in "ABC":
            for s in range(3):
                who[f"cb_{arm}_s{s}"] = load_ema_model(ROOT / "runs" / "paper" / f"cb_{arm}_s{s}")
        for name, model in who.items():
            model.eval()
            for nfe in TABLE_NFE:
                grid = [0.0] + [float(x) for x in np.linspace(0.01, 0.99, nfe)]
                for r in range(a.repeats):
                    g = torch.Generator().manual_seed(sampler_seed(nfe, r))
                    x = torch.randn((4096, 2), generator=g)
                    with torch.no_grad():
                        for k in range(nfe - 1, -1, -1):
                            s_, t_ = grid[k], grid[k + 1]
                            x0h = denoise(model, t_, x, g) if hasattr(model, "parameters") else model(t_, x)
                            c1, c2, sd = bridge_coeffs(s_, t_, 1.0)
                            x = c1 * x + c2 * x0h + sd * torch.randn(x.shape, generator=g)
                    xn = x.numpy()
                    res = rbf_mmd2(xn, D.eval_set("checkerboard", 4096, k=r), 1.0)
                    res.update(who=name, nfe=nfe, repeat=r, churn=1.0, grid="[0] + linspace(0.01, 0.99, N)",
                               frac_in_support=float(D.in_checkerboard(xn).mean()))
                    rows.append(res)
            print("safety grid", name, [(n, f"{np.mean([q['mmd2_u'] for q in rows if q['nfe'] == n and q['who'] == name]):.2e}") for n in TABLE_NFE], flush=True)
        (OUT / "grid_mmd.json").write_text(json.dumps(rows, indent=0) + "\n")
    else:
        for arm in "ABC":
            for s in range(3):
                run = f"cb_{arm}_s{s}"
                res = mmd_vs_nfe(load_ema_model(ROOT / "runs" / "paper" / run), "checkerboard", TABLE_NFE, n=4096, repeats=a.repeats, churn=0.0)
                for r in res:
                    r["who"] = run
                rows += res
                print(run, [(n, f"{np.mean([r['mmd2_u'] for r in res if r['nfe'] == n]):.2e}") for n in TABLE_NFE], flush=True)
        (OUT / "churn0_mmd.json").write_text(json.dumps(rows, indent=0) + "\n")


if __name__ == "__main__":
    main()
