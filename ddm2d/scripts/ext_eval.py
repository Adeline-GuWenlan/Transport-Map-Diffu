"""ddm2dx X3: one evaluation protocol for every 2D run of E1, E2, E4 and the reference runs runs/paper/gmm_{A,B,C}_s{0,1}.

Built on scripts/eval_gmm.py and ddm2d.evaluate: the same sampler (Algorithm 2, churn 1, safety 1e-2, clamp_low), the same seeds
(sampler seed 1e6 + 1000 r + nfe, reference set data.eval_set(ds, 4096, k=r)), RBF sigma = 1, EMA weights of ckpt/final.pt.
The exact posterior is the general GMM posterior (posterior_gmm.py), which equals posterior.py on the paper GMM (tests).

Parts (each writes runs/ext/eval/<part>/<run>.json, resumable: an existing output is skipped unless --force):
  mmd     MMD^2_u vs NFE {5,10,20,30,50,100}, 10 repeats; plus final-sample mode weights (assignment by the largest pi_k N(x; mu_k, s_k^2 I))
          for every (NFE, repeat); repeat-0 samples at NFE 5/10/100 saved as .npz.
  std     posterior std vs t (4096 x_t per t, 8 draws; x0 = eval set 100+i, x_t seed 2e6+i, as evaluate.posterior_std_curve), paper and
          fine t grids; the exact curve is the run "exact".
  fig4    Fig. 4-style draws, t in {0.99,0.88,0.45,0.23,0.01}: gmm/gmm2w from x0 at each mean, gmm9w from x0 at mean k=0 and k=1
          (x_t seed 3e6+10i+j, exact draws seed 4e6+10i+j, xi seed 5e6+10i+j, as eval_gmm.py fig4). 4096 draws per panel, 1024 saved.
  den     E1a denoiser location at t in {0.99,0.88,0.45}: 256 x_t (x0 = eval set 300, x_t seed 6e6+j); x_hat = mean of 8 draws; distance to
          the exact posterior mean, the geometric median (Weiszfeld on 4096 exact posterior draws), the coordinate-wise median and the
          nearest component mean; plus the spread of x_hat over xi.
  condw   conditional mode weights at t in {0.99,0.88,0.7,0.45}: 256 x_t (x0 = eval set 400, x_t seed 7e6+j), 4096 draws per x_t, assigned
          to modes, vs the exact responsibilities r_k(x_t); null = 4096 exact posterior draws per x_t through the same assignment.
  snaps   DIAGNOSTIC: EMA snapshots 10k..100k, MMD^2 at NFE 5/10/100, 2 repeats (non-finite samples dropped, count kept).
  null    per dataset: MMD^2 null (eval set 50+r vs r, r < 10) and the multinomial mode-weight null (100 sets of 4096 true samples,
          eval sets 1000+r).
  summary merge everything into runs/ext/eval/*.csv / *.json.

Usage: python scripts/ext_eval.py <part> [--runs NAME ...] [--threads 1] [--force]
Run names: an ext run is runs/ext/<name>; gmm_{A,B,C}_s{0,1} are the reference runs in runs/paper/.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "runs" / "ext" / "eval"
NFES = [5, 10, 20, 30, 50, 100]
REPEATS = 10
REFS = [f"gmm_{a}_s{s}" for a in "ABC" for s in (0, 1, 2)]           # s2 added by X3 to match the seed-2 extras
FIG4_T = [0.99, 0.88, 0.45, 0.23, 0.01]
DEN_T = [0.99, 0.88, 0.45]
CONDW_T = [0.99, 0.88, 0.7, 0.45]
RARE = {"gmm": [0, 1], "gmm2w": [1], "gmm9w": [1, 3, 5]}          # modes for "largest relative error among the rare modes"


def ext_runs():
    """runs.tsv (X1) + runs_x3.tsv (X3 extras, only once they have DONE.json)."""
    rows = list(csv.DictReader(open(ROOT / "runs/ext/runs.tsv"), delimiter="\t"))
    x3 = list(csv.DictReader(open(ROOT / "runs/ext/runs_x3.tsv"), delimiter="\t")) if (ROOT / "runs/ext/runs_x3.tsv").exists() else []
    return [Path(r["out_dir"]).name for r in rows] + [Path(r["out_dir"]).name for r in x3 if (ROOT / r["out_dir"] / "DONE.json").exists()]


def run_dir(name):
    return ROOT / "runs" / ("paper" if name in REFS else "ext") / name


def run_cfg(name):
    return json.loads((run_dir(name) / "config.json").read_text())


def mode_stats(x, ds, np):
    from ddm2d.posterior_gmm import assign_components, spec
    means, stds, w = spec(ds)
    c = assign_components(x, means, stds, w)
    K = len(w)
    p = np.bincount(c, minlength=K) / len(x)
    dist = np.linalg.norm(x[:, None, :] - means[None], axis=-1) / stds[None]
    rare = RARE[ds]
    return {"frac": p.tolist(), "tv": float(0.5 * np.abs(p - w).sum()),
            "rare_max_rel_err": float(max(abs(p[k] - w[k]) / w[k] for k in rare)),
            "off_mode": float((dist.min(1) > 4).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["mmd", "std", "fig4", "den", "condw", "null", "snaps", "summary"])
    ap.add_argument("--runs", nargs="+", default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    from ddm2d import data as D
    from ddm2d.evaluate import sampler_seed
    from ddm2d.mmd import rbf_mmd2
    from ddm2d import posterior_gmm as PG
    from ddm2d.samplers import posterior_draws, sample
    from ddm2d.schedule import forward_sample
    from ddm2d.train import load_ema_model

    if a.part == "summary":
        return summary(np)
    runs = a.runs or (REFS + ext_runs())
    (OUT / a.part).mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if a.part == "null":
        for ds in (a.runs or ["gmm", "gmm2w", "gmm9w"]):
            f = OUT / "null" / f"{ds}.json"
            if f.exists() and not a.force:
                continue
            mmd = []
            for r in range(REPEATS):
                res = rbf_mmd2(D.eval_set(ds, 4096, k=50 + r), D.eval_set(ds, 4096, k=r), 1.0)
                res.update(repeat=r)
                mmd.append(res)
            modes = [mode_stats(D.eval_set(ds, 4096, k=1000 + r), ds, np) for r in range(100)]
            big = mode_stats(D.generate(ds, 1_000_000, 99), ds, np)          # assignment-only bias (overlap), no sampling noise
            f.write_text(json.dumps({"dataset": ds, "mmd": mmd, "modes": modes, "modes_1e6": big}) + "\n")
            print(ds, "null", f"{np.mean([m['mmd2_u'] for m in mmd]):.2e}", f"tv {np.mean([m['tv'] for m in modes]):.4f}", flush=True)
        return

    for name in runs:
        f = OUT / a.part / f"{name}.json"
        if f.exists() and not a.force:
            print("skip", name, a.part)
            continue
        ds = run_cfg(name)["dataset"]
        means, stds, w = PG.spec(ds)
        model = load_ema_model(run_dir(name))
        res = {"run": name, "dataset": ds, "part": a.part}

        if a.part == "mmd":
            rows, keep = [], {}
            for r in range(REPEATS):
                ref = D.eval_set(ds, 4096, k=r)
                for nfe in NFES:
                    g = torch.Generator().manual_seed(sampler_seed(nfe, r))
                    x, used = sample(model, 4096, nfe, churn=1.0, safety=0.01, generator=g, return_nfe=True, chunk=4096, clamp_low=True)
                    assert used == nfe
                    x = x.numpy()
                    row = rbf_mmd2(x, ref, 1.0)
                    row.update(nfe=nfe, repeat=r, n_finite=int(np.isfinite(x).all(1).sum()))
                    row.update(mode_stats(x.astype(np.float64), ds, np))
                    rows.append(row)
                    if r == 0 and nfe in (5, 10, 100):
                        keep[f"nfe{nfe}"] = x.astype(np.float32)
            res["rows"] = rows
            np.savez_compressed(OUT / "mmd" / f"samples_{name}.npz", **keep)

        elif a.part == "std":
            grids = {"paper": [float(x) for x in np.linspace(0.01, 0.99, 10)],
                     "fine": [float(x) for x in np.round(np.arange(0.01, 0.995, 0.02), 4)]}
            res["curves"] = {}
            for gname, ts in grids.items():
                cur = []
                for i, t in enumerate(ts):
                    x0 = torch.from_numpy(D.eval_set(ds, 4096, k=100 + i))
                    g = torch.Generator().manual_seed(2_000_000 + i)
                    xt = forward_sample(x0, torch.tensor(float(t)), torch.randn(x0.shape, generator=g))
                    dr = posterior_draws(model, float(t), xt, 8, generator=g).numpy()
                    ex = PG.sample_posterior(xt.numpy(), float(t), 8, np.random.default_rng(2_000_000 + i), means, stds, w)
                    _, var = PG.posterior_mean_cov(xt.numpy(), float(t), means, stds, w)
                    cur.append({"t": float(t), "std_mean": float(dr.std(1, ddof=0).mean()), "exact8_std_mean": float(ex.std(1, ddof=0).mean()),
                                "exact_std_mean": float(np.sqrt(var).mean())})
                res["curves"][gname] = cur

        elif a.part == "fig4":
            x0s = [means[0], means[1]]
            xts = np.zeros((2, len(FIG4_T), 2), np.float32)
            panels, keep_m, keep_p = [], np.zeros((2, len(FIG4_T), 1024, 2), np.float32), np.zeros((2, len(FIG4_T), 1024, 2), np.float32)
            for i, x0 in enumerate(x0s):
                for j, t in enumerate(FIG4_T):
                    g = torch.Generator().manual_seed(3_000_000 + 10 * i + j)
                    xts[i, j] = forward_sample(torch.tensor([x0], dtype=torch.float32), torch.tensor(t), torch.randn((1, 2), generator=g)).numpy()[0]
                    xt = xts[i, j][None].astype(np.float64)
                    post = PG.sample_posterior(xt, t, 4096, np.random.default_rng(4_000_000 + 10 * i + j), means, stds, w)[0]
                    g = torch.Generator().manual_seed(5_000_000 + 10 * i + j)
                    dr = posterior_draws(model, t, torch.from_numpy(xts[i, j][None]), 4096, generator=g)[0].numpy().astype(np.float64)
                    r_ex = PG.responsibilities(xt, t, means, stds, w)[0]
                    m_ex, v_ex = PG.posterior_mean_cov(xt, t, means, stds, w)
                    cd = np.bincount(PG.assign_components(dr, means, stds, w), minlength=len(w)) / 4096
                    cp = np.bincount(PG.assign_components(post, means, stds, w), minlength=len(w)) / 4096
                    panels.append({"row": i, "x0": [float(v) for v in x0], "t": t, "xt": xts[i, j].tolist(),
                                   "exact_resp": r_ex.tolist(), "exact_mean": m_ex[0].tolist(), "exact_std": np.sqrt(v_ex[0]).tolist(),
                                   "exact_draw_frac": cp.tolist(), "frac": cd.tolist(), "mean": dr.mean(0).tolist(), "std": dr.std(0).tolist(),
                                   "tv_vs_resp": float(0.5 * np.abs(cd - r_ex).sum()), "tv_exactdraws_vs_resp": float(0.5 * np.abs(cp - r_ex).sum()),
                                   "mmd2_u_vs_posterior": rbf_mmd2(dr, post, 1.0)["mmd2_u"],
                                   "mean_err_vs_exact_mean": float(np.linalg.norm(dr.mean(0) - m_ex[0]))})
                    keep_m[i, j], keep_p[i, j] = dr[:1024], post[:1024]
            res["panels"] = panels
            np.savez_compressed(OUT / "fig4" / f"draws_{name}.npz", xt=xts, t=np.array(FIG4_T, np.float32), model=keep_m, posterior=keep_p)

        elif a.part == "den":
            out = []
            for j, t in enumerate(DEN_T):
                x0 = torch.from_numpy(D.eval_set(ds, 256, k=300))
                g = torch.Generator().manual_seed(6_000_000 + j)
                xt = forward_sample(x0, torch.tensor(float(t)), torch.randn(x0.shape, generator=g))
                dr = posterior_draws(model, float(t), xt, 8, generator=g).numpy().astype(np.float64)
                xhat = dr.mean(1)
                spread = float(dr.std(1, ddof=0).mean())
                xtn = xt.numpy().astype(np.float64)
                post = PG.sample_posterior(xtn, float(t), 4096, np.random.default_rng(6_100_000 + j), means, stds, w)   # [256, 4096, 2]
                pmean, _ = PG.posterior_mean_cov(xtn, float(t), means, stds, w)
                gmed = weiszfeld(post, np)
                cmed = np.median(post, axis=1)
                nearest = means[np.linalg.norm(xhat[:, None] - means[None], axis=-1).argmin(1)]
                r_ex = PG.responsibilities(xtn, float(t), means, stds, w)
                top = means[r_ex.argmax(1)]
                dd = lambda u, v: float(np.linalg.norm(u - v, axis=-1).mean())
                out.append({"t": t, "xi_spread": spread, "d_mean": dd(xhat, pmean), "d_gmed": dd(xhat, gmed), "d_cmed": dd(xhat, cmed),
                            "d_nearest_mode": dd(xhat, nearest), "d_top_resp_mode": dd(xhat, top),
                            "ref_d_mean_gmed": dd(pmean, gmed), "ref_d_mean_top": dd(pmean, top), "ref_d_gmed_top": dd(gmed, top),
                            "frac_closest": {k: float(v) for k, v in zip(["mean", "gmed", "cmed", "mode"], np.bincount(np.stack(
                                [np.linalg.norm(xhat - q, axis=-1) for q in (pmean, gmed, cmed, nearest)]).argmin(0), minlength=4) / 256)}})
            res["rows"] = out

        elif a.part == "condw":
            out = []
            K = len(w)
            for j, t in enumerate(CONDW_T):
                x0 = torch.from_numpy(D.eval_set(ds, 256, k=400))
                g = torch.Generator().manual_seed(7_000_000 + j)
                xt = forward_sample(x0, torch.tensor(float(t)), torch.randn(x0.shape, generator=g))
                xtn = xt.numpy().astype(np.float64)
                r_ex = PG.responsibilities(xtn, float(t), means, stds, w)
                ph, pn = np.zeros((256, K)), np.zeros((256, K))
                rng = np.random.default_rng(7_100_000 + j)
                for s in range(0, 256, 32):
                    dr = posterior_draws(model, float(t), xt[s:s + 32], 4096, generator=g).numpy().astype(np.float64)
                    post = PG.sample_posterior(xtn[s:s + 32], float(t), 4096, rng, means, stds, w)
                    for q in range(dr.shape[0]):
                        ph[s + q] = np.bincount(PG.assign_components(dr[q], means, stds, w), minlength=K) / 4096
                        pn[s + q] = np.bincount(PG.assign_components(post[q], means, stds, w), minlength=K) / 4096
                minority = r_ex.copy()
                minority[np.arange(256), r_ex.argmax(1)] = 0               # mass of the non-dominant modes
                out.append({"t": t, "mean_tv": float(0.5 * np.abs(ph - r_ex).sum(1).mean()), "null_mean_tv": float(0.5 * np.abs(pn - r_ex).sum(1).mean()),
                            "mean_abs_err_per_mode": np.abs(ph - r_ex).mean(0).tolist(), "null_mean_abs_err_per_mode": np.abs(pn - r_ex).mean(0).tolist(),
                            "mean_frac": ph.mean(0).tolist(), "mean_resp": r_ex.mean(0).tolist(), "null_mean_frac": pn.mean(0).tolist(),
                            "nondominant_mass_model": float((ph.sum(1) - ph[np.arange(256), r_ex.argmax(1)]).mean()),
                            "nondominant_mass_exact": float(minority.sum(1).mean()),
                            "nondominant_mass_null": float((pn.sum(1) - pn[np.arange(256), r_ex.argmax(1)]).mean())})
            res["rows"] = out

        elif a.part == "snaps":
            # DIAGNOSTIC only (not the protocol): EMA snapshots at 10k ... 100k, NFE 5/10/100, 2 repeats, same seeds as mmd
            rows = []
            for step in range(10_000, 100_001, 10_000):
                msnap = load_ema_model(run_dir(name), step=step)
                for r in range(2):
                    ref = D.eval_set(ds, 4096, k=r)
                    for nfe in (5, 10, 100):
                        g = torch.Generator().manual_seed(sampler_seed(nfe, r))
                        x = sample(msnap, 4096, nfe, churn=1.0, safety=0.01, generator=g, chunk=4096, clamp_low=True).numpy()
                        fin = np.isfinite(x).all(1)
                        row = rbf_mmd2(x[fin], ref, 1.0) if fin.sum() > 1 else {"mmd2_u": float("nan")}
                        row.update(step=step, repeat=r, nfe=nfe, n_finite=int(fin.sum()))
                        rows.append(row)
            res["rows"] = rows

        res["seconds"] = round(time.time() - t0, 1)
        f.write_text(json.dumps(res) + "\n")
        print(name, a.part, f"{time.time() - t0:.0f}s", flush=True)
        t0 = time.time()


def weiszfeld(pts, np, iters=200, eps=1e-9):
    """Geometric median of each row of pts [n, m, d] (Weiszfeld, started at the mean)."""
    y = pts.mean(1)
    for _ in range(iters):
        dist = np.maximum(np.linalg.norm(pts - y[:, None], axis=-1), eps)
        wgt = 1.0 / dist
        y_new = (wgt[..., None] * pts).sum(1) / wgt.sum(1, keepdims=True)
        if np.abs(y_new - y).max() < 1e-7:
            return y_new
        y = y_new
    return y


def summary(np):
    """Merge per-run JSON into flat CSV tables."""
    import statistics as st
    meta = {}
    for name in REFS + ext_runs():
        c = run_cfg(name)
        meta[name] = dict(run=name, dataset=c["dataset"], arm=c["arm"], beta=c.get("beta"), lam=c.get("lam"), rho=c.get("rho", "l2") if c["arm"] != "A" else "-",
                          rho_c=c.get("rho_c", ""), seed=int(name.rsplit("_s", 1)[1]), group=name.rsplit("_s", 1)[0])
    def write(fn, rows):
        if not rows:
            return
        with open(OUT / fn, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    mm = []
    for name, m in meta.items():
        f = OUT / "mmd" / f"{name}.json"
        if f.exists():
            for r in json.loads(f.read_text())["rows"]:
                ok = r["n_finite"] == 4096          # mode statistics are undefined when some samples overflowed (C beta = 2): NaN
                nan = float("nan")
                mm.append(dict(m, nfe=r["nfe"], repeat=r["repeat"], mmd2_u=r["mmd2_u"], tv=r["tv"] if ok else nan,
                               rare_max_rel_err=r["rare_max_rel_err"] if ok else nan, off_mode=r["off_mode"] if ok else nan, n_finite=r["n_finite"],
                               frac=" ".join(f"{v:.5f}" for v in r["frac"]) if ok else " ".join(["nan"] * len(r["frac"]))))
    write("mmd_raw.csv", mm)
    # per-run means over repeats, then per-group mean and seed spread (min, max of the per-seed means)
    per_run = {}
    for r in mm:
        per_run.setdefault((r["run"], r["nfe"]), []).append(r)
    pr = []
    for (run, nfe), rs in per_run.items():
        pr.append(dict(meta[run], nfe=nfe, mmd2_u=float(np.mean([x["mmd2_u"] for x in rs])), mmd2_u_sd_rep=float(np.std([x["mmd2_u"] for x in rs])),
                       tv=float(np.mean([x["tv"] for x in rs])), rare_max_rel_err=float(np.mean([x["rare_max_rel_err"] for x in rs])),
                       off_mode=float(np.mean([x["off_mode"] for x in rs])), n_rep=len(rs)))
    write("mmd_per_run.csv", pr)
    grp = {}
    for r in pr:
        grp.setdefault((r["dataset"], r["group"], r["nfe"]), []).append(r)
    gr = []
    for (ds, g, nfe), rs in sorted(grp.items()):
        v = [x["mmd2_u"] for x in rs]
        gr.append(dict(dataset=ds, group=g, arm=rs[0]["arm"], beta=rs[0]["beta"], lam=rs[0]["lam"], rho=rs[0]["rho"], rho_c=rs[0]["rho_c"], nfe=nfe,
                       n_seeds=len(rs), mmd2_u_mean=float(np.mean(v)), mmd2_u_min_seed=float(np.min(v)), mmd2_u_max_seed=float(np.max(v)),
                       tv_mean=float(np.mean([x["tv"] for x in rs])), tv_min_seed=float(np.min([x["tv"] for x in rs])), tv_max_seed=float(np.max([x["tv"] for x in rs])),
                       rare_max_rel_err_mean=float(np.mean([x["rare_max_rel_err"] for x in rs])), off_mode_mean=float(np.mean([x["off_mode"] for x in rs]))))
    write("mmd_groups.csv", gr)
    nulls = {}
    for ds in ["gmm", "gmm2w", "gmm9w"]:
        f = OUT / "null" / f"{ds}.json"
        if f.exists():
            d = json.loads(f.read_text())
            tv = np.array([m["tv"] for m in d["modes"]])
            rr = np.array([m["rare_max_rel_err"] for m in d["modes"]])
            om = np.array([m["off_mode"] for m in d["modes"]])
            mu = np.array([m["mmd2_u"] for m in d["mmd"]])
            nulls[ds] = dict(mmd2_u_mean=float(mu.mean()), mmd2_u_sd=float(mu.std()), mmd2_u_absmax=float(np.abs(mu).max()),
                             tv_mean=float(tv.mean()), tv_p95=float(np.quantile(tv, 0.95)), rare_rel_err_mean=float(rr.mean()),
                             rare_rel_err_p95=float(np.quantile(rr, 0.95)), off_mode_mean=float(om.mean()), off_mode_p95=float(np.quantile(om, 0.95)),
                             tv_1e6=d["modes_1e6"]["tv"], frac_1e6=d["modes_1e6"]["frac"])
    (OUT / "nulls.json").write_text(json.dumps(nulls, indent=1) + "\n")
    # std, den, condw flat tables
    sr, dr, cr, fr = [], [], [], []
    for name, m in meta.items():
        f = OUT / "std" / f"{name}.json"
        if f.exists():
            for gname, cur in json.loads(f.read_text())["curves"].items():
                for p in cur:
                    sr.append(dict(m, grid=gname, **p))
        f = OUT / "den" / f"{name}.json"
        if f.exists():
            for p in json.loads(f.read_text())["rows"]:
                fc = p.pop("frac_closest")
                dr.append(dict(m, **p, **{f"closest_{k}": v for k, v in fc.items()}))
        f = OUT / "condw" / f"{name}.json"
        if f.exists():
            for p in json.loads(f.read_text())["rows"]:
                cr.append(dict(m, t=p["t"], mean_tv=p["mean_tv"], null_mean_tv=p["null_mean_tv"], nondominant_mass_model=p["nondominant_mass_model"],
                               nondominant_mass_exact=p["nondominant_mass_exact"], nondominant_mass_null=p["nondominant_mass_null"],
                               mean_frac=" ".join(f"{v:.4f}" for v in p["mean_frac"]), mean_resp=" ".join(f"{v:.4f}" for v in p["mean_resp"])))
        f = OUT / "fig4" / f"{name}.json"
        if f.exists():
            for p in json.loads(f.read_text())["panels"]:
                fr.append(dict(m, row=p["row"], t=p["t"], tv_vs_resp=p["tv_vs_resp"], tv_exactdraws_vs_resp=p["tv_exactdraws_vs_resp"],
                               mmd2_u_vs_posterior=p["mmd2_u_vs_posterior"], mean_err=p["mean_err_vs_exact_mean"],
                               std=" ".join(f"{v:.4f}" for v in p["std"]), exact_std=" ".join(f"{v:.4f}" for v in p["exact_std"]),
                               frac=" ".join(f"{v:.4f}" for v in p["frac"]), exact_resp=" ".join(f"{v:.4f}" for v in p["exact_resp"])))
    write("posterior_std.csv", sr)
    write("denoiser.csv", dr)
    write("condw.csv", cr)
    write("fig4_panels.csv", fr)
    print("summary:", len(mm), "mmd rows,", len(sr), "std,", len(dr), "den,", len(cr), "condw,", len(fr), "fig4")


if __name__ == "__main__":
    main()
