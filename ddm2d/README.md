# ddm2d: reproducing the 2D experiments of Distributional Diffusion Models (De Bortoli et al., ICML 2025)

Reproduction of the 2D experiments of *Distributional Diffusion Models with Scoring Rules* (arXiv 2502.02483 v3):
- the Gaussian mixture (§6.1, Fig. 3–4, App. I.1);
- the checkerboard (App. K.1, Table 2).

Three arms, all trained with the paper's App. I.1 configuration (100k steps, batch 128, Adam 1e-3, EMA 0.99):

| Arm | Model | Loss |
|---|---|---|
| **A** | standard diffusion | Eq. 6 |
| **B** | β = 2, λ = 0 control, stochastic architecture, m = 32 | Eq. 14 |
| **C** | distributional diffusion, β = 0.1, λ = 1, m = 32 | Eq. 14 |

Everything was trained on CPU: 3 seeds per arm and dataset, 18 runs.

**Status (S4, 2026-11-20): complete.** Full account: `docs/reproducibility_report.md`.

## Main result
**The distributional model's large few-step advantage reproduces, on both datasets, for all 3 seeds, with the same sampler, seeds and metric for
every arm.**

| | NFE 5: A | NFE 5: B | NFE 5: C | NFE 10: A | NFE 10: C | source |
|---|---|---|---|---|---|---|
| GMM, ours (MMD²_u, mean of 3 seeds) | 1.66e-2 | 1.74e-2 | **1.8e-4** | 4.7e-3 | **3.1e-4** | `docs/results_gmm.md` §3 |
| GMM, paper (read off Fig. 3 left) | 1.62e-2 | 1.59e-2 | 1.6e-4 | 2.35e-3 | 4.6e-4 | `docs/paper_values/fig3_paper_values.json` |
| Checkerboard, ours (MMD²_u, mean of 3 seeds) | 3.04e-2 | 3.44e-2 | **3.1e-4** | 7.2e-3 | **3.4e-4** | `docs/results_cb.md` §2 |
| Checkerboard, paper (Table 2; metric convention unknown) | 7.00e-3 | — | 1.57e-4 | 2.10e-3 | 1.96e-4 | `docs/paper_values/table2_paper_values.json` |

What does not match, in short (details and investigations in the results docs):
- **GMM.** NFE 5 matches the paper numerically. At NFE 10–30 our A/B are about 2× above the paper's curve, at the level of an exact-posterior-mean
  oracle driving the same sampler.
- **Checkerboard.** Table 2's metric is undefined: "MMD", with no estimator, bandwidth or sample size. So every numeric comparison is marked
  "unverified" (`docs/audit.md` §5).
  - Our A at NFE 5–10 sits at the exact-oracle floor of our sampler, which is 3–5× above the paper's Diffusion column.
  - Our C is ≈ 2× above the paper's Distributional column.
  - At NFE ≥ 200 every arm gets worse. The cause is the networks' low-noise accuracy, measured against the exact posterior.
- **Diagnostic** (not a paper run). Continuing every checkerboard A and C run for 10k steps at lr 1e-4 instead of 1e-3 brings C to
  1.0–1.2e-4 at NFE 5–100 (seed means), at or below the paper's values. A at NFE 5–10 does not move, because it is at the oracle floor.
  So much of the low-noise gap is convergence under the paper's constant lr 1e-3 with EMA 0.99 (`docs/results_cb.md` §5.5).

## Layout
| Path | Content |
|---|---|
| `src/ddm2d/` | `schedule` (Eq. 2–4, sigmoid weight), `data` (GMM, FFJORD checkerboard), `model` (App. I.1 MLP), `losses` (Eq. 6, 14), `samplers` (Algorithm 2 / Eq. 4 bridge), `mmd` (RBF MMD², float64, chunked), `posterior` (exact GMM posterior), `posterior_cb` (exact checkerboard posterior, diagnostics), `train` (resumable), `evaluate` |
| `tests/` | unit tests: schedule, bridge, losses, sampler, MMD, posteriors, model, data, training resume |
| `configs/{smoke,paper}/` | smoke and paper configs (`gmm_{A,B,C}.json`, `cb_{A,B,C}.json`) |
| `scripts/` | `train_job.sbatch` (packed, self-requeueing Slurm job), `check_complete.py`, `eval_gmm.py`, `gmm_extras.py`, `plot_gmm.py`, `eval_cb.py`, `cb_extras.py`, `plot_cb.py`, `diag_finetune.py`, `diag_eval.py`, … |
| `runs/paper/` | 18 paper runs (config, log, DONE.json, ckpt manifests; `ckpt/` is gitignored), `STATUS.md`, `CKPT_MANIFEST.sha256`, `eval_gmm/`, `eval_cb/` (raw metrics JSON/CSV, samples `.npz`) |
| `runs/smoke/` | smoke-test runs (never used in paper tables) |
| `runs/diag/` | S4 diagnostic fine-tunes (never used in paper tables) |
| `figures/gmm/`, `figures/cb/` | all figures (PNG + PDF) |
| `docs/` | `task_spec.md` (binding spec), `audit.md`, `plan.md`, `paper_vs_implementation.md`, `results_gmm.md`, `results_cb.md`, `reproducibility_report.md`, `benchmark.md`, `smoke_checklist.md`, `paper_values/` |
| `standups/` | per-stage reports S1–S4 |

## Regenerate everything
Python: `/scratch/wg2381/.conda/envs/tmap/bin/python`, with torch 2.14.1 (CPU), numpy 2.4.6, scipy 1.17.1 and matplotlib 3.11.2. Run from `ddm2d/`.
Set the thread count explicitly. `os.cpu_count()` on the cluster nodes reports the whole node.

```bash
PY=/scratch/wg2381/.conda/envs/tmap/bin/python
cd ddm2d

# 0. tests (79 tests, ~1-4 min)
OMP_NUM_THREADS=8 $PY -m pytest -q tests

# 1. training: 18 runs = 3 Slurm jobs (one per seed), 6 processes x 1 thread each, ~4-6 h per job, resumable
for s in 0 1 2; do
  sbatch --job-name=ddm2d_train_s$s --export=ALL,RUNLIST=runs/paper/runlists/seed$s.txt scripts/train_job.sbatch
done
#    one run by hand (resumes from ckpt/latest.pt; writes DONE.json + ckpt/final.pt at 100k steps):
OMP_NUM_THREADS=1 PYTHONPATH=src $PY -m ddm2d.train --config configs/paper/cb_C.json --out runs/paper/cb_C_s0 --seed 0 --threads 1

# 2. completeness + checkpoint integrity
$PY scripts/check_complete.py --threads 8                     # runs/paper/STATUS.md, status.json
(cd runs/paper && grep -v '^#' CKPT_MANIFEST.sha256 | sha256sum -c)

# 3. GMM: Fig. 3 left/right, Fig. 4, loss curves, samples, investigations (~35 min on 8 CPUs)
$PY scripts/eval_gmm.py mmd   --repeats 10 --threads 2
$PY scripts/eval_gmm.py snaps --repeats 2  --threads 2
$PY scripts/eval_gmm.py std   --threads 2
$PY scripts/eval_gmm.py fig4  --threads 2
$PY scripts/gmm_extras.py oracle --repeats 5; $PY scripts/gmm_extras.py grid --repeats 5; $PY scripts/gmm_extras.py churn0 --repeats 3
$PY scripts/check_std_reduction.py
$PY scripts/plot_gmm.py all                                   # figures/gmm/*

# 4. Checkerboard: Table 2, metric vs NFE, sample grids, investigations (~45 min on 8 CPUs)
$PY scripts/eval_cb.py mmd --runs cb_A_s0 cb_C_s2 --threads 2 --repeats 10 &
$PY scripts/eval_cb.py mmd --runs cb_A_s1 cb_C_s1 --threads 2 --repeats 10 &
$PY scripts/eval_cb.py mmd --runs cb_A_s2 cb_C_s0 --threads 2 --repeats 10 &
$PY scripts/eval_cb.py mmd --runs cb_B_s0 cb_B_s1 cb_B_s2 --threads 2 --repeats 10 &
wait
$PY scripts/eval_cb.py null  --threads 2 --repeats 20
$PY scripts/eval_cb.py big   --threads 2 --repeats 3
$PY scripts/eval_cb.py snaps --threads 2
$PY scripts/cb_extras.py oracle --repeats 5; $PY scripts/cb_extras.py grid --repeats 3; $PY scripts/cb_extras.py churn0 --repeats 3
$PY scripts/cb_extras.py scale --repeats 3; $PY scripts/cb_extras.py dnerr --threads 4
$PY scripts/plot_cb.py all                                    # figures/cb/*, eval_cb/table2.md, table2_summary.json, cb_mmd_raw.csv

# 5. diagnostic only (not in any paper table): +10k steps at lr 1e-4 from each final.pt (~20-50 min per run on 1-2 threads)
for r in cb_A_s0 cb_A_s1 cb_A_s2 cb_C_s0 cb_C_s1 cb_C_s2 gmm_A_s0 gmm_C_s0; do
  $PY scripts/diag_finetune.py --run $r --lr 1e-4 --steps 10000 --threads 2
done
$PY scripts/diag_eval.py runs/paper/{cb_A_s0,cb_A_s1,cb_A_s2,cb_C_s0,cb_C_s1,cb_C_s2,gmm_A_s0,gmm_C_s0} --out runs/diag/diag_eval_orig.json
$PY scripts/diag_eval.py runs/diag/*_lr0.0001_10000 --out runs/diag/diag_eval_ft.json
$PY scripts/diag_summary.py                                   # runs/diag/diag_summary.md
```
The extraction of the paper's Fig. 3 values from the arXiv source (`scripts/extract_paper_fig3.py`) needs `pypdf` and the arXiv tarball; see
`third_party/README.md`.

## Conventions worth knowing
- **One NFE = one denoiser call** (asserted by the sampler). Sampler: Eq. (4) bridge, churn ε = 1, uniform grid t_k = k/N. The denoiser time is
  clamped to the training range [0.01, 0.99]; NFE 1000 is also reported with `clamp_low=False`.
- **Metric.** RBF MMD² with σ = 1 and 4096 vs 4096 samples. The unbiased MMD²_u is primary; MMD²_b and D_ρ = ½MMD² are stored. The reference
  sets are fresh draws, never training points.
- **Evaluation.** Always the EMA weights of `final.pt`, with the same seeds for every arm.
- **Paper values** live only in `docs/paper_values/` and in columns labelled "paper".
- **Deviations from the paper:** every one is a row in `docs/paper_vs_implementation.md` (49 rows).
