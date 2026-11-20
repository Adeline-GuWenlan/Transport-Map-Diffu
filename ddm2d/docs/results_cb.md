# Checkerboard results (S4): Table 2, metric vs NFE, samples, investigations

Paper: De Bortoli et al., *Distributional Diffusion Models with Scoring Rules*, arXiv 2502.02483 **v3**, App. K.1, Table 2 (added in v3; absent
from v1 and v2). All our numbers come from committed files under `runs/paper/eval_cb/` and `figures/cb/`. The paper's values are in
`docs/paper_values/table2_paper_values.json` and are **never ours**. Labels: **[E]** evidence in a committed file, **[G]** guess / assumption.

## 1. What was evaluated
- **Runs:** 9 checkerboard runs, arms A (standard diffusion, Eq. 6), B (β = 2, λ = 0, m = 32) and C (β = 0.1, λ = 1, m = 32) × training
  seeds 0, 1, 2. All have 100k steps, and every final checkpoint loads and matches `DONE.json` (`runs/paper/STATUS.md`) **[E]**. The paper
  has only A ("Diffusion") and C ("Distributional"); B is our extra control.
- **Dataset and training config are approximations.** The generator is FFJORD `inf_train_gen("checkerboard")` ([−4, 4]², 102,400 points).
  The training config is borrowed from the GMM (App. I.1). Neither is specified for the checkerboard by the paper or by Galashov et al.
  (`paper_vs_implementation.md` rows 4–7, `audit.md` §3–4).
- **Protocol (same for every arm):**
  - EMA weights of `final.pt`; 4096 generated vs 4096 reference samples; the reference is a fresh p_0 draw (eval set r, never training data).
  - Sampler: Eq. (4) bridge, churn 1, uniform grid t_k = k/N, denoiser time clamped to [0.01, 0.99]. One NFE = one denoiser call
    (asserted by the sampler).
  - 10 repeats per run (sampler seed 1e6 + 1000 r + NFE, reference set r); the sampler seeds are identical across arms.
  - NFE grid {5, 10, 20, 30, 50, 70, 100, 200, 500, 1000} ⊃ Table 2's {5, 10, 50, 100, 1000}. NFE 1000 is also run with `clamp_low=False`.
- **Metric:** RBF MMD², σ = 1 (the paper's only defined metric, §6.1). Unbiased MMD²_u is primary; the biased MMD²_b is on the same samples;
  σ = 0.5 and 2 are stored too (`cb_mmd_raw.csv`). Why we cannot pin down Table 2's convention: `audit.md` §5.

Scripts: `scripts/eval_cb.py` (mmd, null, big, snaps), `scripts/cb_extras.py` (oracle, grid, churn0, scale, dnerr),
`scripts/diag_finetune.py`, `scripts/plot_cb.py`.

## 2. Table 2
`runs/paper/eval_cb/table2.md` (generated), `table2_summary.json`, raw `cb_mmd_raw.csv`. Per seed: the mean of 10 repeats.

| NFE | arm | MMD²_u seed 0 / 1 / 2 | MMD²_u mean (sd over seeds) | MMD²_b mean | in-support fraction | **paper** Table 2 (not ours) | numeric comparison |
|---|---|---|---|---|---|---|---|
| 5 | A | 2.83e-02 / 3.14e-02 / 3.16e-02 | **3.04e-02** (1.8e-03) | 3.08e-02 | 0.889 | 7.00e-03 | unverified |
| 5 | B | 3.74e-02 / 3.46e-02 / 3.12e-02 | 3.44e-02 (3.1e-03) | 3.49e-02 | 0.877 | — | — |
| 5 | C | 2.03e-04 / 2.54e-04 / 4.61e-04 | **3.06e-04** (1.4e-04) | 7.50e-04 | 0.917 | 1.57e-04 | unverified |
| 10 | A | 5.46e-03 / 8.40e-03 / 7.86e-03 | **7.24e-03** (1.6e-03) | 7.68e-03 | 0.939 | 2.10e-03 | unverified |
| 10 | B | 9.07e-03 / 9.34e-03 / 8.49e-03 | 8.97e-03 (4.3e-04) | 9.41e-03 | 0.933 | — | — |
| 10 | C | 2.71e-04 / 3.73e-04 / 3.63e-04 | **3.36e-04** (5.7e-05) | 7.80e-04 | 0.926 | 1.96e-04 | unverified |
| 50 | A | 7.43e-04 / 5.08e-04 / 7.29e-04 | 6.60e-04 (1.3e-04) | 1.10e-03 | 0.959 | 2.70e-04 | unverified |
| 50 | B | 6.59e-04 / 6.62e-04 / 5.26e-04 | 6.16e-04 (7.7e-05) | 1.06e-03 | 0.955 | — | — |
| 50 | C | 3.27e-04 / 2.46e-04 / 5.65e-04 | 3.79e-04 (1.7e-04) | 8.23e-04 | 0.936 | 1.55e-04 | unverified |
| 100 | A | 8.78e-04 / 6.77e-04 / 6.02e-04 | 7.19e-04 (1.4e-04) | 1.16e-03 | 0.956 | 1.96e-04 | unverified |
| 100 | B | 8.86e-04 / 5.46e-04 / 5.26e-04 | 6.53e-04 (2.0e-04) | 1.10e-03 | 0.953 | — | — |
| 100 | C | 4.56e-04 / 2.52e-04 / 5.73e-04 | 4.27e-04 (1.6e-04) | 8.71e-04 | 0.936 | 2.00e-04 | unverified |
| 1000 | A | 3.69e-03 / 3.41e-03 / 2.75e-03 | 3.28e-03 (4.8e-04) | 3.73e-03 | 0.915 | 1.71e-04 | unverified |
| 1000 | B | 3.87e-03 / 2.20e-03 / 2.20e-03 | 2.76e-03 (9.6e-04) | 3.20e-03 | 0.918 | — | — |
| 1000 | C | 1.65e-03 / 9.71e-04 / 1.38e-03 | 1.33e-03 (3.4e-04) | 1.78e-03 | 0.899 | 1.83e-04 | unverified |
| 1000, `clamp_low=False` | A | 1.78e-02 / 8.32e-03 / 1.67e-02 | 1.43e-02 (5.2e-03) | 1.48e-02 | 0.699 | — | — |
| 1000, `clamp_low=False` | B | 2.03e-02 / 1.66e-02 / 1.54e-02 | 1.75e-02 (2.6e-03) | 1.79e-02 | 0.631 | — | — |
| 1000, `clamp_low=False` | C | 3.15e-03 / 2.15e-03 / 4.13e-03 | 3.14e-03 (9.9e-04) | 3.59e-03 | 0.829 | — | — |

The ground-truth in-support fraction is 1 by construction. Monte Carlo noise: one 4096-vs-4096 evaluation of MMD²_u has sd 1.8e-4 under the
p_0-vs-p_0 null (20 draws, `mmd_null.json`), and the repeat-to-repeat sd of a model is 2.5–3e-4 (C, every NFE), 2.4e-3 (A, NFE 5) and
2.8e-4 (A, NFE 50–100). The standard error of a per-seed mean of 10 repeats is therefore ≈ 1e-4 at NFE ≥ 50.

**Scale-free comparison: the ratio Diffusion / Distributional (A / C)** (a constant normalisation such as D_ρ = ½MMD² cancels):

| NFE | ours, MMD²_u (mean of seeds) | ours, per seed | paper |
|---|---|---|---|
| 5 | **99×** | 139 / 124 / 69 | **45×** |
| 10 | **22×** | 20 / 23 / 22 | **11×** |
| 50 | 1.7× | 2.3 / 2.1 / 1.3 | 1.7× |
| 100 | 1.7× | 1.9 / 2.7 / 1.1 | 0.98× |
| 1000 | 2.5× | 2.2 / 3.5 / 2.0 | 0.93× |

## 3. Figures
- `figures/cb/ground_truth.png`: the training set, the evaluation reference set and the training density.
- `figures/cb/metric_vs_nfe.png`: three panels: ours MMD²_u, ours MMD²_b (with the V-statistic null band), and the **paper**'s Table 2 in a
  separate panel. Lines are seed means, bands the seed min–max, open squares NFE 1000 with `clamp_low=False`.
- `figures/cb/samples_grid_seed{0,1,2}.png`: rows A, B, C; columns ground truth, NFE 5, 10, 50, 100, 1000 (repeat-0 samples; MMD²_u and
  in-support fraction in each title). The samples are in `samples_cb_*.npz`.
- `figures/cb/snapshots_mmd2.png`: MMD²_u of the EMA snapshots at 10k … 100k steps (NFE 5, 10, 100). Points at 1e-6 are clipped negatives.
- `figures/cb/denoiser_error_vs_t.png`: denoiser accuracy vs t against the exact posterior (§5.3).
- Training-loss curves of the 9 checkerboard runs: `figures/gmm/loss/loss_cb_*.png` and `figures/gmm/loss_overview_cb.png` (made by S3).

## 4. What reproduces
- **The low-NFE claim reproduces qualitatively, for all 3 seeds [E].** At NFE 5, C reaches MMD²_u 3.1e-4 and A 3.0e-2 (99×). At NFE 10 it is
  3.4e-4 vs 7.2e-3 (22×). The seed ranges are disjoint at NFE 5 and 10 (A: 2.8–3.2e-2 and 5.5–8.4e-3; C: 2.0–4.6e-4 and 2.7–3.7e-4).
  This also holds for MMD²_b, for σ = 0.5 and 2, and at n = 10,000 (`table2_summary.json` keys `mmd2_*_s0.5`, `mmd2_*_s2`, `big_n10000`).
  - The samples show why (`samples_grid_seed0.png`). A and B at NFE 5 are blurred and pulled toward the centre (per-coordinate std ≈ 1.77 vs
    the true 2.28): this is posterior-mean shrinkage. C's NFE-5 samples already fill the eight squares.
- **C is flat in NFE from 5 to 100 [E]** (3.1e-4 → 4.3e-4, within ≈ 1 sd of each other). The paper's distributional column is also flat
  (1.55–2.0e-4).
- **B behaves like A [E]** (NFE 5: 3.4e-2; NFE 10: 9.0e-3). As on the GMM, the gain of C comes from the loss, not from the ξ input.
- **A catches up with C at NFE ≈ 50 [E]:** the A / C ratio is 1.7 at NFE 50 in ours and in the paper.
- **What does not reproduce [E]:**
  - our ratio at NFE 5–10 is 2× *larger* than the paper's;
  - our A stays 1.7–2.5× above C at NFE 100–1000, while in the paper the two tie (0.93–0.98×);
  - all our absolute values are higher than the paper's (§5).

## 5. Discrepancies and how we investigated them

### 5.1 A at NFE 5–10 is 3–4× above the paper: it is at the sampler's floor
We drove the same sampler with the **exact** checkerboard posterior instead of a network. `src/ddm2d/posterior_cb.py` is a mixture of
truncated normals over the 8 squares, tested against importance sampling (`tests/test_posterior_cb.py`). Raw data: `oracle_mmd.json`
(5 repeats).

| sampler input (uniform grid, clamp) | NFE 5 | 10 | 20 | 30 | 50 | 100 | 200 | 500 | 1000 | 1000, no clamp |
|---|---|---|---|---|---|---|---|---|---|---|
| exact E[X_0\|x_t] (ideal A/B), churn 1 | 3.3e-2 | 7.7e-3 | 1.6e-3 | 5.4e-4 | 8e-5 | −6e-5 | 4e-6 | −3e-5 | 2.9e-4 | 4.8e-5 |
| exact draw of p(X_0\|x_t) (ideal C), churn 1 | −5e-5 | −1e-5 | −8e-5 | 5e-5 | −2e-5 | −9e-5 | −6e-5 | 2.1e-4 | 4.2e-4 | −4.8e-5 |
| exact E[X_0\|x_t], churn 0 | 2.0e-2 | 4.7e-3 | 1.0e-3 | 3.6e-4 | 9e-5 | −6e-5 | −3e-5 | −4e-5 | 1.2e-4 | −7e-5 |

- Our trained A (3.0e-2 / 7.2e-3) is **at the exact-mean oracle** (3.3e-2 / 7.7e-3): per seed, A is 0.86–0.96× of it at NFE 5 and 0.71–1.09×
  at NFE 10, and B is 0.95–1.14× and 1.10–1.21×. Its few-step error is the discretisation error of a posterior-mean sampler, not a training deficit. The EMA snapshots confirm
  this: A at NFE 5 is ≈ 3e-2 from 10k steps on (`snapshots_mmd2.png`).
- The paper's A (7.0e-3 / 2.1e-3) is 3–5× *below* this oracle. No x_0-regression network with this sampler can get there on our board with
  σ = 1. Things we tried, identically for every arm and the oracles:
  - **churn 0** (DDIM-type): oracle 2.0e-2 / 4.7e-3; trained A 1.5e-2 / 3.9e-3 (`churn0_mmd.json`). Closer, but still ≥ 2× above the paper.
    Churn 0 makes C much worse (2.8e-3 at NFE 5), as on the GMM.
  - **grid restricted to the safety interval** ([0] + linspace(0.01, 0.99, N), `grid_mmd.json`): the oracle gets worse at NFE 5 (5.0e-2).
  - **a smaller board** (×0.5, ×0.25; `scale_oracle_mmd.json`): oracle at NFE 5 is 3.8e-2 and 2.0e-2. Not enough.
  - **a different metric:** a constant factor (½) or a different σ cannot bring A down 4× while leaving C where it is. At σ = 0.5, A is
    1.4e-2 at NFE 5.
- **Conclusion: unresolved [E for the oracle, G for the cause].** The paper's diffusion baseline must use a sampler, grid, data scale or metric
  that we cannot identify from the text. It affects only the size of the gap, not its direction: even the paper's own ratio is 45×.

### 5.2 C is ≈ 2× above the paper's distributional column at NFE 5–100
- Ours: 3.1–4.3e-4 (MMD²_u, n = 4096). At n = 10,000 it is 2.3–6.7e-4 per seed (`big_cb_*.json`), so this is real model error, not noise.
  The exact-draw oracle is ≈ 0 at every NFE ≤ 200, so the sampler is not the cause.
- The paper's 1.55–2.0e-4 is about the V-statistic floor of n ≈ 10,000 samples (1.7e-4, `audit.md` §5.2). Its C may be better than ours, or its
  numbers may be dominated by the estimator floor. The lower-lr diagnostic (§5.5) shows that our C reaches 5e-5–1.6e-4 once its weights
  converge further, so a better-converged C is a sufficient explanation **[E for ours, G for the paper's]**.
- Where our C loses accuracy: at low noise it is over-dispersed (§5.3). Its draws have std ≈ 0.06 at every t ≤ 0.02, while the exact posterior
  has 0.001–0.02. This blurs the square edges: the in-support fraction is 0.92–0.94, not 1. The GMM's C shows the same (std 0.036 vs 0.009 at
  t = 0.01; `results_gmm.md` §4), and so does the paper's own Fig. 3 right for the GMM (0.039 vs 0.012).

### 5.3 NFE ≥ 200 gets worse for every arm
- MMD²_u rises from NFE 100 to 1000: A 7.2e-4 → 3.3e-3, B 6.5e-4 → 2.8e-3, C 4.3e-4 → 1.3e-3. The paper's values instead stay at
  1.7–2.0e-4. A and B at NFE 1000 show filament artefacts (`samples_grid_seed0.png`, last column).
- **Not the clamp alone [E].** With exact posterior draws the clamp costs only 2–4e-4 at NFE 500–1000. Without the clamp the oracle is ≈ 0,
  but the trained networks get much *worse* (A 1.4e-2, C 3.1e-3; they are fed t below their training range).
- **Safety-interval grid** (no denoiser call below t = 0.01, last step to 0; `grid_mmd.json`, 3 repeats): NFE 1000 improves to A 1.2e-3,
  B 1.0e-3, C 5.1e-4, with the oracle at ≈ 0. A and B stay above their NFE-100 values on this grid (5.4e-4, 7.2e-4);
  C is about level (4.6e-4 → 5.1e-4).
- **Cause: the networks' low-noise accuracy [E].** See `denoiser_err.json` and `figures/cb/denoiser_error_vs_t.png` (4096 x_t per t):
  - At t = 0.01, A's RMS error against the exact posterior mean is 0.048, while the posterior std is 0.010. B's is 0.046; C's draws have
    std 0.059.
  - Below t = 0.01, outside the training range, the error grows to 0.14–0.20 (A, B) and 0.06–0.08 (C) if the true t is fed.
  - With churn 1, a step from t to t − h weights x̂ by ≈ 2h/t. Over a fine grid the denoiser's low-noise error is applied with total weight
    ≈ 2·ln(t_max/t_min) **[derivation]**. This integrated error grows with NFE as the grid resolves smaller t. It is the same low-noise floor
    that S1 and S3 saw on the GMM (`results_gmm.md` §3.3b).
- **Is the floor optimisation noise?** Largely yes for C, partly for A: see the diagnostic in §5.5.

### 5.4 Training length and stability
- EMA snapshots at 10k … 100k steps (`snaps_cb_*.json`, 2 repeats, NFE 5 / 10 / 100): from 20k steps on there is no trend in any arm.
  A and B stay at their oracle level throughout (NFE 5: 2.6–3.7e-2; NFE 10: 5.7e-3–1.0e-2). C fluctuates between −9e-5 and 9.5e-4 at
  NFE 5 and 10 (mean of 2 repeats), which is within single-evaluation noise.
- Loss curves: like on the GMM, the β = 2 arms (A, B) have isolated loss spikes and C has none (`figures/gmm/loss_overview_cb.png`, S3).

### 5.5 Diagnostic: lower learning rate after 100k steps
`scripts/diag_finetune.py` + `scripts/diag_eval.py`; raw `runs/diag/diag_eval.json`, `runs/diag/*/DONE.json`, `runs/diag/*/log.jsonl`.
**Diagnostic only: one seed per arm (cb_A_s0, cb_C_s0). It is not a paper run and is not in Table 2.**
- What was done: each run continued from its own 100k `final.pt` (model, EMA, Adam state, data order, RNG) for 10k more steps. The only
  change was a constant lr of 1e-4 instead of 1e-3. EMA 0.99, loss, batch and clipping were unchanged. Both arms got exactly the same treatment.
- Evaluation: the Table 2 protocol (same seeds, 10 repeats, MMD²_u) and the denoiser error vs the exact posterior. Re-evaluating the original
  checkpoints reproduces their Table 2 values exactly (A 2.83e-2, C 2.03e-4 at NFE 5), which checks determinism.

| run | NFE 5 | 10 | 50 | 100 | 1000 | t = 0.01: RMS err of E[x̂] | t = 0.01: draw std (exact 0.010) |
|---|---|---|---|---|---|---|---|
| cb_A_s0, 100k (paper config) | 2.83e-2 | 5.46e-3 | 7.43e-4 | 8.78e-4 | 3.69e-3 | 0.047 | 0 |
| cb_A_s0, + 10k at lr 1e-4 | 3.10e-2 | 6.34e-3 | 4.29e-4 | 4.26e-4 | 2.46e-3 | 0.036 | 0 |
| cb_C_s0, 100k (paper config) | 2.03e-4 | 2.71e-4 | 3.27e-4 | 4.56e-4 | 1.65e-3 | 0.027 | 0.062 |
| cb_C_s0, + 10k at lr 1e-4 | **6.1e-5** | **1.56e-4** | **4.7e-5** | **7.4e-5** | 7.47e-4 | 0.013 | 0.027 |
| paper Table 2 (not ours; A = Diffusion, C = Distributional) | 7.00e-3 / 1.57e-4 | 2.10e-3 / 1.96e-4 | 2.70e-4 / 1.55e-4 | 1.96e-4 / 2.00e-4 | 1.71e-4 / 1.83e-4 | — | — |

Findings **[E, one seed per arm]**:
- **A large part of the low-noise floor is optimisation noise** from the constant lr 1e-3 with the short EMA (0.99 averages about 100 steps).
  With lr 1e-4:
  - C's low-noise spread halves (0.062 → 0.027) and its mean error halves (0.027 → 0.013);
  - C's MMD²_u at NFE 5–100 drops 2–7×, to 5e-5–1.6e-4, at or below the paper's distributional column;
  - A improves where it is not sampler-bound (NFE 50–100: 7–9e-4 → 4.3e-4);
  - A at NFE 5–10 does not move, because it is at the exact-mean oracle floor (§5.1).
- So the few-step gap between C and A grows (A / C ≈ 500× at NFE 5).
- NFE 1000 improves (A 3.7e-3 → 2.5e-3, C 1.65e-3 → 7.5e-4) but still degrades relative to NFE 100. The networks remain coarser than the
  posterior at t ≈ 0.01, whose std is 0.010.
- We did **not** apply this to the paper runs. The paper specifies a constant lr 1e-3 with EMA 0.99 (App. I.1), and Table 2 must keep the
  paper's configuration. The diagnostic shows that our C's distance from the paper's distributional column is mostly a convergence matter **[G
  for the paper's own runs]**.

## 6. Seed variability (task spec §8)
- At NFE 5 and 10, the A-vs-C ordering is the same for every seed. Over all 9 (A seed, C seed) pairs the factor A / C is 61–156 at NFE 5 and 15–31 at NFE 10. The per-seed values are in
  the Table 2 above and in `table2_summary.json` (`per_seed`, `sd_over_seeds`, `sd_over_repeats_pooled`).
- The sd over seeds of the 10-repeat means: A 1.8e-3 and C 1.4e-4 at NFE 5; 0.8–2e-4 for every arm at NFE 50–100; 3.4–9.6e-4 at NFE 1000.
- The paper's Table 2 is presumably one training run and one evaluation (not stated). Its values are kept in their own column and file.

## 7. Reproduce
```bash
PY=/scratch/wg2381/.conda/envs/tmap/bin/python; cd ddm2d
# Table 2 + curve + samples: 4 processes x 2 threads, ~20 min on 8 CPUs
$PY scripts/eval_cb.py mmd --runs cb_A_s0 cb_C_s2 --threads 2 --repeats 10 &   # ... and the other run lists (all 9 runs)
$PY scripts/eval_cb.py null  --threads 2 --repeats 20        # p0-vs-p0 floors, n in {1000, 2048, 4096, 10000}
$PY scripts/eval_cb.py big   --threads 2 --repeats 3         # n = 10,000
$PY scripts/eval_cb.py snaps --threads 2                     # EMA snapshots
$PY scripts/cb_extras.py oracle --repeats 5; $PY scripts/cb_extras.py grid --repeats 3; $PY scripts/cb_extras.py churn0 --repeats 3
$PY scripts/cb_extras.py scale --repeats 3; $PY scripts/cb_extras.py dnerr --threads 4
$PY scripts/diag_finetune.py --run cb_A_s0 --lr 1e-4 --steps 10000; $PY scripts/diag_finetune.py --run cb_C_s0 --lr 1e-4 --steps 10000
$PY scripts/plot_cb.py all                                   # figures/cb/*, table2.md, table2_summary.json, cb_mmd_raw.csv
```
