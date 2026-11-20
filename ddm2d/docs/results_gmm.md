# GMM results (S3): Fig. 3 left, Fig. 3 right, Fig. 4

Paper: De Bortoli et al., *Distributional Diffusion Models with Scoring Rules*, arXiv 2502.02483 v3, §6.1, Fig. 3–4, App. I.1.
All numbers below come from committed files under `runs/paper/eval_gmm/` (raw JSON/CSV) and `figures/gmm/`. Labels: **[E]** = evidence in a
committed file, **[G]** = guess / assumption.

**Paper values.** The paper's Fig. 3 is a vector PDF in the arXiv source (`figures/mmd_2d.pdf`, `figures/std_over_eps_avg_over_xt.pdf`).
`scripts/extract_paper_fig3.py` reads the plotted polylines and the tick marks out of the PDF content stream, so the "paper" values below are the
plotted coordinates. They are exact up to the PDF's 6-decimal coordinates, but they are still read off a figure, from what is presumably a single run
and a single evaluation. They are never our results. File: `docs/paper_values/fig3_paper_values.json`. Fig. 4 is a PNG, so we compare it only
qualitatively.

## 1. Runs used
- 9 GMM runs: arms A (standard diffusion, Eq. 6), B (β = 2, λ = 0, m = 32), C (β = 0.1, λ = 1, m = 32) × training seeds 0, 1, 2.
  All 9 reached 100k steps. Their final checkpoints load and match `DONE.json` (`runs/paper/STATUS.md`, 18/18 complete incl. checkerboard) **[E]**.
- Every evaluation uses the **EMA weights of `final.pt`** (100k steps). Every arm gets the same sampler, NFE grid, sample count, seeds and x_t
  (`scripts/eval_gmm.py`; protocol in `docs/plan.md`).
- Sampler: Eq. (4) bridge on the uniform grid t_k = k/N, churn ε = 1. Arm A uses x̂(t, x_t); arms B and C use x̂(t, x_t, ξ) with fresh ξ
  (Algorithm 2). One NFE = one denoiser call.

## 2. Training-loss curves
`figures/gmm/loss/loss_<run>.png` (all 18 paper runs, checkerboard included), `figures/gmm/loss_overview_{gmm,cb}.png`, `runs/paper/eval_gmm/loss_summary.json`.

| run | mean loss, steps 90k–100k | largest 100-step window loss after 2k steps |
|---|---|---|
| gmm_A_s0 / s1 / s2 | 0.207 / 0.207 / 0.357 | 8.3e4 / 1.3e4 / 4.7e3 |
| gmm_B_s0 / s1 / s2 | 2795 / 0.242 / 0.369 | 2.8e5 / 4.5 / 9.2e4 |
| gmm_C_s0 / s1 / s2 | 0.226 / 0.226 / 0.227 | 0.23 / 0.23 / 0.23 |

- **Finding: loss spikes in the β = 2 arms (A and B), never in C [E].** A and B show 3–6 isolated 100-step windows with a huge loss
  (up to 2.8e5) and a huge gradient norm (up to 1.7e10, before clipping to 1). The next logged window is back at ≈ 0.21. The 10k-step medians are
  flat at 0.207–0.209 from 20k steps on. The large 90k–100k means of gmm_B_s0 (2795), gmm_A_s2 and gmm_B_s2 come from one spike window each
  (at 97.6k, 99.1k and 96.5k). The β = 0.1 loss of C has bounded gradients (‖d‖^0.1) and never spikes (max gradient norm 3.5–10.6).
  Likely cause **[G]**: Adam at a constant lr 1e-3 (no decay, as in App. I.1) after a long quiet phase with small second-moment estimates.
  This is a property of the paper's configuration. We did not change it and did not relaunch, because every run finished as specified.
- Do spikes reach the evaluated weights? EMA decay 0.99 forgets a spike after about 1000 steps (0.99^1000 ≈ 4e-5). The latest spikes before
  100k are at 99.1k (gmm_A_s2) and 98.4k (gmm_B_s1). The EMA snapshot study (§3.3) shows a spike *does* reach a snapshot: gmm_A_s2 at 50k
  (spike at 49.9k) has MMD² ≈ 0.1 at NFE 50–100. The final checkpoints of these runs are in line with the other seeds (§3.1 table).

## 3. Fig. 3 left: MMD² vs sampling steps
Files: `figures/gmm/fig3_left.png` (ours | zoom | paper), `figures/gmm/fig3_left_per_seed.png`, raw `runs/paper/eval_gmm/fig3_left_mmd2_raw.csv`,
summary `fig3_left_summary.json`. Metric: unbiased MMD²_u (RBF σ = 1, n = m = 4096), against fresh p_0 reference sets (never training points).
The biased MMD²_b and D_ρ = ½MMD² are in the CSV too. Each seed's value is the mean of 10 repeats (sampler seed 1e6 + 1000 r + NFE,
reference set r). The sampler seeds are identical across arms.

### 3.1 Numbers (MMD²_u; per seed = mean of 10 repeats)

| arm | NFE | seed 0 | seed 1 | seed 2 | mean of seeds | sd over seeds | paper (read off Fig. 3) |
|---|---|---|---|---|---|---|---|
| A | 5 | 1.81e-2 | 1.54e-2 | 1.64e-2 | **1.66e-2** | 1.4e-3 | 1.62e-2 |
| A | 10 | 4.97e-3 | 4.14e-3 | 5.06e-3 | **4.72e-3** | 5.1e-4 | 2.35e-3 |
| A | 20 | 2.08e-3 | 1.50e-3 | 2.15e-3 | 1.91e-3 | 3.6e-4 | 1.02e-3 |
| A | 30 | 1.26e-3 | 6.06e-4 | 1.04e-3 | 9.7e-4 | 3.3e-4 | 6.1e-4 |
| A | 50 | 7.12e-4 | 2.37e-4 | 6.05e-4 | 5.2e-4 | 2.5e-4 | 6.9e-4 |
| A | 70 | 6.30e-4 | 3.53e-4 | 7.35e-4 | 5.7e-4 | 2.0e-4 | 2.7e-4 |
| A | 90 | 6.81e-4 | 6.84e-4 | 1.02e-3 | 8.0e-4 | 2.0e-4 | 2.8e-4 |
| A | 100 | 6.28e-4 | 4.95e-4 | 7.20e-4 | 6.2e-4 | 1.1e-4 | 2.0e-4 |
| B | 5 | 1.84e-2 | 1.74e-2 | 1.63e-2 | **1.74e-2** | 1.1e-3 | 1.59e-2 |
| B | 10 | 5.12e-3 | 6.06e-3 | 5.47e-3 | **5.55e-3** | 4.7e-4 | 2.49e-3 |
| B | 20 | 1.65e-3 | 2.28e-3 | 2.10e-3 | 2.01e-3 | 3.3e-4 | 5.5e-4 |
| B | 30 | 1.05e-3 | 1.53e-3 | 1.49e-3 | 1.36e-3 | 2.6e-4 | 2.7e-4 |
| B | 50 | 5.75e-4 | 1.32e-3 | 9.39e-4 | 9.5e-4 | 3.7e-4 | 1.08e-3 |
| B | 70 | 2.31e-4 | 1.09e-3 | 7.01e-4 | 6.7e-4 | 4.3e-4 | 1.0e-4 |
| B | 90 | 6.06e-4 | 1.69e-3 | 1.05e-3 | 1.12e-3 | 5.5e-4 | 1.3e-4 |
| B | 100 | 8.21e-4 | 1.25e-3 | 8.52e-4 | 9.8e-4 | 2.4e-4 | 1.3e-4 |
| C | 5 | 1.37e-4 | 1.93e-4 | 2.22e-4 | **1.8e-4** | 4.3e-5 | 1.6e-4 |
| C | 10 | 5.33e-5 | 2.92e-4 | 5.94e-4 | **3.1e-4** | 2.7e-4 | 4.6e-4 |
| C | 20 | 1.03e-4 | 2.82e-4 | 4.17e-4 | 2.7e-4 | 1.6e-4 | 5.9e-4 |
| C | 30 | 6.66e-5 | 2.18e-4 | 3.32e-4 | 2.1e-4 | 1.3e-4 | 5.5e-4 |
| C | 50 | 4.10e-4 | 4.76e-4 | 3.62e-4 | 4.2e-4 | 5.7e-5 | 1.1e-4 |
| C | 70 | 1.94e-4 | 4.36e-4 | 2.62e-4 | 3.0e-4 | 1.3e-4 | 7.8e-4 |
| C | 90 | 4.77e-4 | 8.23e-4 | 8.32e-4 | 7.1e-4 | 2.0e-4 | 7.1e-4 |
| C | 100 | 5.68e-4 | 4.75e-4 | 3.26e-4 | 4.6e-4 | 1.2e-4 | 3.0e-4 |

**Monte Carlo noise [E]** (`mmd_null.json`, `fig3_left_mmd2_raw.csv`). Two independent 4096-sample sets from p_0 give MMD²_u between −1.9e-4 and
+3.7e-4 (10 draws). The repeat-to-repeat sd of one model's MMD²_u is 2–7e-4 at NFE ≥ 20. So a *single* 4096-sample evaluation, as in the paper,
cannot resolve differences below ≈ 5e-4. Our per-seed means of 10 repeats have a standard error of ≈ 1e-4.

### 3.2 Agreement with the paper
- **Main claim reproduced [E].** With few steps the distributional model is far better: at NFE 5, C reaches 1.8e-4, while A reaches 1.66e-2
  and B 1.74e-2, about 90× worse. At NFE 10, C is ≈ 15× better (3.1e-4 vs 4.7e-3 and 5.6e-3). This holds for all three seeds, and the
  seed ranges do not overlap at NFE 5–30 (`fig3_left.png`). Paper: C 1.6e-4, A 1.62e-2, B 1.59e-2 at NFE 5.
- **NFE 5 matches numerically** for all three arms: ours 1.66e-2 / 1.74e-2 / 1.8e-4 vs paper 1.62e-2 / 1.59e-2 / 1.6e-4.
- **B behaves like A [E]**, as in the paper. The β = 2, λ = 0 control with the stochastic architecture has the same few-step error as standard
  diffusion. So the gain of C comes from the loss (λ = 1, β = 0.1), not from the ξ input.
- **Large NFE.** All arms end at 2e-4 to 1e-3 at NFE 50–100. The paper shows 1e-4 to 1.1e-3 there, also not monotone. Within MC noise, the arms
  are tied at NFE ≥ 50 in both ours and the paper's figure ("achieves similar performance as the diffusion model using a large number of steps",
  §6.1).

### 3.3 Discrepancies and how we investigated them
**(a) A and B at NFE 10–30 are ≈ 2× (A) to 4× (B, NFE 20–30) above the paper's curve.**
- *Is the network undertrained?* No **[E]**. The EMA snapshots at 10k, 20k, …, 100k (`snapshots_mmd2.png`, `snapshots_mmd2_raw.csv`) give the
  same A/B values at NFE 5 and 10 from 10k steps on (≈ 1.7e-2 and ≈ 5e-3). There is no trend with training length.
- *What is the best possible value for this sampler?* We drove the same sampler with the **exact** posterior mean E[X_0 | x_t] (an oracle for
  A/B) and with exact posterior draws (an oracle for C) (`scripts/gmm_extras.py oracle`, `oracle_mmd.json`, 5 repeats).

  | sampler input (churn 1) | NFE 5 | 10 | 20 | 30 | 50 | 70 | 90 | 100 |
  |---|---|---|---|---|---|---|---|---|
  | exact E[X_0\|x_t] (ideal A/B) | 1.7e-2 | 4.9e-3 | 1.4e-3 | 6.6e-4 | 1.3e-4 | 3.0e-4 | 2.1e-5 | 7.5e-5 |
  | exact draw of p(X_0\|x_t) (ideal C) | 1.8e-4 | −3e-5 | −4e-5 | −1.1e-4 | 5e-5 | −1.3e-4 | −1e-5 | −5e-5 |

  Our trained A and B are **at the oracle level at NFE 5 and 10**: 0.90–1.08× at NFE 5 and 0.84–1.24× at NFE 10 for all 6 runs. At NFE 20–30
  they are 0.9–2.3× the oracle. So the ≈ 5e-3 at NFE 10 is the discretisation error of the posterior-mean sampler itself, not a training
  deficit. The exact-draw oracle is ≈ 0 at every NFE: Algorithm 2 with exact posterior draws is exact at any step count, which is the
  mechanism the paper claims.
- The paper's A and B at NFE 10 (2.35e-3, 2.49e-3) are about 2× *below* the exact-mean oracle. A trained network is not strictly bounded by this
  oracle: a biased x̂ can partly cancel discretisation error. But all 6 of our A/B runs sit within −16 %…+24 % of it at NFE 10.
- *Sampler variants (same for all arms)*:
  - churn 0 (DDIM-type, `churn0_mmd.json`): A 4.3e-3, B 4.4e-3 at NFE 10, and 1.1e-3 at NFE 20;
  - the oracle on a grid restricted to the safety interval [0.01, 0.99] (`oracle_grid_mmd.json`): 5.7e-3 at NFE 10, and worse at NFE 5 (2.5e-2).

  None reaches 2.35e-3 at NFE 10.
- *Metric normalisation.* The paper says it plots "the MMD squared given by D_ρ" with ρ = −k. By Eq. (10), D_ρ = ½·MMD² (audit §1.4). If the
  paper's y-axis is D_ρ, its A values would be MMD² = 3.2e-2, 4.7e-3, 2.0e-3, 1.2e-3 at NFE 5, 10, 20, 30. NFE 10–30 then match ours
  (4.7e-3, 1.9e-3, 9.7e-4), but NFE 5 is 2× worse than ours. Neither normalisation fits all NFE.
- **Conclusion:** unresolved. The most likely sources are the normalisation (D_ρ vs MMD²) or a sampler detail for the baseline that the
  paper does not give (grid, churn, final step) **[G]**. Either one affects A and B in the same way and does not change the qualitative result.

**(b) Low-noise floor at NFE ≥ 50 (S1 open point 1).**
- Ours: A 5–8e-4 and B 7e-4–1.1e-3 at NFE 50–100, while the exact-mean oracle reaches 2e-5 to 1.3e-4. C is 3–7e-4, while its oracle is ≈ 0.
- So at many steps all three networks are limited by their accuracy at small t, not by the sampler **[E]**. Fig. 3 right shows this directly
  for C: at t = 0.01 its std is 0.036, vs the exact 0.009.
- The snapshot study shows no improvement from 40k to 100k steps at NFE 50–100.
- The paper's values at NFE ≥ 50 (1e-4 to 1.1e-3, non-monotone) are of the same order. With 1 evaluation of 4096 samples they are within MC
  noise of ours.
- In S1's 20k-step smoke runs, MMD² grew with NFE beyond 50. This persists at 100k only as a mild rise in A, B and C between NFE 50 and 90–100,
  within about 2 MC standard errors.

**(c) C at NFE 10–30: ours 2–3e-4, paper 4.6–5.9e-4.** Both are within single-evaluation noise (≈ ±2–4e-4) of each other.

### 3.4 Churn-0 sensitivity (extra, same for every arm)
`churn0_mmd.json` (3 repeats). A and B are slightly better with churn 0 (NFE 10: 4.3e-3 and 4.4e-3). C is clearly worse: 1.7–2.9e-3 at every
NFE, vs 2–7e-4 with churn 1. The exact-draw oracle is ≈ 0 with churn 0 as well, so the extra error of C comes from its imperfect x̂ at
small/large t, which churn 1 averages out **[G]**. The paper's 2D setting does not state the churn. We use ε = 1 (plan, §"Time, weighting,
safety"); the main Fig. 3 uses churn 1 for all arms.

## 4. Fig. 3 right: posterior std vs t
Files: `figures/gmm/fig3_right.png` (ours | paper), raw `runs/paper/eval_gmm/posterior_std.json` (paper grid and a fine grid of 50 points),
`fig3_right_summary.json`.
- Protocol: for each t, 4096 pairs (x_0, x_t), the same for every arm, and 8 draws per x_t. The std over the 8 draws is taken per coordinate
  (ddof 0), then averaged over the 2 coordinates and the 4096 x_t. The exact posterior is evaluated with the same 8-draw protocol (solid
  green) and with its exact std (dotted).
- t grid: the paper's curves have vertices at exactly `linspace(0.01, 0.99, 10)` (read from the vector PDF) **[E]**. We use that grid, and
  the fine grid for the plot.
- **Arm A** is deterministic: its std is identically 0. The computed value is ≈ 1e-7 (float32 rounding of identical draws). It is not drawn in
  the figure and is listed in the JSON.

| t | exact (8 draws) | C s0 | C s1 | C s2 | C mean ± sd | B mean | paper exact | paper C | paper B |
|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.009 | 0.038 | 0.036 | 0.034 | 0.036 ± 0.002 | 0.0016 | 0.012 | 0.039 | 0.0036 |
| 0.12 | 0.118 | 0.127 | 0.125 | 0.124 | 0.125 ± 0.002 | 0.0016 | 0.155 | 0.131 | 0.0035 |
| 0.23 | 0.230 | 0.230 | 0.235 | 0.236 | 0.234 ± 0.003 | 0.0016 | 0.302 | 0.232 | 0.0034 |
| 0.34 | 0.322 | 0.325 | 0.325 | 0.327 | 0.326 ± 0.001 | 0.0016 | 0.423 | 0.333 | 0.0035 |
| 0.45 | 0.386 | 0.385 | 0.390 | 0.399 | 0.391 ± 0.007 | 0.0016 | 0.505 | 0.385 | 0.0039 |
| 0.55 | 0.462 | 0.477 | 0.476 | 0.474 | 0.476 ± 0.002 | 0.0019 | 0.565 | 0.487 | 0.0051 |
| 0.66 | 0.687 | 0.677 | 0.703 | 0.671 | 0.684 ± 0.017 | 0.0032 | 0.687 | 0.694 | 0.0083 |
| 0.77 | 1.122 | 1.031 | 1.088 | 1.076 | 1.065 ± 0.030 | 0.0041 | 0.972 | 1.086 | 0.0115 |
| 0.88 | 1.504 | 1.419 | 1.386 | 1.411 | 1.406 ± 0.017 | 0.0029 | 1.510 | 1.478 | 0.0104 |
| 0.99 | 1.634 | 1.426 | 1.351 | 1.401 | 1.393 ± 0.038 | 0.0012 | 1.836 | 1.627 | 0.0077 |

**Agreement.**
- **C learns the posterior spread, B collapses [E]**, as the paper claims ("distributional diffusion models have std close to that of the
  true posterior"; "classical diffusion models cannot model the variance").
- Our **C curve agrees with the paper's C curve within 0–9 %** at t ≤ 0.88: 0.036 vs 0.039, 0.125 vs 0.131, 0.234 vs 0.232, 0.326 vs 0.333,
  0.391 vs 0.385, 0.476 vs 0.487, 0.684 vs 0.694, 1.065 vs 1.086 and 1.406 vs 1.478. At t = 0.99 the gap is 14 % (1.393 vs 1.627).
- Our B is ≈ 0.002, the paper's ≈ 0.004–0.011: both are ≈ 0 on this scale.
- Seed spread of C is at most 0.04 (sd), so the seed-to-seed agreement is tight.

**Discrepancies.**
1. *C vs exact posterior at the ends [E].* At t = 0.01, C's std is 0.036 vs the exact 0.009 (≈ 4×). The paper has the same, 0.039 vs 0.012.
   At t = 0.99, C is 15 % below the exact value (1.39 vs 1.63); the paper's C is 11 % below its exact curve (1.63 vs 1.84). Both ends are
   where the sigmoid weight is extreme (w_t ≈ 1 at t = 0.01, ≈ 1e-4 at t = 0.99; S1 audit). So this reproduces a limitation of the paper's model.
2. *The paper's exact-posterior curve differs from ours [E].* At t ≤ 0.45 the paper's curve is a constant 1.31× our 8-draw exact value. At
   t = 0.77 it is 0.87×. Our C sits on our exact curve (ratio 0.99–1.06 for t ∈ [0.12, 0.66]); the paper's C sits 16–24 % below its exact curve
   at t ∈ [0.12, 0.45].
   - We tested 7 reductions of the 2-vector std (`scripts/check_std_reduction.py`, `std_reduction_check.json`).
   - "Mean Euclidean distance of the 8 draws to their mean" reproduces the paper's exact curve to within 1 % for t ≤ 0.45, but not at high t
     (0.60–0.92×).
   - No single reduction reproduces both the paper's exact curve and its C curve with our data.
   - With either reduction, our C matches our exact posterior at t ∈ [0.12, 0.66] (the `models` block of the same JSON).
   - So the paper's apparent "C slightly below the posterior" gap at mid t is most likely a different reduction or estimator for the posterior
     curve than for the model curves **[G]**. It cannot be checked without the paper's code. App. I.1 only says "compute their standard
     deviation and then average".

## 5. Fig. 4: conditional posterior samples
Files: `figures/gmm/fig4_seed0.png` (main), `fig4_seed1.png`, `fig4_seed2.png`; per-panel statistics `runs/paper/eval_gmm/fig4_summary.json`.
The 4096-draw arrays (`fig4_draws.npz`, 1.8 MB) are regenerated by `scripts/eval_gmm.py fig4` and not committed.
- Layout: 2 rows × 5 columns, t = 0.99, 0.88, 0.45, 0.23, 0.01, with one x_t | x_0 per panel (seed 3e6 + 10·row + col), shared by all arms.
  Each panel has 4096 exact posterior draws (light blue), 4096 draws of C (green) and 4096 draws of B (red), with x_0 as an orange × and x_t
  as a black ×. Axes are the same in all panels.
- **Row order.** We follow `docs/task_spec.md` §6: top row x_0 = (3, 3), bottom row x_0 = (−3, 3). The paper's caption says "Top/bottom row:
  x_0 is the mean of the left/right Gaussian", so the paper's rows are in the opposite order. Each of our rows is labelled with its x_0.

Per-panel numbers (3 seeds; C's MMD²_u is vs the 4096 exact draws):

| x_0 | t | exact P(left mode) | C P(left mode) | C MMD²_u vs exact | C std (x, y) seed 0 | exact std (x, y) | B std |
|---|---|---|---|---|---|---|---|
| (3, 3) | 0.99 | 0.49 | 0.21–0.26 | 8e-2 – 1.1e-1 | (2.60, 0.45) | (3.04, 0.50) | 0.002 |
| (3, 3) | 0.88 | 0.44 | 0.43 | 0.9–3.4e-3 | (2.96, 0.47) | (3.00, 0.50) | 0.003 |
| (3, 3) | 0.45 | 0.00 | 0.00 | ≤ 7e-5 | (0.42, 0.42) | (0.43, 0.43) | 0.001 |
| (3, 3) | 0.23 | 0.00 | 0.00 | ≤ 2.7e-4 | (0.27, 0.26) | (0.26, 0.26) | 0.002 |
| (3, 3) | 0.01 | 0.00 | 0.00 | ≤ 8.5e-4 | (0.05, 0.03) | (0.01, 0.01) | 0.002 |
| (−3, 3) | 0.99 | 0.50 | 0.31–0.32 | 4.4–4.6e-2 | (2.77, 0.43) | (3.04, 0.50) | 0.001 |
| (−3, 3) | 0.88 | 0.76 | 0.74–0.83 | 0.8–2.1e-2 | (2.50, 0.50) | (2.61, 0.50) | 0.003 |
| (−3, 3) | 0.45 | 1.00 | 1.00 | ≤ 6.6e-4 | (0.43, 0.43) | (0.43, 0.43) | 0.002 |
| (−3, 3) | 0.23 | 1.00 | 1.00 | ≤ 5e-5 | (0.26, 0.26) | (0.26, 0.26) | 0.002 |
| (−3, 3) | 0.01 | 1.00 | 1.00 | ≤ 9e-5 | (0.04, 0.03) | (0.01, 0.01) | 0.001 |

**Agreement [E].**
- As in the paper, C's draws cover the exact posterior: both modes at t = 0.99 and 0.88 (with a thin bridge of draws between them, also visible
  in the paper), and a single Gaussian blob of the right size at t = 0.45 and 0.23.
- B's draws collapse to a point (std ≈ 0.002) near E[X_0 | x_t]. That point is between the modes at high t, and its distance to the exact mean
  is ≤ 0.65 at t = 0.99 and ≤ 0.04 at t ≤ 0.45.
- The panel geometry looks like the paper's (Fig. 4 PNG): two clusters at t ≥ 0.88, one shrinking cluster at t ≤ 0.45, and red points
  at the posterior mean.

**Discrepancies [E].**
- At t = 0.99, C gives the two modes unequal weight: 21–32 % on the left mode vs the exact 49–50 %. The heavier mode is the one on the same
  side as x_t. This is the high-noise limitation of §4 (std 15 % too small at t = 0.99).
- At t = 0.01, C's draws are 3–5× wider than the exact posterior (0.03–0.05 vs 0.01). This is invisible at the figure's scale.
- Both match the paper's own Fig. 3 right values at those t, so we read them as properties of the method/configuration, not of our
  implementation **[G]**.

## 6. Seed variability (summary)
- Fig. 3 left (sd over the 3 seeds of the 10-repeat means): A 1.4e-3, B 1.1e-3 and C 4e-5 at NFE 5; 1–5e-4 for every arm at NFE ≥ 20.
  At NFE 5–30, the C vs A/B ordering is the same for every seed, and the seed ranges are disjoint.
- Fig. 3 right: C's sd over seeds ≤ 0.007 for t ≤ 0.55 and ≤ 0.038 at t ≥ 0.66. B's ≤ 0.002.
- Fig. 4: P(left mode) and MMD² vs exact differ little between seeds (table above); `fig4_seed{1,2}.png` look like `fig4_seed0.png`.
- Training: A and B have seed-specific loss spikes (§2); C's final losses agree to 0.1 % across seeds (0.2264–0.2267).

## 7. Reproduce
```bash
PY=/scratch/wg2381/.conda/envs/tmap/bin/python; cd ddm2d
$PY scripts/check_complete.py --threads 8                         # runs/paper/STATUS.md
$PY scripts/eval_gmm.py mmd   --repeats 10 --threads 2            # Fig. 3 left raw + samples + p0-vs-p0 null
$PY scripts/eval_gmm.py snaps --repeats 2  --threads 2            # EMA snapshot study
$PY scripts/eval_gmm.py std   --threads 2                         # Fig. 3 right raw
$PY scripts/eval_gmm.py fig4  --threads 2                         # Fig. 4 draws + per-panel stats
$PY scripts/gmm_extras.py oracle --repeats 5; $PY scripts/gmm_extras.py grid --repeats 5; $PY scripts/gmm_extras.py churn0 --repeats 3
$PY scripts/check_std_reduction.py
$PY -I scripts/extract_paper_fig3.py third_party/arxiv_src/x/figures docs/paper_values   # needs pypdf (see third_party/README.md)
$PY scripts/plot_gmm.py all                                       # every figure in figures/gmm/
```
Wall time on 8 CPUs (4 parts in parallel, 2 threads each): mmd ≈ 11 min, snaps ≈ 18 min, std < 1 min, fig4 ≈ 1 min.
