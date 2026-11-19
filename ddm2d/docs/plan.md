# Implementation and run plan (S1, 2026-11-19)

Every choice below is either taken from the paper (see `audit.md`, cited there) or labelled **[G]** with the reason. The audit table is
`paper_vs_implementation.md`.

## File layout (all under `ddm2d/`)
```
src/ddm2d/
  schedule.py    alpha/sigma, forward sampling, Gaussian bridge Eq. (4), sigmoid weight, sampling time grid
  data.py        GMM and checkerboard generators, cached datasets (data/cache/, gitignored), eval sets, checkerboard support test
  model.py       App. I.1 MLP (stochastic and deterministic variants), sinusoidal embedding
  losses.py      Eq. (6) regression loss, Eq. (14) generalized energy loss
  samplers.py    Algorithm 2 (DDM arms) and the same bridge with a deterministic x̂ (standard arm)
  mmd.py         RBF MMD², chunked, float64 accumulation, biased and unbiased
  posterior.py   analytic GMM posterior (App. I.1), sampling, exact std
  train.py       training loop, checkpoint/resume, logging, DONE.json; CLI `python -m ddm2d.train`
  runinfo.py     git commit, threads, peak RSS, environment
configs/smoke/*.json, configs/paper/*.json
tests/test_<module>.py
runs/smoke/…  (S2), runs/paper/<dataset>_<arm>_s<seed>/ (S2 launches, S3/S4 evaluate)
```
Run: `PYTHONPATH=src /scratch/wg2381/.conda/envs/tmap/bin/python -m ddm2d.train --config configs/paper/gmm_C.json --out runs/paper/gmm_C_s0 --seed 0 --threads 2`.

## Arms
| Arm | Model | Loss | Sampler |
|---|---|---|---|
| A standard | App. I.1 MLP **without ξ input, output width 2** | Eq. (6), x_0-prediction, sigmoid weight | Eq. (4) bridge with X̂_0 = x̂_θ(t, x_t), churn ε = 1 ("SDE sampler") |
| B diffusion-like | App. I.1 MLP with ξ, output width 4, keep dims 2:4 | Eq. (14), β = 2, λ = 0, m = 32 | Algorithm 2, ε = 1 |
| C distributional | same as B | Eq. (14), β = 0.1, λ = 1, m = 32 | Algorithm 2, ε = 1 |

**Arm A backbone decision.** The paper's standard model in the 2D runs is "a baseline unconditional diffusion model by optimizing (6)", and
separately arm B is "a classical diffusion model using the same architecture as the distributional models" (§6.1). So arm A is *not* the
stochastic architecture. App. I.1 describes one MLP: the "output MLP layer of dimension 2" is the standard model, and the ξ concatenation + width 4
output is the distributional change. Arm A = the same MLP with x-branch input width 2 and output width 2. **[G, strongly suggested by the text]**

**Arm B, m.** Algorithm 1 with m = 32, the same as arm C, so B and C have identical cost and architecture; only (β, λ) differ. With λ = 0 the loss is the
regression loss averaged over 32 draws of ξ. **[G]**

## Time, weighting, safety
- Training: t ~ U[η, 1 − η], η = 0.01 (safety interval of App. K.2). **[G: exact use of η]**
- Weight w_t = sigmoid(log(α_t²/σ_t²) − b), b = 0, per data point, the same for all three arms (Eq. 6 and Eq. 14). **[G: b]**
- Sampling grid, all arms: t_k = k/N, k = 0..N (uniform, Algorithm 2 as written, t_0 = 0, t_N = 1). X_1 ~ N(0, I). The denoiser is called at
  clamp(t_{k+1}, η, 1 − η) (only t_N = 1 → 0.99 changes for N ≤ 100); the bridge uses the true grid times. The last step goes to s = 0, where
  Σ = 0 and μ = X̂_0, so the output is the last prediction. **[G]** Every denoiser call is useful, nothing divides by zero (α_s ≥ 1/N,
  σ_t ≥ 1/N).
- Churn ε = 1 for every arm (not stated for 2D; §4.1 uses ε = 1; it is the "SDE sampler" of App. I.2). **[G]** A churn sensitivity
  (ε = 0, same for all arms) is an optional extra for S3/S4.

## NFE accounting
One NFE = one forward call of the denoiser on the batch of samples. A sampler with N steps makes exactly N calls, for every arm. The sampler counts
its calls and returns the count; tests assert NFE == N.

## Model (App. I.1)
- t → 1000·t → sinusoidal 2048 ([sin, cos], max period 10000) → Linear(2048, 2048) → GELU → Linear(2048, 2048). **[G: 1000·t, max period, no
  GELU after the 2nd layer]**
- time branch: 4 × [Linear(→64), GELU] on the 2048-d embedding; x branch: 4 × [Linear(→64), GELU] on x_t (A) or [x_t, ξ] (B, C).
- concat (128) → 4 × [Linear(→64), GELU] → Linear(64 → 2 (A) or 4 (B, C)); B, C keep out[..., 2:4].
- Exact speed-up (no change to the function): the time path depends only on t, which the m particles of one data point share. We evaluate it once
  per data point and broadcast over the m particles. Per step this is 128 rows through the 2048 × 2048 layers instead of 4096.
  Parameter count: about 8.6 M (dominated by the time MLP).

## Loss details
- Eq. (14): per data point i, conf_i = mean_j ‖x_0^i − x̂^{ij}‖^β, inter_i = mean over j ≠ j' of ‖x̂^{ij} − x̂^{ij'}‖^β (U-statistic, i = j terms
  excluded), loss = mean_i w_i (conf_i − λ/2 · inter_i). ‖d‖^β = (‖d‖² + δ)^{β/2} with δ = 1e-12 for β ≠ 2, exact ‖d‖² for β = 2. **[G: δ]**
- Eq. (6): loss = mean_i w_i ‖x_0^i − x̂^i‖².

## Optimisation (App. I.1)
Adam(lr 1e-3, betas (0.9, 0.999), eps 1e-8); lr factor ½(1 − cos(π (k+1)/100)) for steps k < 100, then 1 (no decay); gradient clipping to global
norm 1 before Adam **[G]**; EMA 0.99 after every step, initialised to the initial weights; 100k steps, batch 128. EMA weights are used for every
evaluation.

## Data
- GMM: 0.5 N((3, 3), 0.25 I) + 0.5 N((−3, 3), 0.25 I); training set 102,400 points, data seed 0, fixed for all arms and training seeds.
- Checkerboard: FFJORD `inf_train_gen("checkerboard")` re-implemented with a `numpy.random.Generator`; support [−4, 4]², 8 squares of side 2.
  Training set 102,400 points (borrowed from GMM), data seed 0. **[G]**
- Batches: one random permutation of the training set per epoch (800 steps/epoch), drawn from the run's torch generator; the permutation and
  position are checkpointed.
- Evaluation reference sets: fresh i.i.d. draws from p_0 with seeds disjoint from the training data (`data.eval_set(name, n, seed)`, eval seeds
  10_000 + k). Never the training points.
- Training seed (model init, t, noise, ξ, data order) is the `--seed` of the run (0, 1, 2).

## MMD estimator
- k(x, y) = exp(−‖x − y‖²/(2σ²)), σ = 1 (Eq. 9b, §6.1).
- Reported: **unbiased** MMD²_u (U-statistic, diagonal terms removed in the k(x,x) and k(y,y) means) as the primary number, plus the **biased**
  V-statistic MMD²_b, plus D_ρ = ½ MMD² (the paper's Eq. 10 normalisation, audit §1.4), all stored in the metrics JSON.
- Why unbiased: with n = 4096 the V-statistic adds a positive bias of (1 − mean k_xx)/n + (1 − mean k_yy)/n, about 1e-4 to 4e-4 for these
  datasets, which is the size of the differences in Table 2. The unbiased estimate has expectation exactly MMD². The paper does not say which
  it used; S4 tests whether the biased convention explains Table 2's floor.
- Implementation: blocks of 2048 × 2048, kernel sums accumulated in float64; checked against a naive O(n²) float64 computation and the closed
  form for two isotropic Gaussians (tests).

## Evaluation protocol (for S3/S4; identical for every arm)
- EMA weights; 4096 generated samples; reference: 4096 fresh p_0 samples; NFE grid GMM {5, 10, 20, 30, 50, 70, 90, 100} (Fig. 3 x-ticks)
  and checkerboard {5, 10, 50, 100, 1000} (Table 2); sampler seeds fixed per (NFE, repeat) and shared by all arms; several repeats per point
  (different sampler and reference seeds) so we can report the Monte Carlo spread.
- Fig. 3 right: t grid in [0.01, 0.99]; 4096 (x_0, x_t) pairs; 8 ξ draws; per-coordinate std (ddof 0) averaged over coordinates and x_t; the
  analytic posterior with the same 8-sample protocol, plus its exact std.

## Configs
`configs/paper/{gmm,cb}_{A,B,C}.json` = App. I.1 settings (checkerboard: borrowed, labelled). `configs/smoke/*.json` = the same model and
optimiser with few steps (300) and frequent checkpoints, for S2's checklist. A config is a flat JSON dict; the training CLI writes the resolved
config, command, git commit, seed, threads, device, peak RSS and wall time into the run directory.

## Compute
CPU only (`_task.md` §5.1 GPU blocker: the user's `gpu48` QOS has 2 GPUs, both held by other chains; `_rules.md`). Timing of one paper-config
training step per arm is in the section below and in `state/S1.json`; S2 picks the job layout.

## CPU timing (S1, node cs607, this allocation)
(filled in by the timing step)
