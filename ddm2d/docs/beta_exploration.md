# Does β (and the distance inside the energy score) change the results? (ddm2dx)

Exploration on top of the finished ddm2d replica of De Bortoli et al. 2025 (arXiv 2502.02483). Task and the user's words:
`/scratch/wg2381/claude_jobs/ddm2dx/prompts/_task.md`. Stage X3 wrote the 2D sections (E1, E2, E4). Stage X4 adds the CIFAR-10 section (E3)
and the overall summary. Labels: **[E]** evidence in a committed file, **[G]** guess or assumption.

## Answer across E1–E4 (X4)
**Yes, β and the distance change the results, but how much depends on λ, and λ matters more than β [E].**
- **λ = 1 (energy score, arm C), β inside (0, 2):** every β has the same optimum, and on the paper's two-mode GMM they are
  indistinguishable. On the 9-mode GMM with unequal weights, β = 1.5 beats the paper's β = 0.1 (2 seeds, disjoint ranges).
- **The edge breaks training:** β = 2 with λ = 1 is only proper. The spread is not learned, and it fails in 3 of 3 seeds; β = 1.99 is
  poor in 3 of 3.
- **λ = 0 (regression, arm B):** β chooses the point the deterministic denoiser sits at: near a mode, the geometric median, or the
  mean.
  - In 2D, β = 1 halves MMD² at few steps but skews the mode weights on the 9-mode GMM.
  - On CIFAR-10, β = 1 (and pseudo-Huber, which at iCT's c is the same thing) changes nothing at few steps. It is ≈ 15 % better in
    FID at 100 steps than β = 2, but with one training seed.
- **Other distances:** "only proper" choices fail: L1 with λ = 1 collapses the posterior onto curves. The strictly proper ones (L1^0.5,
  pseudo-Huber, IMQ, Laplace) behave like Euclidean on the paper GMM, and RBF with a median bandwidth is worse.
- **DINOv2 feature term (CIFAR-10):** it gives by far the best 2–4-step FID, but the images are visibly textured, and FID gets worse
  as steps are added.
- **λ is the bigger lever:** going from λ = 0 to λ = 1 (B → C) is worth about 100× in MMD² at few steps in 2D, and 2× in FID at 8–30
  steps on CIFAR-10. At 2–4 steps on CIFAR-10 it hurts, unlike in the paper (30k steps vs 1e6).

| change | paper GMM, MMD²_u at NFE 5 [seed range] (§2, §3, §5) | `gmm9w`, 9 unequal modes (§4, §5, §6) | CIFAR-10, FID-10k, 1 seed (§9) |
|---|---|---|---|
| λ = 0 → 1 (B → C, Euclidean) | 1.8e-2 → 1.7e-4 | NFE 5: 4.7e-2 → 7.8e-4 | NFE 10: 34.3 → 15.9; NFE 100: 14.0 → 10.5; NFE 2: 118 → 201 |
| λ = 1: β 0.1 → 1.5 | 1.7e-4 → 8.8e-5, inside noise | 7.8e-4 → 2.3e-4; mode TV 0.036 → 0.022 (seeds disjoint) | not run |
| λ = 1: β = 2 or 1.99 | fails in 3/3 seeds each (2 overflow at β = 2) | not run | not run |
| λ = 0: β 2 → 1 | 1.8e-2 → 8.6e-3 [8.5e-3, 8.7e-3]; ties at NFE 100 | NFE 5: 4.7e-2 → 4.0e-2; NFE 100: 8.9e-4 → 4.1e-3 (n = 1; heavy mode over-weighted) | NFE 10: 34.3 → 33.7; NFE 100: 14.0 → 11.8 |
| λ = 0: β 2 → 3, 3.5 | 1.8e-2 → 3.3e-2, 5.2e-2 (worse at every NFE) | not run | not run |
| λ = 0: pseudo-Huber | 8.8e-3 (= β 1) | 3.3e-2 (n = 1) | = β 1 within 0.7 FID |
| λ = 1: L1 (only proper) | 3.4e-3 (20× worse) | 1.6e-2 (n = 1) | not run |
| λ = 1: L1^0.5, pseudo-Huber, IMQ, Laplace | 1.3e-4 to 2.2e-4 (= Euclidean) | 1.3e-4 to 8.0e-4 (n = 1) | not run |
| λ = 1: RBF, median bandwidth | 6.3e-4 (4× worse) | 2.7e-3 (n = 1) | not run |
| λ = 1 + DINOv2 energy term | not run | not run | NFE 2: 201 → 49; NFE 100: 10.5 → 62; textured images |

Null floors: 2D in §1 (MMD²_u sd of a 10-repeat mean 4–7e-5); CIFAR-10 in §9.2. Next steps: §10.

## 2D answer (X3)
**Yes, β matters, but differently for the two families, and the edges matter most [E].**
- **λ = 0 (regression loss, arm B family).** β decides *where* the deterministic denoiser sits:
  - β ≤ 0.5: at the nearest mode;
  - β = 1: at the geometric median;
  - β = 2: at the posterior mean (= standard diffusion);
  - β > 2: past the mean, away from the dominant mode.

  On the paper GMM this makes few-step sampling up to 2× better: at NFE 5, β = 1 gives MMD² 8.6e-3 (seeds 8.5–8.7e-3) against
  1.8e-2 for β = 2. It costs at many steps for β = 0.1, and badly for β > 2 at every NFE. On the 9-mode GMM with unequal weights, a
  β = 1 denoiser over-weights the heaviest mode and under-weights the rare ones. It stays the deterministic denoiser it is in theory:
  ξ does nothing.
- **λ = 1 (energy score, arm C family).**
  - On the paper GMM every β in (0, 2) gives the same quality: MMD² 0.6–4e-4 at NFE 5 over 2–3 seeds, all inside the seed spread and
    near the null floor. That matches theory: they share the same minimiser, the exact posterior.
  - **On the 9-mode GMM β does matter:** β = 1.5 beats the paper's β = 0.1 (2 seeds each, seed ranges disjoint).
    - MMD² at NFE 5: 2.3e-4 vs 7.8e-4.
    - Final mode-weight TV: 0.022 vs 0.036 (null p95 0.024).
    - Conditional mode-weight TV at t = 0.99: 0.18 vs 0.30.

    The optimum is the same, so this is a finite-training effect of β [G].
  - The edge breaks things. **β = 2 with λ = 1 is only proper**, so the loss does not see the spread of x̂. It fails in 3 of 3 seeds:
    - seeds 0 and 1 were poor from the first EMA snapshot (MMD² 5e-3 to 1e-1 at NFE 5 from 10k steps), and late in training their
      spread ran away (posterior std 1e5–1e8) until the samples overflowed;
    - seed 2 stayed finite but learned the wrong spread (std 0.67 at t = 0.01 vs exact 0.01; MMD² 4.4e-2).
  - β = 1.99 is strictly proper but nearly degenerate, and poor in 3 of 3 seeds. One collapsed to a constant output. The others
    ended at MMD² 2.4e-3 and 9.3e-3 with the wrong spread.
- **Other distances.**
  - L1 at β = 1 is only proper (Lyons 2013). The loss then sees only the coordinate marginals, and C's posterior draws collapse onto
    1-D curves: 20× worse at NFE 5 (3.4e-3 vs 1.7e-4).
  - L1 at β = 0.5 is strictly proper and matches Euclidean.
  - Pseudo-Huber, IMQ and Laplace match Euclidean ‖·‖^0.1 at few steps on the paper GMM.
  - RBF is 4× worse, and IMQ and RBF are worse at 100 steps.
  - On `gmm9w` (one seed) L1^0.5, pseudo-Huber c = 0.1 and Laplace beat Euclidean β = 0.1 at NFE 5 and in the high-noise mode weights.
    Single-seed: suggestive, not established.

Sections below. Every number comes from `runs/ext/eval/` (raw JSON per run, summary CSVs, and `tables.md`, generated by
`scripts/ext_tables.py`). Figures are in `figures/ext/`.

## 1. Protocol (2D, identical for every run)
- Runs: 64 new 100k-step runs (`runs/ext/PLAN.md`, configs `configs/ext/`), plus the reference runs `runs/paper/gmm_{A,B,C}_s{0,1}`.
  All were re-evaluated with the same script. Completeness: `runs/ext/STATUS.md` (70/70 at 100k, final checkpoint loads, sha256
  matches).
- Script `scripts/ext_eval.py`, built on `scripts/eval_gmm.py` and `ddm2d.evaluate`. EMA weights of `final.pt`; Algorithm 2 sampler
  (churn 1, safety 1e-2).
  - **MMD²_u**, RBF σ = 1, 4096 vs 4096, at NFE {5, 10, 20, 30, 50, 100}, 10 repeats. Seeds as in ddm2d: sampler seed
    1e6 + 1000 r + NFE; reference set `eval_set(ds, 4096, k=r)`.
  - Check: re-evaluating `gmm_C_s0` reproduces `runs/paper/eval_gmm/mmd_gmm_C_s0.json` to 1e-16.
  - Cell format in the tables: seed mean [seed min, seed max] of the per-seed 10-repeat means.
- Exact posterior: `posterior_gmm.py` (per-component weights and stds; equals `posterior.py` on the paper GMM, tests in X1).
- **Null floors** (`runs/ext/eval/nulls.json`; p0 vs p0):

  | dataset | MMD²_u sd, one evaluation | sd of a 10-repeat mean | mode-weight TV, mean / p95 | rare-mode rel. error, p95 |
  |---|---|---|---|---|
  | gmm | 1.8e-4 | 5.8e-5 | 0.007 / 0.016 | 0.03 |
  | gmm2w | 2.2e-4 | 6.9e-5 | 0.004 / 0.011 | 0.05 |
  | gmm9w | 1.2e-4 | 3.8e-5 | 0.016 / 0.024 | 0.21 |

  So MMD² differences below ≈ 1.5e-4 between two-seed means are not resolved.
- Mode weights: each sample goes to the component with the largest π_k N(x; μ_k, s_k² I).
  - Rare modes: `gmm9w` k ∈ {1, 3, 5} (π = 0.02, 0.05, 0.03); `gmm2w` k = 1 (π = 0.2).
  - Off-mode: more than 4 s_k from every mean.
- Other parts:
  - posterior std vs t (4096 x_t × 8 draws);
  - Fig. 4-style draws (4096 per panel; 1024 saved);
  - denoiser location (256 x_t, x̂ = mean of 8 ξ draws; geometric median by Weiszfeld on 4096 exact draws);
  - conditional mode weights (256 x_t × 4096 draws at t ∈ {0.99, 0.88, 0.7, 0.45}, null = 4096 exact draws per x_t through the same
    assignment).
- Seeds: 2 per setting (E4 on `gmm9w`: 1). Differences inside the seed range are not called real.

## 2. E1a: β sweep with λ = 0 (B family, paper GMM)
Figures:
- `figures/ext/mmd_vs_beta.png`: blue line;
- `mmd_vs_nfe_beta.png`: left panel;
- `denoiser_location.png`, `fig4_gmm_lambda0.png`;
- `posterior_std_beta.png`: left panel.

| β (λ = 0) | NFE 5 | NFE 10 | NFE 20 | NFE 50 | NFE 100 |
|---|---|---|---|---|---|
| A (reference) | 1.7e-2 [1.5e-2, 1.8e-2] | 4.6e-3 [4.1e-3, 5.0e-3] | 1.8e-3 [1.5e-3, 2.1e-3] | 4.7e-4 [2.4e-4, 7.1e-4] | 5.6e-4 [5.0e-4, 6.3e-4] |
| 0.1 | 1.1e-2 [9.4e-3, 1.2e-2] | 6.7e-3 [5.6e-3, 7.8e-3] | 4.0e-3 [3.1e-3, 4.9e-3] | 2.6e-3 [1.5e-3, 3.8e-3] | 2.7e-3 [1.1e-3, 4.2e-3] |
| 0.5 | 1.0e-2 [9.6e-3, 1.0e-2] | 4.9e-3 [4.7e-3, 5.1e-3] | 2.2e-3 [2.0e-3, 2.4e-3] | 9.8e-4 [6.8e-4, 1.3e-3] | 1.1e-3 [6.7e-4, 1.4e-3] |
| **1** | **8.6e-3 [8.5e-3, 8.7e-3]** | **4.0e-3 [3.8e-3, 4.1e-3]** | 1.6e-3 [1.5e-3, 1.7e-3] | 6.4e-4 [4.1e-4, 8.7e-4] | 6.4e-4 [4.7e-4, 8.0e-4] |
| 1.5 | 1.1e-2 [1.1e-2, 1.2e-2] | 4.3e-3 [4.1e-3, 4.5e-3] | 1.7e-3 [1.5e-3, 1.8e-3] | 6.1e-4 [4.8e-4, 7.4e-4] | 5.0e-4 [3.0e-4, 7.0e-4] |
| 1.99 | 1.9e-2 [1.8e-2, 1.9e-2] | 5.3e-3 [5.3e-3, 5.4e-3] | 1.7e-3 [1.6e-3, 1.8e-3] | 8.2e-4 [5.2e-4, 1.1e-3] | 1.1e-3 [7.6e-4, 1.5e-3] |
| 2 (= B) | 1.8e-2 [1.7e-2, 1.8e-2] | 5.6e-3 [5.1e-3, 6.1e-3] | 2.0e-3 [1.7e-3, 2.3e-3] | 9.5e-4 [5.8e-4, 1.3e-3] | 1.0e-3 [8.2e-4, 1.3e-3] |
| 3 | 3.3e-2 [2.7e-2, 3.9e-2] | 9.6e-3 [9.0e-3, 1.0e-2] | 5.1e-3 [4.7e-3, 5.5e-3] | 5.0e-3 [5.0e-3, 5.1e-3] | 7.6e-3 [6.1e-3, 9.0e-3] |
| 3.5 | 5.2e-2 [4.9e-2, 5.5e-2] | 2.0e-2 [2.0e-2, 2.1e-2] | 1.5e-2 [1.5e-2, 1.6e-2] | 1.5e-2 [1.4e-2, 1.6e-2] | 1.6e-2 [1.4e-2, 1.7e-2] |

**Does β matter for λ = 0? Yes [E].**
- **Few steps (NFE 5): β < 2 helps.** β ∈ {0.1, 0.5, 1, 1.5} gives 8.6e-3 to 1.1e-2, against 1.7–1.8e-2 for β ∈ {1.99, 2} and A.
  The seed ranges do not overlap. β = 1 is best, ≈ 2× below β = 2.
  - At NFE 10 only β = 1 is below A's seed range: 4.0e-3 vs 4.1–5.0e-3.
- **Many steps (NFE 50–100): β ∈ [1, 2] ties with A** (5–11e-4, inside the noise).
  - β = 0.1 is worse (2.7e-3, wide seed range), and β = 0.5 is borderline.
- **β > 2 hurts at every NFE:** 2–5× at NFE 5 and 8–16× at NFE 100.

**Where the B denoiser sits** (`tables.md`, "E1a: where the λ = 0 denoiser sits"; 256 x_t; seed means):

| β | t = 0.88: closest of mean / geo. median / coord. median / nearest mode | d(x̂, mean) | d(x̂, geo. median) | d(x̂, nearest mode) |
|---|---|---|---|---|
| 0.1 | 0.01 / 0.01 / 0.04 / **0.95** | 1.96 | 0.59 | 0.14 |
| 0.5 | 0.00 / 0.00 / 0.05 / **0.95** | 1.89 | 0.57 | 0.15 |
| 1 | 0.01 / **0.71** / 0.26 / 0.03 | 1.46 | 0.10 | 0.56 |
| 1.5 | 0.50 / 0.49 / 0.01 / 0.00 | 0.62 | 0.82 | 1.39 |
| 2 | **0.99** / 0.01 / 0.00 / 0.00 | 0.12 | 1.36 | 1.93 |
| 3.5 | **1.00** / 0.00 / 0.00 / 0.00 | 0.52 | 1.95 | 2.52 |
| A | **1.00** / 0.00 / 0.00 / 0.00 | 0.09 | 1.44 | 2.01 |

Exact mean and geometric median are 1.43 apart at t = 0.88. At t = 0.45 the posterior is unimodal and all of them coincide (every
d < 0.05, except β = 3.5 at 0.24).

- **Reading.**
  - β ≤ 0.5 puts x̂ on a mode.
  - β = 1 puts it on the geometric median.
  - β = 1.5 is halfway between the median and the mean.
  - β ≥ 1.99 puts it on the posterior mean.
  - β = 3–3.5 overshoots the mean, away from the dominant mode, towards the middle of the range. (The L^p centre moves towards the
    midrange as p grows [G].)
- **Why few steps help.** At few steps the posterior-mean sampler puts mass between the two modes (ddm2d `results_gmm.md` §3.3: A/B
  are at the exact-mean oracle). A median- or mode-seeking denoiser does not.
- **Why β = 0.1 costs at many steps.** A mode-seeking x̂ at high t snaps to one mode with probability ≠ r_k(x_t) [G]. Its spread over ξ
  stays ≤ 0.02 at every t (`posterior_std` table), so B with β < 2 is still deterministic in practice, as theory says.

**Stability [E]** (`runs/ext/eval/gnorm_by_beta.txt`, training logs). The `_task.md` worry was that β < 1 would be unstable
(gradient ‖d‖^(β−1) blows up at d → 0). **It did not happen.** The fewest gradient spikes are at β ≤ 0.5:
- β = 0.1: 0 windows with max gradient norm > 100, max 7;
- β = 0.5: 0–1 windows, max ≤ 169.

Spikes grow with β:
- β = 1–2: 4–17 windows, max up to 1e8 (β = 1.99) and 1.7e10 (the paper's β = 2 run);
- β = 3–3.5: 12–24 windows, max 1e14–1e16.

Clipping at 1 bounds each update. The large-d tail ‖d‖^(β−1) for β > 1 drives the spikes, not the small-d singularity [G].

## 3. E1b: β sweep with λ = 1 (C family, paper GMM)
Figures: `figures/ext/mmd_vs_beta.png` (orange line), `mmd_vs_nfe_beta.png` (right), `posterior_std_beta.png` (right),
`fig4_gmm_lambda1.png`, `condw_nondominant.png` (left).

| β (λ = 1) | NFE 5 | NFE 10 | NFE 20 | NFE 50 | NFE 100 |
|---|---|---|---|---|---|
| 0.1 (= C) | 1.7e-4 [1.4e-4, 1.9e-4] | 1.7e-4 [5.3e-5, 2.9e-4] | 1.9e-4 [1.0e-4, 2.8e-4] | 4.4e-4 [4.1e-4, 4.8e-4] | 5.2e-4 [4.8e-4, 5.7e-4] |
| 0.5 | 1.5e-4 [1.3e-4, 1.7e-4] | 1.2e-4 [1.2e-4, 1.2e-4] | 2.8e-5 [−2.9e-5, 8.6e-5] | 2.6e-4 [2.2e-4, 3.0e-4] | 4.4e-4 [3.5e-4, 5.4e-4] |
| 1 | 2.7e-4 [1.3e-4, 4.1e-4] | 2.2e-4 [1.9e-4, 2.4e-4] | 2.5e-4 [6.9e-5, 4.3e-4] | 3.3e-4 [1.8e-4, 4.9e-4] | 5.3e-4 [2.4e-4, 8.1e-4] |
| 1.5 | 8.8e-5 [6.4e-5, 1.1e-4] | 1.3e-4 [4.8e-5, 2.1e-4] | 1.2e-4 [8.9e-5, 1.4e-4] | 2.8e-4 [2.0e-4, 3.5e-4] | 5.7e-4 [4.6e-4, 6.9e-4] |
| 1.99 | 6.5e-1 [2.4e-3, 1.3] | 6.5e-1 [2.6e-3, 1.3] | 6.5e-1 [6.1e-3, 1.3] | 6.5e-1 [9.4e-3, 1.3] | 6.5e-1 [1.1e-2, 1.3] |
| 2 | 0.33 (both seeds) | NaN (overflow) | NaN | NaN | NaN |

**Does β matter for λ = 1 within (0, 2)? Not measurably at few or many steps [E].**
- At NFE 5–20 the four strictly proper settings span 0.3–2.7e-4. That is within ≈ 2 null standard deviations of a two-seed mean.
- β = 1.5 has the lowest NFE-5 value. Its two seeds (6.4e-5, 1.1e-4) lie below the two seeds of β = 0.1 (1.4e-4, 1.9e-4), but the gap
  (≈ 8e-5) is at the noise floor (sd of a 10-repeat mean 5.8e-5). Not called real. **The third seed confirms this:** β = 1.5 seed 2
  gives 1.7e-4, and β = 0.1 seed 2 gives 2.2e-4. Over three seeds the ranges overlap (6.4e-5–1.7e-4 vs 1.4–2.2e-4; §6).
- Posterior std vs t is the same for all β ∈ {0.1, 0.5, 1, 1.5} at t ≥ 0.23 (e.g. 0.38–0.39 at t = 0.45, exact 0.43). All are
  under-dispersed at t ≥ 0.66 (1.39–1.46 vs 1.65 at t = 0.88). All are over-dispersed at t = 0.01 (0.04–0.11 vs 0.01).
- **High-noise mode weights depend on β [E]** (`tables.md`, conditional mode weights). At t = 0.99 the exact posterior puts 0.49 of its
  mass outside the dominant mode. The model puts:
  - β = 0.1: 0.32 (mean TV 0.17);
  - β = 0.5: 0.34 (TV 0.15);
  - β = 1.5: 0.37 (TV 0.12);
  - β = 1: 0.40 (TV 0.10).

  Null TV is 0.006. Larger β reduces the known under-weighting of the far mode by about 40 %, but none removes it. By t = 0.7 every β
  is right (TV ≈ 0.01). Because Algorithm 2 re-noises, this high-t error barely shows in the final MMD².

**Is β = 2 with λ = 1 visibly different? Yes, it fails [E].**
- **Both seeds fail.** The EMA snapshots (diagnostic, §6) are already poor at 10k steps: MMD² 5e-3 to 1e-1 at NFE 5, against ≈ 1e-4
  for every β < 2. Late in training the interaction term then grows without bound: inter ≈ 1e3 at 91k steps and 1e16 at 100k (seed 0);
  1e14 at 100k (seed 1) (`runs/ext/gmm_C_b2_s*/log.jsonl`).
- The posterior std of x̂ at 100k is 1e5–1e8, and sampling overflows.
- Mechanism: at β = 2, E‖X − y‖² − ½E‖X − X'‖² = ‖E X − y‖² + const. The variance terms cancel exactly, so only the mean is learned and
  the spread is a free direction that drifts. That is "proper but not strictly proper", seen directly.
- **β = 1.99 is also unsafe [E]:**
  - Seed 1: the spread blew up at ≈ 73k steps (inter 1e7), then the network fell into a constant output (posterior std 0, every sample
    at the same point, MMD² 1.29).
  - Seed 0: finished, but under-dispersed at high t (std 0.34–0.49 at t ≥ 0.66 vs 0.81–1.77 exact) and over-dispersed at low t (0.26 at t = 0.01 vs 0.01); MMD² 2.4e-3 at NFE 5.
  - The restoring force on the spread scales like (2 − β) [G]. At 1.99 it is too weak to hold against gradient noise and the
    clipped spikes.
  - EMA-snapshot diagnostic (10k–100k steps): `runs/ext/eval/snaps/`, §6. Seed 0 was at 1–4e-3 from 10k steps on, so β = 1.99 never
    trained well.

## 4. E2: unequal mode weights (`gmm2w` 0.8 / 0.2; `gmm9w` 9 modes, π from 0.02 to 0.30)
Figures: `figures/ext/mode_weights_gmm2w.png`, `mode_weights_gmm9w.png`, `fig4_gmm2w_x{0,1}.png`, `fig4_gmm9w_x{0,1}.png`,
`condw_nondominant.png`, `posterior_std_e2_e4.png`.

| run | NFE 5 | NFE 10 | NFE 20 | NFE 50 | NFE 100 |
|---|---|---|---|---|---|
| gmm2w A | 3.5e-2 [3.2e-2, 3.8e-2] | 1.4e-2 [1.3e-2, 1.6e-2] | 5.8e-3 [5.2e-3, 6.5e-3] | 2.3e-3 [1.8e-3, 2.7e-3] | 1.3e-3 [9.4e-4, 1.7e-3] |
| gmm2w B | 2.8e-2 [2.7e-2, 2.9e-2] | 1.0e-2 [9.8e-3, 1.1e-2] | 2.7e-3 [2.5e-3, 3.0e-3] | 8.0e-4 [5.8e-4, 1.0e-3] | 1.1e-3 [1.1e-3, 1.1e-3] |
| gmm2w C | 2.5e-4 [1.4e-4, 3.6e-4] | 1.7e-4 [1.6e-4, 1.9e-4] | 2.8e-4 [1.8e-4, 3.9e-4] | 4.2e-4 [2.9e-4, 5.4e-4] | 8.6e-4 [6.6e-4, 1.1e-3] |
| gmm9w A | 5.5e-2 [5.0e-2, 6.0e-2] | 1.8e-2 [1.5e-2, 2.0e-2] | 4.7e-3 [4.5e-3, 4.9e-3] | 2.0e-3 [1.6e-3, 2.3e-3] | 1.9e-3 [1.2e-3, 2.5e-3] |
| gmm9w B | 4.7e-2 [4.6e-2, 4.7e-2] | 1.1e-2 [9.8e-3, 1.1e-2] | 2.1e-3 [2.1e-3, 2.1e-3] | 6.8e-4 [5.1e-4, 8.5e-4] | 8.9e-4 [5.3e-4, 1.2e-3] |
| gmm9w C | 7.8e-4 [5.4e-4, 1.0e-3] | 4.3e-4 [3.7e-4, 4.8e-4] | 5.7e-4 [4.3e-4, 7.2e-4] | 6.9e-4 [4.5e-4, 9.2e-4] | 9.6e-4 [6.3e-4, 1.3e-3] |
| gmm9w B β = 1 (extra, n = 1) | 4.0e-2 | 1.1e-2 | 3.6e-3 | 3.9e-3 | 4.1e-3 |
| gmm9w C β = 1.5 (extra) | 2.3e-4 [2.3e-4, 2.3e-4] | 1.7e-4 [1.4e-4, 2.0e-4] | 4.4e-4 [3.1e-4, 5.6e-4] | 5.9e-4 [5.8e-4, 6.0e-4] | 8.2e-4 [6.9e-4, 9.5e-4] |

Final-sample mode weights (TV = ½Σ|p̂_k − π_k|; null TV p95: gmm2w 0.011, gmm9w 0.024):

| run | TV NFE 5 | TV NFE 10 | TV NFE 100 | rare-mode max rel. error NFE 10 / 100 (null p95: 0.05 gmm2w, 0.21 gmm9w) |
|---|---|---|---|---|
| gmm2w A | 0.093 [0.086, 0.100] | 0.062 [0.056, 0.068] | 0.022 [0.016, 0.027] | 0.31 / 0.11 |
| gmm2w B | 0.071 [0.066, 0.076] | 0.039 [0.039, 0.039] | 0.004 [0.003, 0.005] | 0.20 / 0.02 |
| gmm2w C | 0.008 [0.006, 0.010] | 0.008 [0.007, 0.010] | 0.009 [0.004, 0.014] | 0.04 / 0.05 |
| gmm9w A | 0.284 [0.274, 0.294] | 0.161 [0.147, 0.176] | 0.045 [0.039, 0.052] | 0.29 / 0.23 |
| gmm9w B | 0.272 [0.267, 0.277] | 0.127 [0.120, 0.134] | 0.019 [0.018, 0.019] | 0.54 / 0.19 |
| gmm9w C | 0.036 [0.028, 0.043] | 0.030 [0.030, 0.030] | 0.033 [0.028, 0.037] | 0.17 / 0.15 |
| gmm9w B β = 1 (n = 1) | 0.189 | 0.107 | 0.063 | 0.48 / 0.57 |
| gmm9w C β = 1.5 (extra) | 0.022 [0.021, 0.023] | 0.022 [0.021, 0.022] | 0.023 [0.023, 0.024] | 0.17 / 0.15 |

**Does each arm get the mode weights right?**
- **Final samples, few steps:**
  - **A and B are wrong.**
    - On `gmm2w` the minor mode (π = 0.2) gets 0.107 (A) and 0.129 (B) at NFE 5.
    - On `gmm9w` the centre mode k = 4 (π = 0.20) gets 2.3× its weight at NFE 5 and 1.5–1.7× at NFE 10. The corners
      k ∈ {0, 2, 6, 8} lose 20–75 % (`mode_weights_gmm9w.png`).
    - Mechanism: the posterior-mean denoiser pulls mass towards the centre of the grid [G].
  - **C is right on `gmm2w`** (0.202 at NFE 5; TV 0.008, at the null).
  - **C is close on `gmm9w`** (TV 0.03–0.04, slightly above the null p95 0.024, at every NFE).
- **Final samples, NFE 100:**
  - B is right (TV 0.004 on gmm2w; 0.019 on gmm9w, inside the null).
  - A is still off (0.022, 0.045).
  - C keeps a small constant bias on `gmm9w` (TV 0.033; k = 4 at 0.188 vs 0.20, k = 6 at 0.113 vs 0.10).
  - So with many steps, B (β = 2) has the best mode weights. This matches the paper's "many steps favour λ ≈ 0, β ≈ 2"
    (`main.tex` ≈ line 1820).
- **B (β = 2) beats A at few steps on both new datasets** (seed ranges disjoint at NFE 5–20). On the paper GMM they tie. A and B have
  the same minimiser, so this is a training or architecture effect, not a loss effect [G].
- **Conditional mode weights at high t** (`tables.md`; mean TV between model draws and exact responsibilities over 256 x_t;
  null 0.005–0.016):

  | run | t = 0.99 | t = 0.88 | t = 0.7 | t = 0.45 |
  |---|---|---|---|---|
  | gmm C (β = 0.1) | 0.169 | 0.032 | 0.009 | 0.001 |
  | gmm2w C | 0.122 | 0.031 | 0.009 | 0.000 |
  | gmm9w C | 0.297 | 0.132 | 0.035 | 0.011 |
  | gmm9w A / B (deterministic) | 0.800 / 0.800 | 0.764 / 0.732 | 0.347 / 0.345 | 0.030 / 0.030 |

  - C's high-noise mode-weight error **does get worse with more, unequal modes**: TV 0.30 on `gmm9w` against 0.17 on the paper GMM.
  - On `gmm9w` the *total* mass outside the dominant mode is right (0.69 vs 0.70). The error is in how that mass is spread over the
    other modes.
  - On `gmm2w` C slightly *over*-weights the minor mode at t = 0.99 (0.23 vs 0.20).
- **Which β / ρ does best on unequal weights?**
  - Final-sample weights: B β = 2 at many steps, C at few steps.
  - Among C variants on `gmm9w` (seed 0), the conditional high-t error is lowest for pseudo-Huber c = 0.1 (0.162) and c = 1
    (0.166), then Laplace (0.209), L1^0.5 (0.215) and IMQ (0.218). Euclidean β = 0.1 has 0.297 over 2 seeds, and seed 0 alone
    is 0.30.
  - The final TV of those variants is 0.021–0.028 vs 0.030–0.036.
  - **The E2 extra C β = 1.5 on `gmm9w` (2 seeds) is the best C setting measured with 2 seeds** (§6): final TV 0.022 [0.021, 0.023]
    vs 0.036 [0.028, 0.043] for β = 0.1, and conditional TV at t = 0.99 0.181 vs 0.297. Its per-mode fractions at NFE 100 are all
    within 0.013 of π_k.
  - **λ = 0 with β = 1 on `gmm9w` is worse.** It over-weights the heaviest mode (k = 0: 0.35 vs 0.30 at NFE 100) and halves the rarest
    (k = 1: 0.009 vs 0.020). This is the mode-seeking bias of §2 hurting, as `_task.md` anticipated.

## 5. E4: distances other than Euclidean ‖·‖^β
Figures: `figures/ext/e4_table.png`, `fig4_gmm_e4.png`, `fig4_gmm9w_x{0,1}.png`, `mode_weights_gmm9w_e4.png`, `posterior_std_e2_e4.png`.
Kernel forms were checked against `main.tex` by X1 (standup X1). Paper GMM: seeds 0, 1. `gmm9w`: seed 0 only.

| run | gmm NFE 5 | gmm NFE 10 | gmm NFE 100 | gmm9w NFE 5 | gmm9w NFE 10 | gmm9w NFE 100 | verdict vs Euclidean (same λ) |
|---|---|---|---|---|---|---|---|
| A | 1.7e-2 | 4.6e-3 | 5.6e-4 | 5.5e-2 | 1.8e-2 | 1.9e-3 | reference |
| B (‖·‖², λ = 0) | 1.8e-2 | 5.6e-3 | 1.0e-3 | 4.7e-2 | 1.1e-2 | 8.9e-4 | reference |
| B_L1b1 (L1 regression) | 8.7e-3 [8.6, 8.9e-3] | 4.2e-3 | 1.0e-3 | 2.5e-2 | 1.2e-2 | 9.3e-3 | better at NFE 5 (both datasets, seeds agree on gmm); worse at NFE 100 on gmm9w (mode-seeking bias) |
| B_pH0.1 (λ = 0) | 8.8e-3 [8.7, 8.9e-3] | 4.2e-3 | 9.2e-4 | 3.3e-2 | 8.8e-3 | 5.2e-3 | same as B with β = 1: better at NFE 5, worse at NFE 100 on gmm9w |
| C (‖·‖^0.1, λ = 1) | 1.7e-4 [1.4, 1.9e-4] | 1.7e-4 | 5.2e-4 | 7.8e-4 [5.4e-4, 1.0e-3] | 4.3e-4 | 9.6e-4 | reference |
| C_L1b1 | 3.4e-3 [3.3, 3.6e-3] | 1.5e-3 | 2.9e-4 [2.4, 3.4e-4] | 1.6e-2 | 6.2e-3 | 6.6e-4 | **much worse at few steps** (20×, seeds agree); draws collapse onto curves |
| C_L1b0.5 | 1.5e-4 [1.1, 1.8e-4] | 1.4e-4 | 4.5e-4 | 1.3e-4 | 1.3e-4 | 5.6e-4 | gmm: no different. gmm9w: lower than both C seeds at NFE 5 (n = 1) |
| C_pH0.1 | 2.2e-4 [2.0, 2.4e-4] | 1.9e-4 | 5.1e-4 | 2.5e-4 | 2.2e-4 | 8.4e-4 | gmm: no different. gmm9w: lower at NFE 5 (n = 1) |
| C_pH1 | 1.5e-4 [7.3e-5, 2.2e-4] | 9.3e-5 | 7.9e-4 | 4.1e-4 | 3.3e-4 | 1.0e-3 | few steps: no different (gmm9w NFE 5 lower, n = 1); gmm NFE 100: slightly worse (seeds 6.5–9.2e-4 vs 4.8–5.7e-4) |
| C_imq (c = 1) | 1.5e-4 [4.6e-5, 2.5e-4] | 1.0e-4 | **2.7e-3** [4.7e-4, 4.9e-3] | 8.0e-4 | 5.8e-4 | 8.1e-4 | few steps: no different; NFE 100 on gmm: worse (one seed 4.9e-3); std at t = 0.23–0.45 too high (0.42 / 0.64 vs 0.25 / 0.43) |
| C_rbf (σ² = median) | **6.3e-4** [5.6, 7.0e-4] | 7.7e-4 | **3.1e-3** | 2.7e-3 | 1.7e-3 | 3.9e-3 | **worse** at every NFE on both datasets |
| C_lap (σ = median) | 1.3e-4 [5.5e-5, 2.0e-4] | 1.4e-4 | 7.7e-4 | 2.7e-4 | 1.3e-4 | 5.5e-4 | gmm: no different. gmm9w: lower at NFE 5 (n = 1) |

Seed ranges are shown where they matter. All cells are in `runs/ext/eval/tables.md`.

**Does any ρ beat Euclidean ‖·‖^0.1?**
- **Paper GMM, few steps:** no. Pseudo-Huber, IMQ, Laplace and L1^0.5 tie with it (1.3–2.2e-4 vs 1.7e-4). L1^1 and RBF are worse.
- **Paper GMM, many steps (NFE 100):** no ρ is better beyond noise. Pseudo-Huber c = 1 is slightly worse (7.9e-4, seeds disjoint from C). L1^1 at 2.9e-4 lies below C's seed range (4.8–5.7e-4) with
  non-overlapping seeds, but it is 20× worse at NFE 5. IMQ and RBF are worse.
- **`gmm9w`:**
  - At NFE 5 three variants have a lower single-seed value than both Euclidean seeds: L1^0.5 (1.3e-4), pseudo-Huber c = 0.1 (2.5e-4)
    and Laplace (2.7e-4), against Euclidean 5.4e-4 and 1.0e-3.
  - Their high-t conditional mode weights are also better (§4).
  - With one seed this is suggestive, not established. A second seed of `gmm9w_C_pH0.1` and `gmm9w_C_L1b0.5` is the cheapest next test.
- **λ = 0:** L1 regression and pseudo-Huber regression behave like Euclidean β = 1 (median-type denoiser). They are 2× better than
  β = 2 at NFE 5 on the paper GMM, but bias the weights towards heavy modes on `gmm9w`.
- **The L1^1 failure is the clearest E4 result** (`fig4_gmm_e4.png`, `fig4_gmm9w_x1.png`). C_L1b1's posterior draws lie on thin
  monotone curves instead of round clusters, on both datasets, at every t down to 0.23.
  - With ρ = ‖·‖₁ the energy score splits into a sum of 1-D energy scores, one per coordinate. So it is minimised by *any* coupling of
    the right marginals; the comonotone one is a 1-D curve.
  - The exact posterior is among the minimisers (it is even a product on the paper GMM), but the trained network found another one.
  - With β = 0.5 the score is strictly proper again (Lyons 2013, Remark 3.19), and the clusters are round.

## 6. Extra runs and diagnostics
**EMA-snapshot diagnostic [E]** (`runs/ext/eval/snaps/`, table at the end of `tables.md`). Not the protocol: 2 repeats; only used to
understand the failures, never to pick a checkpoint. MMD² at NFE 5 of the EMA weights every 10k steps, λ = 1 family:

| run | 10k–70k | 80k–100k |
|---|---|---|
| β ∈ {0.1, 0.5, 1, 1.5}, all seeds | −2e-4 … 1e-3 (noise around ≈ 1e-4) | same |
| β = 1.99, seed 0 | 1e-3 … 4e-3 | 1e-3 … 2e-3 |
| β = 1.99, seed 1 | 2e-3 … 5e-3 | 1.3 (collapsed to a constant) |
| β = 2, seeds 0, 1 | 5e-3 … 1e-1 | 5e-2 … 3e-1, NaN at NFE 100 at 100k |

So β ≥ 1.99 with λ = 1 never learned the posterior well, at any point of training. The late blow-up adds to that. All β < 2 are flat
from 10k steps on: no trend with training length, as for the ddm2d paper runs.

**Seed-2 reference runs** `runs/paper/gmm_{A,B,C}_s2`, same protocol (`tables.md`, "Seed 2"). At NFE 5: A 1.6e-2, B 1.6e-2, C 2.2e-4.
These are inside or next to the seed 0/1 ranges. They are used only for the seed-2 extras below, and the main tables stay on seeds
0, 1.

**Extra runs [E]** (`runs/ext/runs_x3.tsv`, `runs/ext/runlists/x3_list12.txt`, job 24786603: 4 h 46 min × 6 CPUs = 28.6 CPU-h; all 6 at
100k, `STATUS.md`). Same protocol.
- `gmm9w_C_b1.5_s{0,1}` is the E2 extra "C on `gmm9w` with the best E1b β". β = 1.5 had the lowest seed-mean MMD² at NFE 5 in E1b, though
  inside the noise.
- Seed 2 of `gmm_C_b1.99`, `gmm_C_b2`, `gmm_C_b1.5` and `gmm_B_b1`: the surprising or best-looking E1 settings.

Seed 2 next to seeds 0 / 1 (MMD²_u; `tables.md`, "Seed 2"):

| setting | NFE 5: s0 / s1 / s2 | NFE 10 | NFE 100 | reading |
|---|---|---|---|---|
| A (reference) | 1.8e-2 / 1.5e-2 / 1.6e-2 | 5.0e-3 / 4.1e-3 / 5.1e-3 | 6.3e-4 / 5.0e-4 / 7.2e-4 | – |
| B β = 2 (reference) | 1.8e-2 / 1.7e-2 / 1.6e-2 | 5.1e-3 / 6.1e-3 / 5.5e-3 | 8.2e-4 / 1.3e-3 / 8.5e-4 | – |
| **B β = 1** | 8.7e-3 / 8.5e-3 / **9.7e-3** | 4.1e-3 / 3.8e-3 / 4.9e-3 | 8.0e-4 / 4.7e-4 / 1.3e-3 | few-step gain confirmed: all 3 seeds below all A and B seeds at NFE 5 |
| C β = 0.1 (reference) | 1.4e-4 / 1.9e-4 / 2.2e-4 | 5.3e-5 / 2.9e-4 / 5.9e-4 | 5.7e-4 / 4.8e-4 / 3.3e-4 | – |
| C β = 1.5 | 6.4e-5 / 1.1e-4 / **1.7e-4** | 4.8e-5 / 2.1e-4 / 6.5e-5 | 4.6e-4 / 6.9e-4 / 3.8e-4 | ranges now overlap with β = 0.1: no real difference on the paper GMM |
| C β = 1.99 | 2.4e-3 / 1.3 / **9.3e-3** | 2.6e-3 / 1.3 / 9.4e-3 | 1.1e-2 / 1.3 / 1.9e-2 | poor in 3/3 seeds |
| C β = 2 | 0.33 / 0.33 / **4.4e-2** | 0.33 / NaN / 7.5e-2 | NaN / NaN / 0.11 | fails in 3/3 seeds. Seed 2 has no overflow (interaction term ≤ 46 over training), but its spread is wrong at every t (std 0.67 at t = 0.01, exact 0.01; 0.28 at t = 0.99, exact 1.77). Snapshots 2e-2 to 7e-2 at NFE 5 from 10k steps |

`gmm9w`, C with β = 1.5 vs the paper's β = 0.1 (2 seeds each; seed mean [min, max]):

| metric | C β = 0.1 | C β = 1.5 | null |
|---|---|---|---|
| MMD² NFE 5 | 7.8e-4 [5.4e-4, 1.0e-3] | **2.3e-4 [2.3e-4, 2.3e-4]** | sd of a 10-repeat mean 3.8e-5 |
| MMD² NFE 10 | 4.3e-4 [3.7e-4, 4.8e-4] | **1.7e-4 [1.4e-4, 2.0e-4]** | |
| MMD² NFE 100 | 9.6e-4 [6.3e-4, 1.3e-3] | 8.2e-4 [6.9e-4, 9.5e-4] | |
| final mode TV, NFE 5 / 10 / 100 | 0.036 / 0.030 / 0.033 | **0.022 / 0.022 / 0.023** | 0.016 mean, 0.024 p95 |
| conditional mode TV, t = 0.99 / 0.88 / 0.7 | 0.297 / 0.132 / 0.035 | **0.181 / 0.081** / 0.039 | 0.016 / 0.015 / 0.009 |
| posterior std at t = 0.88 / 0.99 (exact 3.00 / 3.31) | 2.38 / 2.39 | 2.67 / 2.87 | |

On `gmm9w`, β = 1.5 is better at few steps and in the mode weights, with seed ranges disjoint. At NFE 100 it ties. Its single-seed
competitors among the ρ variants (pseudo-Huber c = 0.1: conditional TV 0.162 at t = 0.99, NFE-5 MMD² 2.5e-4; L1^0.5: 1.3e-4) are in
the same range. So on this dataset the paper's β = 0.1 is the outlier, not the other distances [G].

**EMA-snapshot diagnostic of the seed-2 runs:** C β = 1.5 seed 2 sits at the noise level from 10k steps on. β = 1.99 seed 2 stays at
1e-3 to 9e-3, and β = 2 seed 2 at 2e-2 to 7e-2 at NFE 5 throughout.

## 7. Theory (`_task.md` "What theory says about β") vs what happened
| prediction | held? | evidence |
|---|---|---|
| λ = 1, β ∈ (0, 2): strictly proper, same minimiser (Gneiting & Raftery 2007; Lyons 2013 Remark 3.19 for ‖·‖^β) → similar results | **on the paper GMM, yes** (NFE 5–100, 3 seeds for β = 0.1 and 1.5). **On `gmm9w`, no:** β = 1.5 is 3× better than β = 0.1 at NFE 5 and has better mode weights (seeds disjoint). The same optimum does not mean the same finite-training result. β also changes the high-t conditional mode weights on the paper GMM (TV 0.10–0.17) | §3, §6 |
| λ = 1, β = 2: only proper, any distribution with the right mean is optimal | **yes, dramatically**: the spread is not learned. 3/3 seeds fail (2 overflow, 1 with the wrong spread at every t). β = 1.99 is also poor in 3/3 | §3, §6, logs |
| λ = 0: β = 2 → posterior mean, β = 1 → geometric median, β → 0 → near the heaviest mode | **yes** (closest-point shares 0.99 mean / 0.71 median / 0.95 mode at t = 0.88). "Heaviest" is in practice "nearest" or most responsible: with two modes they coincide | §2 |
| λ = 0, β < 2: mode-seeking may help at few steps, hurt mode weights | **both**: 2× better MMD² at NFE 5 (paper GMM), wrong heavy/rare mode weights on `gmm9w` at NFE 100 | §2, §4 |
| β < 1 unstable (gradient ‖d‖^(β−1) at d → 0) | **no**: β ≤ 0.5 had the fewest gradient spikes; spikes grow with β | §2 |
| λ = 0, β > 2: pulled towards the tails, strictly convex | x̂ moves past the mean, away from the dominant mode; MMD² 2–16× worse at every NFE | §2 |
| L1^1 only proper; L1^β, β < 1 strictly proper (Lyons 2013) | **yes**: L1^1 collapses onto curves (20× worse), L1^0.5 matches Euclidean | §5 |
| pseudo-Huber proper (multiquadric, Micchelli 1986); strictness unchecked in `_task.md` | **strictly proper, by our derivation [G, checked numerically]**: √(s + c²) − c = ∫(1 − e^(−su)) μ(du) with μ(du) = e^(−c²u) u^(−3/2)/(2√π) du, so D_ρ = ∫ MMD²_{exp(−u‖·‖²)} μ(du) > 0 for P ≠ Q (each Gaussian kernel is characteristic). Check: `scripts/ext_check_pseudohuber.py`, max error 4e-12. Empirically it behaves like Euclidean | §5 |
| kernel scores (IMQ, RBF, Laplace) strictly proper, "similar behaviour to the energy loss" (paper) | IMQ and Laplace: similar at few steps. RBF with the median bandwidth: worse; on the paper GMM the median falls between the within- and cross-mode scales (X1 caveat) [G] | §5 |
| paper, images: λ = 0, β < 2 "comparable or slightly better" than diffusion | 2D: better at few steps (β = 1: 2×), comparable at many steps on the paper GMM; worse on `gmm9w` at many steps | §2, §4 |
| paper: few steps favour λ ≈ 1, small β; many steps favour λ ≈ 0, β ≈ 2 | few steps: yes (C ≫ B everywhere). Many steps: MMD² ties; B β = 2 has the best mode weights on `gmm9w` (TV 0.019 vs C 0.033) | §4 |

Sources:
- Lyons, *Distance covariance in metric spaces*, Ann. Probab. 41(5) 2013, https://arxiv.org/abs/1106.5758:
  - Remark 3.19: "if (X, d) has negative type, then (X, d^r) has strong negative type when 0 < r < 1";
  - the introduction gives "an L1 metric space" as an example with negative type but not strong negative type.
  - Note: the URL in `_task.md` (arXiv 1205.0411) is a different paper (Sejdinovic et al. 2012).
- Gneiting & Raftery 2007, JASA 102(477), §4.3 (energy score).
- Paper: `third_party/arxiv_src/x/main.tex`.

## 8. What to try next (2D)
- Second seeds for the promising `gmm9w` variants: C with pseudo-Huber c = 0.1, L1^0.5 and Laplace. Cheap (≈ 5 CPU-h each). Plus
  C with β ∈ {0.5, 1} on `gmm9w`, to see where the β gain on that dataset peaks.
- λ between 0 and 1 with β = 1 (the paper's image sweep has λ ∈ {0.1, 0.5}). It may combine B β = 1's few-step gain with C's weights.
- Do not use β ≥ 1.99 with λ = 1, or L1^1 with λ = 1.

## 9. E3: CIFAR-10 trial (X4)
Six class-conditional pixel-space runs with the paper's App. I.2 U-Net (140 M parameters), 30k steps at batch 128, one training seed
each. Plan, shared settings and cost: `cifar/runs/PLAN.md`. Jobs: `cifar/runs/JOBS.md`. Code: `cifar/src/ddmcifar/` (X2).

### 9.1 Protocol (identical for every run)
- FID-10k: clean-fid `legacy_tensorflow` against the CIFAR-10 *train* statistics. Every JSON records the stats file's sha256, which
  equals transport_map's file.
- Samples: labels i mod 10 (1000 per class), sampler seed 0, EMA weights (0.9999 with warm-up), Algorithm 2 sampler (churn 1). A
  uses x̂ = x_t + t·v̂. One NFE is one network call.
- Snapshots 10k / 20k / 30k at NFE 10 and 100 for every arm. The spec asked for NFE 10 for B/C and NFE 100 for A; X2 ran both for every
  arm. Final (30k) EMA at NFE {2, 4, 8, 10, 15, 30, 100}. Grids at NFE 10 and 100.
- ξ-spread: per-pixel std of x̂(t, x_t, ξ) over 8 ξ draws, averaged over 256 test images.
- Raw JSON: `cifar/runs/<run>/eval/`. Tables: `cifar/runs/eval/fid_summary.md` and `fid_table.csv`, from
  `cifar/scripts/x4_report.py`. Figures: `cifar/figures/`.
- Noise floor: see §9.2 and `cifar/runs/eval/real_floor.json`.

### 9.2 Results [E]
Final EMA, FID-10k, sampler seed 0, one training seed per run. Lower is better. Best per column in bold.

| run | ρ, β, λ | NFE 2 | NFE 4 | NFE 8 | NFE 10 | NFE 15 | NFE 30 | NFE 100 |
|---|---|---|---|---|---|---|---|---|
| `cifar_A` | diffusion, v-target | 102.86 | 56.01 | 33.61 | 29.03 | 22.58 | 15.43 | **9.20** |
| `cifar_B` | ‖·‖², λ = 0 | 117.98 | 63.29 | 39.51 | 34.30 | 27.31 | 19.41 | 13.99 |
| `cifar_B_b1` | ‖·‖¹, λ = 0 | 112.68 | 63.06 | 38.86 | 33.69 | 26.62 | 18.29 | 11.83 |
| `cifar_B_pH` | pseudo-Huber c = 0.030, λ = 0 | 113.07 | 62.44 | 38.55 | 33.24 | 26.11 | 18.10 | 11.73 |
| `cifar_C` | ‖·‖^0.1, λ = 1 | 201.27 | 72.08 | **23.01** | **15.91** | **9.29** | **7.75** | 10.49 |
| `cifar_C_dino` | C + DINOv2 energy term | **49.35** | **42.57** | 44.37 | 46.75 | 49.71 | 54.91 | 62.43 |

Training progress (EMA snapshots, sampler seed 0):

| run | 10k / 20k / 30k at NFE 10 | 10k / 20k / 30k at NFE 100 |
|---|---|---|
| `cifar_A` | 34.24 / 30.77 / 29.03 | 14.40 / 10.30 / 9.20 |
| `cifar_B` | 41.20 / 36.58 / 34.30 | 20.16 / 15.83 / 13.99 |
| `cifar_B_b1` | 38.87 / 36.15 / 33.69 | 16.92 / 13.17 / 11.83 |
| `cifar_B_pH` | 39.59 / 36.11 / 33.24 | 17.50 / 13.33 / 11.73 |
| `cifar_C` | 43.68 / 23.92 / 15.91 | 14.20 / 10.39 / 10.49 |
| `cifar_C_dino` | 71.36 / 63.68 / 46.75 | 108.36 / 107.58 / 62.43 |

ξ-spread (`cifar/figures/xi_spread.png`, `cifar/runs/<run>/eval/xi_spread_snap_0030000_ema.json`), in pixel units on [−1, 1]:

| run | t = 0.02 | t = 0.1 | t = 0.5 | t = 0.9 | t = 0.98 |
|---|---|---|---|---|---|
| `cifar_B` | 0.0021 | 0.0015 | 0.0017 | 0.0024 | 0.0018 |
| `cifar_B_b1` | 0.0013 | 0.0011 | 0.0016 | 0.0029 | 0.0027 |
| `cifar_B_pH` | 0.0013 | 0.0011 | 0.0016 | 0.0028 | 0.0026 |
| `cifar_C` | 0.0111 | 0.0400 | 0.1516 | 0.3614 | 0.4008 |
| `cifar_C_dino` | 0.0194 | 0.0510 | 0.2346 | 0.3860 | 0.3172 |

Figures:
- `cifar/figures/fid_vs_nfe.png`: our runs (left) and the paper's App. table (right). The paper's numbers are FID-50k after 1e6 steps,
  so they are context only.
- `cifar/figures/fid_vs_step.png`
- `cifar/figures/xi_spread.png`
- `cifar/figures/grids_nfe10.png`, `grids_nfe100.png`: one row per class.

**Noise floor [E]** (`cifar/runs/eval/real_floor.json`: real images through the same clean-fid call, on CPU):

| real set | FID-10k vs train stats |
|---|---|
| CIFAR-10 test, first 10k | 3.15 |
| 3 disjoint random 10k train subsets | 2.10 / 2.11 / 2.11 |

- A perfect generator scores ≈ 2.1 in this protocol (finite-sample bias of FID-10k).
- Redrawing 10k real images moves FID-10k by only ≈ 0.01.
- Generated sets far from the data have a larger sampling spread [G]. {{CPUSEED}}
- **Not measured:** the training-seed spread. The planned GPU check (sampler seeds 1 and 2 at NFE 2 / 4 / 10 / 100 for every run;
  `cifar/scripts/eval_seeds_gpu.slurm`, jobs 24821915 / 24821916) was cancelled before it started. `l40s_public` was mostly draining,
  with an estimated start 12 h later. Running it is a ready next step (§10).

### 9.3 Answers
**Does the DDM train on CIFAR-10 in a short run? Yes [E].**
- All six runs reached 30k steps with finite losses. FID at NFE 10 and 100 fell from the 10k to the 30k snapshot for every run. Two
  exceptions: C at NFE 100 between 20k and 30k (10.39 → 10.49), and C + DINOv2 at NFE 100, flat at 108 from 10k to 20k.
- C (β = 0.1, λ = 1) learned a real conditional spread. Its ξ-spread rises from 0.011 at t = 0.02 to 0.40 at t = 0.98. For
  comparison, the pixel std of CIFAR-10 in [−1, 1] is ≈ 0.5 [G: standard value, not measured here].
- C is the best run at NFE 8–30. At NFE 15 it gives FID 9.29 against A's 22.58, and at NFE 30 7.75 against 15.43: about 2× better.
  This matches the paper's claim that the distributional loss helps at few steps.
- At NFE 2 and 4 C is *worse* than A (201 vs 103; 72 vs 56). This is unlike the paper, where the distributional model is best at
  NFE 2 (29 vs 81, FID-50k after 1e6 steps). Two guesses [G]:
  - at NFE 2 the first step starts at t = 0.99, where C's spread is largest and probably least converged after 30k steps;
  - the paper picks the best (β, λ) for each NFE, and we have one setting.
- At NFE 100, A is best (9.20), C is close (10.49), and B β = 2 is worse (13.99).

**B (β = 2) vs B (β = 1) vs B (pseudo-Huber): does β or ρ change FID?**
- **At many steps, slightly yes [E, one training seed].** At NFE 100, β = 1 gives 11.83 and pseudo-Huber 11.73, against 13.99 for
  β = 2. The gap is about 15 %, and it appears at all three snapshots (10k: 16.9 / 17.5 vs 20.2; 20k: 13.2 / 13.3 vs 15.8). At NFE 30 the
  gap is 1.1–1.3 FID.
- **At few steps, no [E].** At NFE 4–15, the three B variants are within 1.2 FID of each other. That is smaller than the gap between
  B and A. At NFE 2, β = 1 is 5 FID better (113 vs 118), on a scale where all three are far from usable.
- This is the opposite of the paper GMM, where β = 1 helped at few steps (2× at NFE 5) and tied at many (§2). On CIFAR-10 the B
  denoiser is a near-deterministic regression in 3072 dimensions. There the geometric median and the mean are close, and β mostly
  changes the per-sample weighting of the loss (‖d‖^(β−1)) [G].
- ξ does nothing in any B variant, as theory says for λ = 0. Its spread stays at 0.001–0.003 for every t, 5× (t = 0.02) to 220×
  (t = 0.98) below C's. That level is about what bf16 rounding alone would give [G].
- **Pseudo-Huber = β = 1 here [E].** B_pH and B_b1 differ by at most 0.7 FID at every NFE and snapshot, and their ξ-spreads agree to
  1e-4. With c = 0.00054·√3072 = 0.030 and ‖x̂ − x_0‖ ≫ c over 3072 pixels, √(‖d‖² + c²) − c ≈ ‖d‖ − c. iCT's c was tuned for the
  gap between two consistency-model outputs, which is much smaller than a denoising residual [G].
- **B never beats A, not even B β = 2, which has the same minimiser (the posterior mean) [E].** Candidate causes, not tested [G]:
  - A's loss is ‖v̂ − v‖² = ‖x̂ − x_0‖²/t², because x̂ − x_0 = t(v̂ − v). Under the same sigmoid weight, A weights small t by an extra
    1/t².
  - B spends 4 particles per image on a ξ it ignores, and it has 6 input channels.
  - With clip 1, A and every B variant clip on 100 % of steps, so the effective step is a normalised gradient. Its direction differs
    between loss forms.

**C and C + DINOv2 vs A at few steps.**
- **C + DINOv2 has the best FID at NFE 2 and 4 by far [E]:** 49.4 and 42.6, against A's 102.9 / 56.0 and C's 201.3 / 72.1.
- **But its FID gets *worse* with more steps** (44 at NFE 8, 62 at NFE 100; 108 at NFE 100 for the 10k and 20k snapshots). Its grids
  show strong high-frequency texture at every NFE (`cifar/figures/grids_nfe10.png`, `grids_nfe100.png`). Plain C's samples have none.
- So the DINOv2 term helps FID at 2–4 steps but makes the images visibly wrong. This is the risk the task named, after iCT: a feature
  loss can move FID for reasons other than image quality. DINOv2 is not the FID network, but both are ImageNet-trained feature
  extractors [G].
- Probable mechanism [G]: the feature term rewards matching DINOv2 features, and the network learns textures that do that. More
  sampler steps at small t apply the denoiser more often, and add more of that texture. C + DINOv2's ξ-spread at t = 0.02 is 1.7× C's.
  Its gradient clipping also rose from 37 % of steps (first 5k) to 99 % (last 5k), against 0 % for C (`cifar/runs/*/log.jsonl`).
- This differs from transport_map's one-step generator, where the same DINOv2 term helped a lot (`/scratch/wg2381/transport_map/docs/experiments_unet.md`).
  There the network takes one step. Here the term is applied at every t, including t ≈ 0, where x̂ must equal x_t to pixel precision
  [G].

### 9.4 Caveats
- 30k steps at batch 128 against the paper's 1e6 at 256: about 1.5 % of the paper's training images. Every run is far from
  converged, and rankings can change with more training.
- One training seed per run. All runs share the data order, the t draws and X_1 (seed 0), so arm differences are not inflated by data
  noise. But the training-seed spread is unknown. In 2D, seed ranges were often tens of percent of the mean
  (§2–§5 tables). So the ≈ 15 % β gap at NFE 100 is suggestive, not established.
- FID-10k, not FID-50k. FID-10k is biased upward against FID-50k, by a roughly constant amount for a given model quality [G], so
  compare within this table only.
- A feature loss (C + DINOv2) can move FID for reasons other than quality, as seen in the grids. LPIPS was rejected for this reason
  and because it is not of negative type (`_task.md` E4).
- Gradient clipping at 1 binds on every step for A and B, never for C, and on 37–99 % of steps for C + DINOv2. The shared setting acts
  differently per arm.
- The sigmoid weighting with bias 0 was not tuned. The paper sweeps the bias from −2 to 2.

## 10. What to try next (overall)
In order of information per GPU-hour:
1. **Is the CIFAR β effect real?**
   - Run the sampler-seed check that was cancelled: two `cifar/scripts/eval_seeds_gpu.slurm` jobs, ≈ 3 GPU-h in total.
   - Train a second seed of `cifar_B` and `cifar_B_b1` (≈ 9 GPU-h each).

   This is the only β effect measured on images, and it rests on one training seed. The chain used 54.4 of its 75 GPU-h.
2. **Does the 2D C winner transfer?** Train `cifar_C` with β = 1.5, which beat β = 0.1 on `gmm9w`, and one λ between 0 and 1
   (e.g. 0.5, β = 1). The paper's image sweep has both.
3. **A vs B confound:** B β = 2 with an extra 1/t² weight, which makes its loss equal to A's velocity loss up to the ξ input and
   m = 4. If it then matches A, the B–A gap is the weighting, not the energy loss.
4. **DINOv2 term without textures:** apply it only at large t (e.g. t > 0.3), or anneal it. Score with a second metric besides
   Inception FID (e.g. KID, or precision/recall), to check that the few-step gain is not just in feature space.
5. **Pseudo-Huber with a scale that matters:** c of the order of a typical ‖x̂ − x_0‖ over 3072 pixels (≈ 5–20 here [G]). At iCT's c
   it is just β = 1.
6. **Longer training for the few-step regime:** C loses to A at NFE 2–4 after 30k steps. The paper's 1e6-step C wins there. A 100k–200k
   run of A and C would show whether the crossing moves.
7. 2D items are in §8.
