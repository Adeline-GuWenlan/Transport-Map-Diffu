# Reproducibility report: DDM 2D experiments (GMM §6.1 / Fig. 3–4; checkerboard App. K.1 / Table 2)

Paper: De Bortoli, Galashov, Guntupalli, Zhou, Murphy, Gretton, Doucet, *Distributional Diffusion Models with Scoring Rules*, ICML 2025,
arXiv 2502.02483 v3. Spec: `docs/task_spec.md`. Chain S1–S4, 2026-11-19 to 2026-11-20, CPU only. Labels: **[E]** evidence in a committed
file, **[G]** guess / assumption.

## 1. Primary success criterion
> Does the distributional method reproduce the paper's substantial low-NFE improvement while keeping an explicitly matched baseline and
> evaluation protocol?

**Yes, qualitatively, on both datasets and for all 3 training seeds [E].**

| | NFE | A (standard diffusion) | B (β = 2, λ = 0) | C (β = 0.1, λ = 1) | A / C ours | A / C paper |
|---|---|---|---|---|---|---|
| GMM | 5 | 1.66e-2 | 1.74e-2 | 1.8e-4 | **92×** | 101× |
| GMM | 10 | 4.7e-3 | 5.6e-3 | 3.1e-4 | **15×** | 5× |
| checkerboard | 5 | 3.04e-2 | 3.44e-2 | 3.1e-4 | **99×** | 45× |
| checkerboard | 10 | 7.2e-3 | 9.0e-3 | 3.4e-4 | **22×** | 11× |

MMD²_u (RBF σ = 1, 4096 vs 4096), mean over 3 training seeds × 10 evaluation repeats. Sources: `results_gmm.md` §3.1, `results_cb.md` §2;
the paper's ratios are computed from `paper_values/`.

What "matched" means here: the protocol below is identical for every arm, and nothing was tuned for C.

| What is matched | How |
|---|---|
| Training | same data set, steps, batch, optimizer, clipping, EMA, schedule, safety, weighting and seeds (`configs/paper/*.json` differ only in arm / β / λ / m) |
| Network | same backbone; A has no ξ input and an output of width 2; B and C share the stochastic architecture |
| Sampler | Eq. (4) bridge, churn 1, uniform grid, one denoiser call per NFE; the same sampler RNG seeds for every arm |
| Evaluation | EMA weights of the 100k-step checkpoint; 4096 samples vs fresh 4096-point reference sets; 10 repeats; same NFE grid |
| Control | B has C's architecture and C's m = 32 but β = 2, λ = 0. It behaves like A on both datasets (above), so the gain comes from the loss, not from the extra input or the particle count |

The seed ranges of A/B and C are disjoint at NFE 5–30 (GMM) and NFE 5–10 (checkerboard). Over all (A seed, C seed) pairs on the
checkerboard, A / C is 61–156 at NFE 5 and 15–31 at NFE 10.

**Quantitative agreement is partial.**
- **GMM NFE 5** matches the paper's figure for all three arms (ours 1.66e-2 / 1.74e-2 / 1.8e-4; paper 1.62e-2 / 1.59e-2 / 1.6e-4).
  Fig. 3 right (posterior std) and Fig. 4 (posterior samples) also reproduce.
- **Checkerboard:** the paper does not define its metric, so numbers can only be compared as "unverified" (`audit.md` §5). Our A/C gap at
  NFE 5–10 is 2× *larger* than the paper's.
- **Large NFE:** in the paper both methods tie at NFE ≥ 100. In ours, all arms are limited by low-noise denoiser accuracy, and on the
  checkerboard they get *worse* from NFE 200 on (§4).

## 2. What was run
| | |
|---|---|
| Runs | 18 paper runs: {GMM, checkerboard} × {A, B, C} × seeds {0, 1, 2}; 100k steps each. All complete; final checkpoints verified by sha256 and load tests (`runs/paper/STATUS.md`, `runs/paper/CKPT_MANIFEST.sha256`) |
| Hardware | CPU only (no GPU available to this chain: the user's `gpu48` QOS was held by other chains; `_rules.md`). 1 thread per run, 6 runs per 6-CPU `cpu_short` job |
| Training compute | 96.7 CPU-h by `sacct` (3 jobs + 3 continuations × 6 CPUs; budget 600). Wall time 3.5–5.8 h per run |
| Evaluation compute | inline in the agent allocations (8 CPUs): GMM ≈ 35 min, checkerboard ≈ 1.5 h incl. diagnostics |
| Diagnostics | 2 × 10k-step fine-tunes (`runs/diag/`, §4.4), never mixed into paper tables |
| Software | PyTorch 2.14.1 (float32 training, float64 MMD), numpy 2.4.6, scipy 1.17.1. Our own implementation; the third-party `edluyuan/ddm` was audited, not used (no license; two bugs found, `audit.md` §2) |
| Tests | 79 unit tests pass (`runs/paper/eval_cb/logs/pytest_s4.log`): schedule marginals, bridge (DDIM/DDPM limits), Eq. 14 vs a double loop, MMD vs naive and closed form, both exact posteriors vs importance sampling, model shapes, sampler NFE counting, bitwise checkpoint resume |

## 3. Results by experiment
| Paper item | Our output | Agreement |
|---|---|---|
| Fig. 3 left (GMM MMD² vs steps) | `figures/gmm/fig3_left.png`, `runs/paper/eval_gmm/fig3_left_mmd2_raw.csv` | NFE 5: numeric match for all three arms. NFE 10–30: A/B ≈ 2× above the paper (§4.1). NFE ≥ 50: all arms within MC noise of each other in ours and in the paper |
| Fig. 3 right (posterior std vs t) | `figures/gmm/fig3_right.png`, `posterior_std.json` | our C is within 0–9 % of the paper's C at t ≤ 0.88; B ≈ 0 in both. The paper's exact-posterior curve uses an unstated reduction (§4.5) |
| Fig. 4 (posterior samples) | `figures/gmm/fig4_seed{0,1,2}.png` | C covers both posterior modes and B collapses to the mean, as in the paper. C under-weights the far mode at t = 0.99 (21–32 % vs 49–50 %) |
| Table 2 (checkerboard) | `runs/paper/eval_cb/table2.md`, `figures/cb/metric_vs_nfe.png`, `figures/cb/samples_grid_seed*.png` | ordering at NFE 5–10 and the crossover near NFE 50 reproduce; absolute values "unverified"; NFE ≥ 200 degrades (§4.3) |
| Loss curves | `figures/gmm/loss/`, `figures/gmm/loss_overview_{gmm,cb}.png` | — (not in the paper). Finding: the β = 2 arms (A, B) have isolated loss/gradient spikes; C never does |

## 4. Discrepancies and what we checked
### 4.1 The diffusion baseline at few steps is worse than the paper's (GMM NFE 10–30; checkerboard NFE 5–10)
- **Exact-posterior oracles.** We drove our sampler with the exact posterior mean, i.e. the best any x_0-regression model can do with it.
  GMM: 4.9e-3 at NFE 10. Checkerboard: 3.3e-2 / 7.7e-3 at NFE 5 / 10. Our trained A and B are at this oracle level (0.71–1.24×) on both
  datasets **[E]**: `eval_gmm/oracle_mmd.json`, `eval_cb/oracle_mmd.json`.
- The paper's baseline is 2× (GMM, NFE 10) to 3–5× (checkerboard, NFE 5–10) below the oracle.
- Tried, identically for all arms; none reaches the paper's values **[E]**:
  - churn 0 (closer: 2–3× above the paper on the checkerboard);
  - a time grid restricted to the safety interval (worse);
  - smaller checkerboards ×0.5 and ×0.25 (oracle 3.8e-2 and 2.0e-2 at NFE 5);
  - σ = 0.5 and 2;
  - D_ρ = ½MMD² (a constant factor cannot fit every NFE).
- **Open [G]:** the paper's baseline sampler, time grid or metric must differ in a way the text does not state. This changes the *size* of
  the improvement, not its direction.

### 4.2 Table 2's metric is undefined
- The paper says only "MMD". There is no code, Table 2 is absent from v1 and v2, and the OpenReview discussion is not readable from the
  cluster.
- Tests against the null floors (`audit.md` §5.2):
  - "MMD" unsquared is implausible.
  - A biased MMD² at n = 4096 is ruled out: its floor of 4.7e-4 lies above the paper's values.
  - A biased MMD² at n ≈ 10,000 fits the paper's flat ≈ 1.7e-4 level: the paper's large-NFE values could be the estimator floor alone **[G]**.
- We report MMD²_u (primary), MMD²_b, σ ∈ {0.5, 1, 2} and n = 10,000, and mark every numeric comparison "unverified".

### 4.3 Accuracy at low noise limits every arm at many steps (both datasets)
- **Measured against the exact posterior** (`eval_cb/denoiser_err.json`, `figures/cb/denoiser_error_vs_t.png`; GMM: `results_gmm.md` §3.3b, §4):
  - checkerboard A at t = 0.01: RMS error of x̂ = 0.048, while the posterior std is 0.010;
  - checkerboard C at t ≤ 0.02: the spread of its draws is 0.059, against an exact 0.001–0.02;
  - GMM C at t = 0.01: std 0.036 vs 0.009 (the paper's own Fig. 3 right shows 0.039 vs 0.012).
- With churn 1, fine grids integrate this error. The x̂ weight per step is ≈ 2h/t, summing to ≈ 2·ln(t_max/t_min) **[derivation]**.
- Consequences on the checkerboard:
  - MMD²_u rises from NFE 100 to NFE 1000 (A 7.2e-4 → 3.3e-3, C 4.3e-4 → 1.3e-3).
  - A and B at NFE 1000 show filament artefacts.
- What is and isn't the cause **[E]**:
  - The exact-draw oracle stays ≈ 0 up to NFE 200, so the sampler itself is not the cause.
  - The low-t clamp contributes only 2–4e-4.
  - Feeding the true t < 0.01 (`clamp_low=False`) makes the networks much worse, because that is outside their training range.
  - A safety-interval grid halves the NFE-1000 error but does not remove it.
- The paper's values at NFE 1000 (1.7–1.8e-4) imply more accurate low-noise denoisers, or a floor-dominated metric **[G]**.

### 4.4 Is the low-noise floor optimisation noise? (diagnostic)
Diagnostic only (`results_cb.md` §5.5, `runs/diag/diag_summary.md`). Checkerboard A and C for seeds 0–2 and GMM A and C for seed 0 were
continued for 10k steps at a constant lr of 1e-4 instead of 1e-3, with everything else unchanged. Every arm got the same treatment.
- **Checkerboard C**, NFE 5 / 10 / 50 / 100, seed means: 3.1 / 3.4 / 3.8 / 4.3e-4 → **1.2 / 1.2 / 1.0 / 1.2e-4**. That is at or below the
  paper's distributional column. The low-noise spread halves (0.06 → 0.03 at t = 0.01) on every seed.
- **Checkerboard A** improves at NFE 50–100 (6.6–7.2e-4 → 3.4–3.9e-4) but not at NFE 5–10, where it is at the sampler's oracle floor.
  A / C becomes 270× at NFE 5 and 56× at NFE 10.
- **GMM (one seed).** A improves 1.7–4× at NFE 20–100. C's low-noise spread halves, but its MMD² changes are within noise; it was already
  near the evaluation floor.
- **NFE 1000** improves but is still worse than NFE 100 for every arm.
- **Conclusion [E].** Much of our distance from the paper's distributional column, and part of the large-NFE floor, is convergence under the
  constant lr 1e-3 / EMA 0.99 that App. I.1 specifies. The few-step advantage of C *grows* when both arms converge further. The paper runs
  and tables keep the paper's configuration.

### 4.5 Smaller items
- *GMM Fig. 3 right, exact-posterior curve.* The paper's curve is 1.31× ours at t ≤ 0.45. "Mean distance to the 8-draw mean" reproduces it
  there but not at high t. No single std reduction fits all of the paper's curves (`results_gmm.md` §4).
- *Loss spikes in A and B (β = 2).* They are isolated, recover within one 100-step window, and are not visible in any final checkpoint. One
  EMA snapshot (gmm_A_s2 at 50k) is hit (`results_gmm.md` §2).
- *Fig. 4 row order.* We follow the binding spec (top row x_0 = (3, 3)); the paper's caption has the opposite order. Every row is labelled.

## 5. Deviations from the paper (full list: `docs/paper_vs_implementation.md`, 49 rows)
Most consequential, all applied identically to every arm:
- **Checkerboard dataset and training config.** FFJORD generator on [−4, 4]², 102,400 points, and the GMM's App. I.1 config with
  β = 0.1, λ = 1, m = 32. These are borrowed or approximated; the paper and Galashov et al. do not specify them.
- **Unstated 2D details.** Churn ε = 1; sigmoid bias b = 0; training t ~ U[0.01, 0.99] with the sampler clamping the denoiser time to
  [0.01, 0.99]; the baseline sampler is the ε = 1 bridge with a deterministic x̂; gradient clipping before Adam; time input 1000·t.
- **Bridge.** The App. G form of r_{i,j}; the main-text form does not preserve the marginals (`audit.md` §1.2).
- **Evaluation.** 3 seeds × 10 evaluation repeats instead of (presumably) one; the unbiased MMD² is primary.
- **Hardware.** CPU / PyTorch instead of TPU-GPU / JAX.

## 6. Integrity
- No number was tuned, backfilled or selected per arm. All raw metrics are committed: `runs/paper/eval_gmm/`, `runs/paper/eval_cb/`
  (JSON/CSV; samples as `.npz`).
- Paper values are only in `docs/paper_values/` and in columns labelled "paper".
- Smoke runs (`runs/smoke/`) and diagnostics (`runs/diag/`) appear in no paper table or figure.
- Every evaluation uses the EMA weights of `final.pt` (100k steps) with the same seeds for every arm. Snapshots are a separate study.

## 7. Open questions for the user
1. The paper's baseline sampler, time grid and metric for Fig. 3 left and Table 2 (§4.1, §4.2). The OpenReview rebuttal (forum
   `N82967FcVK`) may contain Table 2's details; it could not be read from the cluster.
2. Whether the low-noise floor (§4.3–4.4) should be attacked: e.g. lr decay, a longer EMA, or an x_t-skip parametrisation. This would be a
   change to the paper's configuration, so it was not applied to the paper runs.
