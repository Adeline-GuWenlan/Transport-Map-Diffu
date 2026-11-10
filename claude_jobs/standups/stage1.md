Status: DONE

# Stage 1 standup: derive and verify the energy-score loss

Written 2026-11-06 ~03:10 EST by the stage-1 agent (Slurm job 22047080). This is the first stage; there were no
earlier standups to read. The job was interrupted once (about 02:00–02:23) and resumed from what was on disk.

## Verdict on the user's formula
The formula in CLAUDE.md is **not correct as written**, but it is one symbol away from the energy score. The
nested expectation in the first term is fine: with x independent of y, it equals the joint expectation. The
minus sign and the factor ½ are right. The ½ is exactly the strictly proper energy score: coefficient 1 makes
the loss unbounded below, and coefficient λ/2 with λ < 1 under-disperses. The **second term is wrong**. As
typed, `E[F(x) − F(x')]` is a raw vector difference whose expectation is 0, so the loss reduces to
`E‖F(x) − y‖`. That is minimised by a point mass at the geometric median of the data: total mode collapse.
Proved in the note (Prop. 13) and reproduced in the 2-D toy, where all samples sit at one point. The fix is
the **norm** `½ E‖F(x) − F(x')‖`, with the same power β as the first term. x and x' must be i.i.d. from the
**source Gaussian** (not the data), independent of y, and in the conditional case must **share y's class
label**. "Euclidean distance" must mean the norm (β = 1); squared norms (β = 2) fit only the mean.

## Final loss (one line of LaTeX)
```
\mathcal L(\theta)=\mathbb E_{(Y,C)\sim Q}\,\mathbb E_{Z,Z'\overset{iid}{\sim}\mathcal N(0,I)}\Big[\|F_\theta(Z,C)-Y\|-\tfrac12\|F_\theta(Z,C)-F_\theta(Z',C)\|\Big] \;=\; \tfrac12\sum_c\pi_c\,D_1(P^c_\theta,Q_c)+\text{const}
```
(Z, Z') is independent of (Y, C). Unconditional: drop C. General β ∈ (0, 2): use ‖·‖^β in both terms. β = 1 is the choice.

## Estimator stage 4 must implement (note Sec. 7, eq. (9); reference code `docs/stage1_checks/es_core.py`)
- **β = 1.**
- **Batch:** N labelled data points, grouped **by class** within the batch. Class c has n_c data points
  y and m_c ≥ 2 generated samples x = F(z, c), with z fresh N(0, I) independent of the data. Simplest
  option: one generated sample per data point (m_c = n_c), which costs the same network passes as a normal
  batch. Use class-stratified batches so every class present has ≥ 2 points, or draw r = 2 samples per data point.
- **Loss:** `Σ_c (n_c/N) [ mean_{i,j} ‖x_j − y_i‖ − 1/(2 m_c(m_c−1)) Σ_{j≠k} ‖x_j − x_k‖ ]`.
  The confinement term pairs every class-c sample with every class-c data point. The interaction term is a
  U-statistic: the diagonal is excluded, so divide by m_c(m_c − 1), not m_c². It uses **only same-class
  pairs**. This is unbiased for the expected conditional energy score (Prop. 15).
- **Do not:** pool the interaction across classes (wrong minimiser: Prop. 12, check d); keep the diagonal
  (a V-statistic, equivalent to λ = (m−1)/m < 1, under-disperses); or compute `sqrt(sum sq)` on the full
  matrix and mask afterwards (NaN gradient).
- **Zero distances:** compute ‖u‖ with the double-where trick (`torch_pow_dist` in es_core.py): exact value,
  gradient 0 at u = 0. This matters at step 0 for DiT. Its adaLN-Zero init zeroes the final layer
  (DiT `models.py` `initialize_weights`, commit ed81ce2), so F(z, c) ≡ 0 and every generated pair has
  distance exactly 0 at step 0. A naive sqrt gives NaN at step 0 and destroys training (check f).
  `torch.cdist` gives finite gradients, but its matmul path reports ~0.04 instead of 0 for identical
  3072-d rows in float32. Exclude the diagonal by index.
- **Metric to log:** per-class held-out unbiased Ď₁ = 2·mean‖x−y‖ − U-mean‖x−x'‖ − U-mean‖y−y'‖. It can be
  < 0 on a single batch; its expectation is ≥ 0 and is 0 iff model = data.
- **Cost:** distances O(Σ_c (n_c m_c + m_c²) d) ≈ 4e8 multiply-adds for N = M = 256, d = 3072, negligible
  next to the network. Use matrix form or cdist, not an (M, N, d) broadcast (0.8 GB).

## Numerical checks (all CPU, 17 min total via `bash docs/stage1_checks/run_all.sh`, reproduced end to end at 03:08; `docs/stage1_checks/`, figures in `docs/figures/`)
- **(a) Estimator.** P = Q (2-D, n = m = 32, 4000 reps): U-statistic mean 0.0014 ± 0.0010; negative in
  61% of reps; V-statistic biased (+0.112). Shifted Gaussian: 0.1102 ± 0.0020 vs closed form 0.1099. 1-D
  Gaussian closed forms: all five cases within |z| < 1.9. Fourier constant C(d, β) of the Székely–Rizzo
  lemma matches quadrature to < 2e-8 (d = 1, 2, 3; β = 0.5, 1, 1.5). Variance vs m: plateaus at the
  data-noise floor σ²/n. At equal compute, per-class pooling has up to 19× lower loss variance than
  per-data-point (engression-style) groups.
- **(b) β = 2.** N(0,1) vs N(0,4): D̂₂ = 0.002 ± 0.002 (closed form 0), D̂₁ = 0.1833 ± 0.0010 (closed
  form 0.1831). The expected β = 2 score is flat in the scale. λ-weighted interaction: minimiser
  s² = λ²/(2 − λ²) confirmed to 4e-8; λ = 2 is unbounded below.
- **(c) 2-D toy** (MLP 2→256³→2, 8 Gaussians, 3000 steps, cosine LR, 3 seeds; held-out D̂₁; source
  Gaussian 0.241, point mass 1.47):
  corrected β = 1: **0.0098 ± 0.0005**; β = 1.5: 0.0159 ± 0.0011.
  Literal reading (raw difference): **1.46**, collapsed to a point (std ≈ 0.002). x, x' from the data:
  1.46, collapsed. Squared norms (β = 2): 1.46, collapsed. Coefficient 1 instead of ½: diverges (std ~3e8).
  ‖·‖ then −½‖·‖²: diverges (std ~1e9).
  **(c2)** After 3000 steps the corrected model learns the ring but separates modes only partly (49% of
  samples within 3σ of a mode). After 20000 steps D̂₁ = 6e-5 and 84% (target 99%): fine structure is
  learned slowly.
- **(d) Conditional grouping** (1-D, classes N(∓3, 1)): grouped gives per-class std 0.94–1.06 and
  D̂₁ ≤ 0.015. Pooled interaction gives std 0.60–0.65, class means pushed out to ±3.7–3.95, and
  D̂₁ 0.39–0.63. The 2-D interleaved version shows a smaller, seed-dependent effect (reported in the note).
- **(e)/(f) Gradient at 0:** see the estimator section above.

## Files
- `docs/energy_score_loss.tex`, `docs/energy_score_loss.pdf` (19 pp., compiles cleanly with tectonic; read
  back including equations and figures).
- `docs/stage1_checks/` (README.md, `run_all.sh`, `es_core.py`, `check_*.py`, `results/*.json`, `logs/`).
- `docs/figures/check_{a_variance,b_beta2,c_samples,c_curves,c2_long,d_conditional,d2_conditional_2d}.pdf`.
- Helper env: `/scratch/wg2381/.conda/envs/tmap-tex` (conda-forge; tectonic 0.17, poppler, numpy 2.4.6,
  scipy 1.17.1, matplotlib 3.11, CPU torch 2.14.1). Not the project env; stage 3 builds `tmap`.
- Source PDFs and text dumps used to check citations (not in the project): `/scratch/wg2381/.cache-home/stage1_refs/`.

## Citations verified against primary text
Gneiting & Raftery 2007 (eqs. 1, 2, 7, 22–24; Thm. 4). Székely & Rizzo 2013 (Props. 1–2, Lemma 1,
eq. 5.2). Sejdinovic et al. 2013 (Def. 13, Thm. 22: D_E = 2·MMD²). Shen & Meinshausen, Engression, JRSSB
2025 (eqs. 12–13, m = 2 per observation). De Bortoli et al. 2025 (eqs. 8, 12, 13; Prop. 4.1). DiT
`models.py` (zero-init). Bibliographic data checked on Crossref. Bellemare 2017, GMMN, Gritsenko 2020 and
Reverse Markov Learning were checked at abstract level only.

## Open questions / notes for later stages
1. **Fine detail converges slowly.** The energy-distance weight ‖t‖^{−d−β} emphasises low frequencies.
   Check (c2) needed ~20k steps to separate 8 modes in 2-D. In pixel space this may show up as blur or
   slow detail. Stage 2 should weigh latent vs pixel space with this in mind.
2. β = 1 vs β ∈ (1, 2) on images: both are strictly proper; untested here.
3. The batch structure for DiT: m_c = n_c with class-stratified batches is my recommendation. Engression
   used m = 2 per observation. Stage 2 picks N.
4. The note's unbiased-gradient argument assumes F_θ is locally Lipschitz in θ with an integrable constant
   and P_θ atomless. This is plausible for DiT but not verified.
