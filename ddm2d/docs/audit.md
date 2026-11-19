# Audit: paper vs. reference code (S1, 2026-11-19)

Sources read (all fetched 2026-11-19; clones are gitignored, see `third_party/README.md`):

| Key | Source | Version / hash |
|---|---|---|
| **P** | De Bortoli et al., *Distributional Diffusion Models with Scoring Rules*, ICML 2025. PDF https://arxiv.org/pdf/2502.02483 | v3 (7 Jul 2025); PDF sha256 `1db1c19d…`, 46 pages |
| **P-tex** | LaTeX source https://arxiv.org/e-print/2502.02483 (`main.tex`, 2187 lines) | tarball sha256 `f989b28c…` |
| **R** | Third-party PyTorch code https://github.com/edluyuan/ddm | commit `8a43f77d9bb528937c96a712d34abf5473af2c3f` (2025-09-22). **No LICENSE file** |
| **F** | FFJORD https://github.com/rtqichen/ffjord `lib/toy_data.py` | commit `994864ad0517db3549717c25170f9b71e96788b1`, MIT |
| **G** | Galashov, De Bortoli, Gretton, *Deep MMD Gradient Flow without adversarial training*, ICLR 2025. https://arxiv.org/abs/2405.06780 (+ e-print) | v1 (only version on arXiv); tarball sha256 `d8cf1c9c…` |

Equation numbers below are those of the compiled PDF v3 (checked by extracting its text: `(4)` is Σ_{s,t}, `(6)` the diffusion loss,
`(9b)` the RBF kernel, `(10)` D_ρ, `(12)` the conditional score, `(13)` the energy diffusion loss, `(14)` the empirical loss).
Page numbers are PDF pages. Labels: **[E]** evidence in a cited source, **[G]** our guess / assumption.

---

## 1. Paper

### 1.1 Schedule and forward process — Eq. (2), (3), p. 2
- p(x_t | x_0) = N(α_t x_0, σ_t² I) (Eq. 2); flow-matching schedule α_t = 1 − t, σ_t = t (Eq. 3). Used for the 2D runs: App. I.1, p. 33 ("We use the flow
  matching noise schedule (3)"). **[E]**

### 1.2 Gaussian bridge and churn — Eq. (4), p. 2; App. G (pp. 26–29)
- p(x_s | x_0, x_t) = N(μ_{s,t}, Σ_{s,t}), with r_{i,j}(s,t) = (α_t/α_s)^i (σ_s²/σ_t²)^j:
  - μ_{s,t} = (ε² r_{1,2} + (1 − ε²) r_{0,1}) x_t + α_s (1 − ε² r_{2,2} − (1 − ε²) r_{1,1}) x_0
  - Σ_{s,t} = σ_s² (1 − (ε² r_{1,1} + (1 − ε²))²) I. **[E]**
- Caution: in the main text r_{i,j} is defined with (σ_s²/σ_t²)^j, but in App. G (prop. after Eq. 22's section, `main.tex` l. 1475–1480) it is
  r_{i,j} = (α_t/α_s)^i (σ_s/σ_t)^j. The two are inconsistent. We checked which one makes the bridge correct: only the **App. G form** (σ_s/σ_t)^j
  gives the DDPM posterior at ε = 1 (μ = (α_t σ_s²)/(α_s σ_t²) x_t + α_s(1 − α_t²σ_s²/(α_s²σ_t²)) x_0) and the DDIM update at ε = 0
  (μ = (σ_s/σ_t) x_t + (α_s − α_t σ_s/σ_t) x_0, Σ = 0). With the main-text form, ε = 0 would give coefficient σ_s²/σ_t² on x_t, which does not
  preserve the marginal. **We implement the App. G form**; the unit test checks the marginal p(x_s|x_0) = N(α_s x_0, σ_s²) for ε ∈ {0, 0.5, 1}. [E: derivation + test]
- ε ∈ [0, 1] is the **churn** parameter: ε = 0 deterministic (DDIM), ε = 1 stochastic (DDPM). Prop. in App. G.2 (p. 29) maps it to DDIM's η. **[E]**
- **The churn value for the 2D experiments is not stated anywhere.** The Gaussian analysis (§4.1, p. 4) uses ε = 1 "for simplicity". The image
  comparison (App. K.2, p. 39) sweeps ε ∈ {0, 0.25, 0.5, 0.75, 1}. **[E: absent]** → our choice ε = 1 for every arm, see `plan.md`. **[G]**
- Churn ε is a sampling parameter. It is separate from the training **safety ε = 1e-2** (§1.7).

### 1.3 Standard diffusion loss — Eq. (6), p. 2
- L_diff(θ) = ∫ w_t E‖X_0 − x̂_θ(t, X_t)‖² dt: **x_0-prediction** with weighting w_t. **[E]**
- In the image experiments the classical model uses **velocity** prediction (App. I.2, p. 34). For 2D, §6.1 says the baseline optimizes "(6)", which
  is x_0-prediction. We follow (6). **[E]**

### 1.4 Scoring rules, MMD — Eqs. (8), (9b), (10), p. 3
- Generalized kernel score S_{λ,ρ}(p, y) = (λ/2) E_{p⊗p} ρ(X, X') − E_p ρ(X, y) (Eq. 8); k_rbf(x, x') = exp(−‖x − x'‖²/(2σ²)) (Eq. 9b);
  D_ρ(p, q) = S(q, q) − S(p, q) (Eq. 10). **[E]**
- **Factor 2.** With ρ = −k and λ = 1, S(p,q) = −½E_{pp}k + E_{pq}k and S(q,q) = ½E_{qq}k, so D_ρ(p,q) = ½ E_{pp}k + ½ E_{qq}k − E_{pq}k = **½ · MMD²(p,q)** in the
  usual (Gretton et al. 2012) convention. The paper says D_ρ "recovers the squared MMD" and plots "Squared MMD". Whether its numbers are D_ρ or
  the standard MMD² is **unknown**. We compute the standard MMD² and also store D_ρ = MMD²/2. **[E: derivation]**
- Evaluation metric for the GMM: "MMD squared given by D_ρ (10) with ρ = −k_rbf, σ = 1", §6.1 p. 7 and App. I.1 p. 33. 4096 samples
  (App. I.1). Biased (V-statistic) vs unbiased (U-statistic) estimator: **not stated**. **[E: absent]**
- App. I.1 writes "For Figure [snr_study], right, we use the MMD squared": a broken reference in `main.tex` l. 1718 (it should be Fig. 3 left).

### 1.5 Generalized energy score and the training loss — Eqs. (12)–(14), pp. 3–4
- Conditional score (12): S_{λ,β}(p^θ_{0|t}(·|x_t), x_0) = (λ/2) E‖X − X'‖^β − E‖X − x_0‖^β, β ∈ (0, 2], λ ∈ [0, 1]. **[E]**
- Energy diffusion loss (13): L(θ) = −∫ w_t E_{p_{0,t}}[S_{λ,β}] dt. **[E]**
- Empirical loss (14), m particles ξ_i^j per data point:
  L_{n,m} = (1/(nm)) Σ_{i,j} w_{t_i} [ ‖X_0^i − x̂_θ(t_i, X_{t_i}, ξ_i^j)‖^β − λ/(2(m−1)) Σ_{j'≠j} ‖x̂^{ij} − x̂^{ij'}‖^β ]. **[E]**
  - **β** is the exponent of both terms; **λ** multiplies only the interaction term.
  - The **i = j (j = j') terms are excluded**; the interaction is the U-statistic mean over the m(m−1) ordered pairs. Pseudo-code App. F (p. 30,
    `compute_rho_fn`: (1/(m(m−1))) Σ_{j≠j'}; score = 0.5·λ·interaction − confinement; loss = −score) agrees. **[E]**
  - The weight w_{t_i} is applied **per data point** i, inside the sum. **[E]**
  - β = 2, λ = 0 gives the regression loss (6) averaged over ξ (§3, p. 4). **[E]**
  - How ‖·‖^β is made differentiable at 0 (β < 2): not stated. **[E: absent]**
- Gaussian intuition (Prop. 4.1, p. 4): λ < 1 shrinks the variance by f(λ, β) = (2λ^{−2/(2−β)} − 1)^{−1}; λ = 1, β < 2 is strictly proper.

### 1.6 Algorithms 1 and 2 — p. 4
- **Algorithm 1 (training):** t_i ~ U[0, 1]; X_0^i ~ p_0; X_{t_i}^i from Eq. (2); ξ_{i,j} ~ N(0, I), i ∈ [n], j ∈ [m]; gradient step on (14).
  The pseudo-code (App. F) replicates t, x_t, x_0 m times and draws a fresh ξ for each copy; the same x_t is shared by the m copies. **[E]**
- **Algorithm 2 (sampling):** grid 0 = t_0 < … < t_N = 1; X_{t_N} ~ N(0, I); for k = N−1 … 0: ξ ~ N(0, I), Z ~ N(0, I),
  X̂_0 = x̂_θ(t_{k+1}, X_{t_{k+1}}, ξ), X_{t_k} = μ_{t_k,t_{k+1}}(X̂_0, X_{t_{k+1}}) + Σ^{1/2} Z; return X_0. **[E]**
  - **One NFE = one denoiser call.** N grid intervals = N denoiser calls = N NFE, for every arm. At t_0 = 0, σ_0 = 0 so Σ = 0 and μ = X̂_0:
    the last step returns the last prediction. **[E: from Eq. 4]**
  - Grid: "uniform" time schedule for ordinary diffusion and DDM (App. K.2, p. 39). **[E]** (image section; we assume the same in 2D) **[G]**

### 1.7 Safety ε
- 2D: "we use safety epsilon 1e-2" (App. I.1, p. 33). **[E]**
- Its meaning is given only in App. K.2 (p. 39): the "safety" parameter η "specifies the time interval to be [η, 1 − η]". **[E]**
- Whether training times are drawn from U[η, 1 − η] or clipped, and exactly how the sampling grid uses it, is not stated. **[E: absent]**
  Consistent with the interval reading: Fig. 4 shows t = 0.99 and t = 0.01 as the extreme noise levels. **[E: figure titles]**

### 1.8 Time sampling and sigmoid weighting
- t ~ U[0, 1] (Alg. 1), restricted by the safety interval (§1.7). **[E]/[G]**
- Sigmoid weighting w_t = (1 + exp[b − log(α_t²/σ_t²)])^{−1} = sigmoid(log SNR_t − b) (§4.2, p. 5, citing Kingma et al. 2021 and Hoogeboom et al.
  2023). **[E]** It multiplies the x_0-space loss. This equals Kingma & Gao's ε-space sigmoid weighting sigmoid(b − λ) after converting
  ‖ε − ε̂‖² = SNR · ‖x_0 − x̂_0‖² (our derivation), so the direction (small weight at high noise) is consistent. **[E: derivation]**
- **The bias b for 2D is not stated** (images: swept over {−2, −1, 0, 1, 2}, App. I.2 p. 34). **[E: absent]** → b = 0 (also R's default). **[G]**
- Under b = 0 and safety 0.01: w(0.01) ≈ 1.0, w(0.5) = 0.5, w(0.99) ≈ 1.0e−4 (computed from the formula).

### 1.9 App. I.1: 2D architecture and training (pp. 32–33)
Quoted settings **[E]**: dataset of 102,400 points; 100k steps; batch 128; Adam lr 1e-3, b1 0.9, b2 0.999, eps 1e-8; cosine warmup for the
first 100 iterations; "clip the updates by their global norm (with the maximal norm being 1)"; EMA decay 0.99; flow-matching schedule;
safety epsilon 1e-2; sigmoid weighting; m = 32; β = 0.1, λ = 1 (distributional); β = 2, λ = 0 (diffusion-like).

Architecture **[E]**, with our reading:
- Time: sinusoidal embedding of dimension 2048, then a 2-layer MLP, hidden 2048, output 2048, GELU.
- "9-layers MLP with gelu, hidden dimension 64": 4 preprocessing layers applied to the time embedding, and **separately** 4 to x_t; concatenate on
  the last dimension; 4 MLP layers; one output layer. 4 + 4 + 1 = 9 counts the two parallel preprocessing stacks once. **[G: layer counting]**
- Distributional model: ξ has the dimension of x_t and is **concatenated with x_t** on the last dimension, i.e. **at the input of the x_t
  preprocessing branch** ("the noise ξ has the same dimensionality as x_t and we concatenate the two"). **[E]**
- "After passing it through the MLP, which produces a vector of dimension 4, we **ignore the first 2 dimensions**": the output layer has width 4
  (= dim [x_t, ξ]) and x̂ = out[..., 2:4]. For the standard model the output layer has width 2. **[E]** (The image models do the same: output 2c
  channels, drop half, App. I.2.)
- Not stated **[E: absent]**: whether a GELU follows the last preprocessing layer before the concatenation (we put GELU after every hidden
  layer, none after the output layer) **[G]**; the sinusoidal embedding's max period and sin/cos order (we use 10000, [sin, cos]) **[G]**;
  initialization (PyTorch default) **[G]**; whether the time value fed to the embedding is t or 1000·t (we use 1000·t, since frequencies up to
  1 over t ∈ [0, 1] with max period 10000 would make most of the 2048 features nearly constant) **[G]**.
- Clipping: "clip the updates by their global norm". In optax, `chain(clip_by_global_norm(1), adam(...))` clips the gradients that enter Adam;
  clipping Adam's output is also possible. We clip **gradients before Adam** (`clip_grad_norm_`). **[G]**
- Warmup: "cosine warmup for first 100 iterations": lr_k = 1e-3 · ½(1 − cos(π k/100)) for k < 100, then constant 1e-3 (no decay mentioned). **[G: exact shape]**
- Data order: how batches are drawn from the fixed 102,400-point dataset is not stated. We shuffle each epoch (800 steps/epoch). **[G]**

### 1.10 Figure 3 and Figure 4 protocol (§6.1 p. 7; App. I.1 p. 33)
- **Fig. 3 left:** squared MMD (§1.4) between 4096 generated samples and 4096 target samples, versus the number of sampling steps. The x-ticks in
  the source figure `figures/mmd_2d.pdf` are **5, 10, 20, 30, 50, 70, 90, 100** (text extracted from the vector PDF). y-axis 0 to 0.016.
  Curves: x̂_θ(β=0.1, λ=1), x̂_θ(β=2, λ=0), "Diffusion model". **[E]**
- **Fig. 3 right:** for 4096 pairs (X_0^i, X_t^i), 8 draws of ξ, the std of x̂_θ(t, x_t^i, ξ) over ξ, averaged over i, versus t ∈ [0, 1]; the
  same with 8 draws from the exact posterior. y-axis 0 to 1.75. How the std of a 2-vector is reduced to a scalar is **not stated** **[E: absent]**.
  We take the per-coordinate std (ddof 0, numpy/jnp default) averaged over the 2 coordinates: at t → 1 the posterior is p_0, whose per-coordinate
  std averages (√9.25 + 0.5)/2 ≈ 1.77, consistent with the plot's y-range. **[G]**
- **Analytic posterior** (App. I.1): p(x_0|x_t) = Σ_k w_k N(ν_k, Σ), Σ = (α_t²/σ_t² + 1/σ²)^{−1} I, ν_k = Σ(α_t/σ_t² x_t + μ_k/σ²),
  w_k ∝ N(x_t; α_t μ_k, (α_t²σ² + σ_t²) I). (The paper writes w_k ∝ N(α_t μ_k, …) without the argument x_t.) **[E]**
- **Fig. 4:** x_0 ∈ {μ_1 = (3, 3), μ_2 = (−3, 3)} (rows); one x_t | x_0 per t; t ∈ {0.99, 0.88, 0.45, 0.23, 0.01} (panel titles of
  `figures/distributional_2d.png`); 4096 posterior samples (light blue), 4096 draws of x̂_θ(β=0.1, λ=1) (green), x̂_θ(β=2, λ=0) (red), x_0
  (orange ×), x_t (black ×). Same axis limits in all panels. **[E: figure + caption]** (An unused `distributional_2d_new.pdf` in the source has
  t ∈ {0.99, 0.66, 0.34, 0.01}; the paper uses the PNG.)
- Which run/seed/checkpoint the figures use: not stated (single run assumed). **[E: absent]**

### 1.11 Which sampler the standard-diffusion baseline uses
- 2D: not stated. Images: "we follow Algorithm 2 for distributional models and **SDE sampler** for classical diffusion models" (App. I.2, p. 34).
  App. G.1 shows Eq. (4) with ε = 1 is the discretization of the reverse SDE ("we recover the usual sampler derived from the stochastic process
  point of view"). So the "SDE sampler" is Algorithm 2 with X̂_0 = x̂_θ(t, x_t) (no ξ) and ε = 1. **[E + G]**
- NFE: one denoiser call per step for every arm (App. H: "same computational complexity during sampling as ordinary diffusion models"). **[E]**

### 1.12 App. K.1 and Table 2 (pp. 35–36)
- "unconditional standard diffusion model and our distributional models on … checkerboard, see Galashov et al. (2025)". Reports "MMD between
  sampled and target distributions with different NFEs". **[E]**
- Table 2 values (paper, not ours): NFE 5: 7.00e-3 / 1.57e-4; 10: 2.10e-3 / 1.96e-4; 50: 2.70e-4 / 1.55e-4; 100: 1.96e-4 / 2.00e-4;
  1000: 1.71e-4 / 1.83e-4 (Diffusion / Distributional). **[E]**
- **Not stated** **[E: absent]**: the generator and scale, dataset size, training settings, β/λ/m of the distributional model, whether "MMD" is
  MMD or MMD², kernel, bandwidth, sample count, estimator, seeds. Only two arms (no β=2, λ=0 control).
- Observation: the distributional values are flat at ≈ 1.6–2.0e-4 from NFE 5 to 1000, and the diffusion values converge to the same level. A
  floor of that size is what a V-statistic (biased) MMD² would show from finite samples alone: with n = m samples it adds
  (1/n)(1 − Ē_{pp}k) + (1/m)(1 − Ē_{qq}k). This is a hypothesis for S4 to test, not a finding. **[G]**

---

## 2. Reference code R (`edluyuan/ddm` @ 8a43f77)

What it does: `run_example.py` trains a distributional MLP on fresh GMM samples, samples with Algorithm 2 (20 steps), and reports an RBF MMD².
It is a third-party re-implementation; the authors (Google DeepMind, JAX) released no code that we could find (GitHub search 2026-11-19 for the
paper title and for Galashov et al.: only R and an unrelated `zahramh99/dmmd-mmd-ddm-gen`).

| Item | R (file:line) | Paper | Consequence for us |
|---|---|---|---|
| Model | `DDDMMLP` (`dddm/model.py`): input [x_t, ξ, 32 Fourier time feats] → 4 × Linear(128)+**SiLU** → Linear(2) | App. I.1: 2048-d sinusoidal + 2-layer MLP, two branches × 4 layers, 4 joint, out 4 → last 2, GELU, width 64 | Not used. We build App. I.1. |
| Standard-diffusion arm | **absent** (only the distributional model) | §6.1: Eq. (6) baseline | We implement arm A ourselves. |
| Bridge | `schedules.py` `gaussian_bridge_mu_sigma`: r12 = (α_t/α_s)(σ_s/σ_t)², r01 = σ_s/σ_t, r11 = (α_t/α_s)(σ_s/σ_t); **r22 = (α_t/α_s)(σ_s/σ_t)²** | r_{2,2} = (α_t/α_s)²(σ_s/σ_t)² (App. G form) | **Bug in R** for ε > 0: r22 has α-ratio to the power 1, not 2. ε = 1 then does not give the DDPM posterior mean. R uses the App. G (σ_s/σ_t)^j form, as we do. 1e-8 added to denominators. |
| Loss | `losses.py` `generalized_energy_terms`: conf = mean over (B, m); inter = mean over B·m·(m−1) off-diagonal pairs; `training.py`: loss = w̄ · (conf − **λ/(2(m−1))** · inter) | (14): conf − (λ/2) · (mean over pairs) | **Bug in R:** the (m−1) is divided twice, so the effective λ is λ/(m−1) (1/7 at R's m = 8). |
| Weighting | `w̄ = sigmoid_weight(t).mean()` multiplies the batch-mean loss | w_{t_i} per data point inside the sum | R decouples weight and sample; we weight per data point. Formula itself matches (b = 0 default). |
| ‖·‖^β | (‖d‖² + 1e-12)^{β/2} for β ≠ 2; exact square for β = 2 | not stated | We use the same δ = 1e-12 smoothing (documented). |
| Training | Adam lr 2e-3, batch 512, m = 8, 10k "epochs" (= steps) by default; `toy_gmm.yaml`: 5000 steps, seed 42; **no warmup, no clipping, no EMA**; t ~ U[0, 1], no safety ε | lr 1e-3, batch 128, m = 32, 100k steps, warmup, clip 1, EMA 0.99, safety 1e-2 | Not used. |
| Data | `sample_gmm`: fresh samples each step, no fixed dataset | fixed 102,400-point dataset | We cache a fixed dataset. |
| Sampler | `sampling.py`: grid `linspace(0, 1, N+1)`, X_1 ~ N(0, I), N calls to the model, ε = 1 default | Alg. 2 | Same structure; no safety interval in R. |
| MMD | `metrics.py` `rbf_mmd2`: **unbiased** MMD² (diagonals removed), σ = 1, Gretton convention (no ½) | D_ρ (½ MMD²?), estimator not stated | R picks unbiased; we report unbiased as primary and also biased. |
| Time embedding | `SinusoidalTimeEmbedding` (unused by the toy MLP) uses max period 10000, [sin, cos] | — | Same convention as our choice. |

## 3. FFJORD `lib/toy_data.py` (@ 994864a), `inf_train_gen("checkerboard")`
```
x1 = U[0,1)*4 - 2 ; x2_ = U[0,1) - randint{0,1}*2 ; x2 = x2_ + (floor(x1) % 2) ; return [x1, x2] * 2
```
- x1 ∈ [−2, 2); x2 ∈ [−2, 2); then scaled by 2 → support in [−4, 4]², a 4 × 4 board of 2 × 2 squares, 8 occupied squares, uniform density 1/32.
  A point is in the board iff (⌊x1/2⌋ + ⌊x2/2⌋) is even, on the scaled coordinates (our unit test checks this rule). **[E: code + test]**
- Uses the global `np.random` state and ignores its `rng` argument for this dataset. We re-implement it with an explicit `numpy.random.Generator`.

## 4. Galashov et al. (G), the checkerboard reference
- §6.1 / App. "Toy 2-D datasets" of arXiv v1 (`main.tex` l. 1352, 2149–2152): "a simple checkerboard 2-D dataset"; DMMD trained 50k steps, batch 256,
  128 noise levels per batch, DDPM forward process with modified β, t ∈ [0.05, 1.0], sampling with T = 100 steps.
- **Not specified in G:** the checkerboard generator, its scaling, the dataset size, and any MMD metric used for evaluation (G evaluates the
  2D experiments only qualitatively, plus MMD²(P_t, P; t) of its learned kernels). G is a different method (MMD particle flow), so none of its
  training settings transfer to the DDM checkerboard run. No code released (search above).
- Conclusion: the checkerboard of Table 2 cannot be pinned down. We use FFJORD's generator as a documented approximation and borrow the GMM
  training configuration (both labelled in `paper_vs_implementation.md`).
