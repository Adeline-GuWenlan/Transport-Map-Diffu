# Paper vs. implementation

Status: **verified** = value stated in the paper and our code matches it (checked by a test or by reading the code);
**matched** = stated in the paper, implemented as stated, not separately tested; **approximated** = not (fully) stated, we chose a value or
borrowed one; **unknown** = not stated and not determinable. P = arXiv 2502.02483 v3 PDF page; tex = `main.tex` line. Details: `audit.md`.

| # | Setting | Paper | Ours | Status | Source |
|---|---|---|---|---|---|
| 1 | GMM target | 0.5 N((3,3), 0.5² I) + 0.5 N((−3,3), 0.5² I) | same | verified (test) | §6.1 P7, App. I.1 P32 |
| 2 | GMM training set | 102,400 points | 102,400, data seed 0 | matched | App. I.1 P32 |
| 3 | Data order | not stated | per-epoch shuffle | approximated | — |
| 4 | Checkerboard generator | "see Galashov et al. (2025)"; not specified there | FFJORD `inf_train_gen("checkerboard")` (×2 scale, [−4,4]²) | approximated (S4: G's sample figure has the same 4×4 layout and parity as FFJORD, but no axis ticks, so the scale is unknown; an exact-posterior oracle on ×0.5 and ×0.25 boards does not move the diffusion baseline to the paper's values, `results_cb.md` §5.1) | App. K.1 P35; G App. (tex 2149), G `figures/2d_grad_flow_toy_samples.png`; F `toy_data.py` l.106–110 |
| 5 | Checkerboard dataset size | not stated | 102,400 (borrowed from GMM) | approximated | — |
| 6 | Checkerboard training config | not stated | GMM App. I.1 config (borrowed) | approximated | — |
| 7 | Checkerboard DDM β, λ, m | not stated | 0.1, 1, 32 (borrowed from GMM) | approximated | — |
| 8 | Schedule | α_t = 1 − t, σ_t = t | same | verified (test) | Eq. 3 P2, App. I.1 |
| 9 | Bridge Eq. (4) | μ, Σ with r_{i,j}; main text (σ_s²/σ_t²)^j vs App. G (σ_s/σ_t)^j | App. G form (the only one that preserves marginals) | verified (test: marginals, ε = 0 DDIM, ε = 1 DDPM) | Eq. 4 P2; App. G tex 1475–1480 |
| 10 | Churn ε (2D) | not stated | 1.0, all arms | approximated | §4.1 P4 uses ε = 1 |
| 11 | Safety ε | 1e-2; "time interval [η, 1 − η]" | training t ~ U[0.01, 0.99]; sampler calls the denoiser at clamp(t, 0.01, 0.99); for NFE > 100 the low clamp biases the sampler (oracle test), so `clamp_low` is an option to report both ways | approximated (exact use not stated) | App. I.1 P33; App. K.2 P39 |
| 12 | Training time distribution | U[0, 1] (Alg. 1) | U[0.01, 0.99] | approximated | Alg. 1 P4 |
| 13 | Loss weighting | sigmoid, w_t = (1 + exp[b − log(α²/σ²)])^{−1} | same formula, per data point, all arms | matched | §4.2 P5, App. I.1 |
| 14 | Sigmoid bias b | not stated for 2D (images: sweep {−2..2}) | 0 | approximated | App. I.2 P34 |
| 15 | Standard loss | Eq. (6), x_0-prediction | same | verified (test) | Eq. 6 P2 |
| 16 | Energy loss | Eq. (14), i = j excluded, λ/(2(m−1)) | same; checked vs double loop | verified (test) | Eq. 14 P3, App. F P30 |
| 17 | ‖·‖^β at 0 | not stated | (‖d‖² + 1e-12)^{β/2} for β ≠ 2 | approximated | R `losses.py` uses the same |
| 18 | m | 32 | 32 (arms B, C) | matched | App. I.1 |
| 19 | β, λ (C) | 0.1, 1 | same | matched | App. I.1 |
| 20 | β, λ (B) | 2, 0 | same | matched | App. I.1 |
| 21 | m for arm B | not stated (Alg. 1) | 32 | approximated | — |
| 22 | Time embedding | sinusoidal 2048 → 2-layer MLP (2048, 2048), GELU | same; input 1000·t, max period 10000, [sin, cos] | matched (embedding details approximated) | App. I.1 P33 |
| 23 | Backbone | 9-layer MLP, width 64, GELU: 4 preprocessing layers on t-emb and separately on x_t, concat, 4 layers, output layer | same (4 + 4 + 1 counting the parallel branches once) | matched (layer counting approximated) | App. I.1 P33 |
| 24 | ξ input | same dim as x_t, concatenated with x_t | concat at the x-branch input | matched | App. I.1 P33 |
| 25 | DDM output | width 4, ignore the first 2 dims | out[..., 2:4] | verified (test) | App. I.1 P33 |
| 26 | Arm A backbone | "baseline unconditional diffusion model" | same MLP, no ξ, output width 2 | approximated (decision in `plan.md`) | §6.1 P7 |
| 27 | Steps / batch | 100k / 128 | same | matched | App. I.1 |
| 28 | Optimizer | Adam lr 1e-3, (0.9, 0.999), eps 1e-8 | same | matched | App. I.1 |
| 29 | Warmup | cosine, first 100 iterations | ½(1 − cos(π(k+1)/100)), then constant | approximated (exact shape) | App. I.1 |
| 30 | Clipping | "clip the updates by their global norm", max 1 | clip gradients (before Adam) to norm 1 | approximated | App. I.1 |
| 31 | EMA | 0.99 | 0.99 every step, eval with EMA | matched | App. I.1 |
| 32 | DDM sampler | Algorithm 2 | same, uniform grid t_k = k/N | verified (test: NFE, Gaussian target) | Alg. 2 P4; App. K.2 "uniform" |
| 33 | Standard sampler (2D) | not stated; images: "SDE sampler" | Eq. (4) bridge, ε = 1, deterministic x̂ | approximated | App. I.2 P34; App. G.1 |
| 34 | NFE | 1 denoiser call per step | same, counted by the sampler | verified (test) | App. H |
| 35 | Fig. 3 metric | "MMD squared given by D_ρ", RBF σ = 1, 4096 samples | standard MMD²_u (primary), MMD²_b, D_ρ = ½MMD² all stored | approximated (normalisation and estimator unknown) | §6.1 P7, App. I.1 P33 |
| 36 | Fig. 3 left NFE grid | x-ticks 5, 10, 20, 30, 50, 70, 90, 100 | same | matched | source `figures/mmd_2d.pdf` |
| 37 | Fig. 3 right std | 8 draws, std over ξ, average over 4096 x_t | per-coordinate std (ddof 0), mean over coords (same for exact posterior and models) | approximated (vector→scalar reduction not stated; S3: the paper's exact-posterior curve matches "mean distance to the 8-draw mean" at t ≤ 0.45 but no single reduction fits all its curves, `docs/results_gmm.md` §4) | App. I.1 P33 |
| 38 | Analytic posterior | mixture formula App. I.1 | same; tested vs importance sampling | verified (test) | App. I.1 P33 |
| 39 | Fig. 4 | x_0 ∈ {μ_1, μ_2}, t ∈ {0.99, 0.88, 0.45, 0.23, 0.01}, 4096 samples, one x_t per panel; caption: top row = left Gaussian | same, but rows in the task-spec order (top = (3, 3) = right Gaussian), each row labelled | matched (row order swapped vs the paper, by the binding spec) | Fig. 4 caption (tex l. 525), `docs/task_spec.md` §6 |
| 40 | Table 2 metric | "MMD", nothing else stated (v3 only; v1/v2 have no Table 2; no code; OpenReview not readable from the cluster) | RBF σ = 1, n = m = 4096, unbiased MMD²_u primary; MMD²_b, σ ∈ {0.5, 2} and n = 10,000 also reported; every row marked "unverified" | unknown (S4: biased MMD² at n = 4096 is ruled out by its null floor 4.7e-4 > the paper's values; biased MMD² at n ≈ 10,000 is consistent with the paper's flat ≈ 1.7e-4 level; `audit.md` §5) | App. K.1 P35, tex l. 1795–1811 |
| 41 | Seeds | not stated (single run) | seeds 0, 1, 2 if budget allows; seed 0 is the primary run | approximated | — |
| 42 | Hardware | TPU/GPU (JAX) | CPU, PyTorch 2.14.1 float32 | approximated | `_rules.md` GPU blocker |
| 43 | Fig. 3 right t grid | not stated in the text | `linspace(0.01, 0.99, 10)` (+ a 50-point grid for the plot) | verified (vertices of the paper's vector figure) | `figures/std_over_eps_avg_over_xt.pdf`, `docs/paper_values/fig3_paper_values.json` |
| 44 | Fig. 3 left evaluation repeats | one curve, presumably one evaluation of 4096 samples | 10 repeats (sampler seed + reference set) per seed, 3 training seeds; mean reported, all repeats in the CSV | approximated (extra repeats reduce MC noise; single-evaluation noise ≈ ±2–4e-4 is reported) | `docs/results_gmm.md` §3 |
| 45 | Fig. 3 / Fig. 4 checkpoint | not stated | EMA weights of the 100k-step `final.pt` for every arm and seed | approximated | — |
| 46 | Table 2 NFE / arms | NFE 5, 10, 50, 100, 1000; Diffusion and Distributional | the same NFE (plus 20, 30, 70, 200, 500 for the curve); arms A, C and the extra control B | matched | App. K.1 Table 2 |
| 47 | Table 2 sampler | not stated (presumably as for the GMM) | the GMM protocol: Eq. (4) bridge, churn 1, uniform grid, denoiser time clamped to [0.01, 0.99], EMA weights, same seeds for every arm | approximated | `results_cb.md` §1 |
| 48 | NFE 1000 and the safety clamp | not stated | main table: `clamp_low=True` (grid times < 0.01 fed as 0.01); also `clamp_low=False` and a safety-interval grid ([0] + linspace(0.01, 0.99, N)) as diagnostics, identical for every arm | approximated | `results_cb.md` §5.3, `samplers.py` docstring |
| 49 | Table 2 evaluation repeats / seeds | one value per cell (single run presumably) | 3 training seeds × 10 evaluation repeats; per-seed means and the spread reported; paper values in their own column/file | approximated | `results_cb.md` §2, §6 |
