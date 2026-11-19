# Paper vs. implementation

Status: **verified** = value stated in the paper and our code matches it (checked by a test or by reading the code);
**matched** = stated in the paper, implemented as stated, not separately tested; **approximated** = not (fully) stated, we chose a value or
borrowed one; **unknown** = not stated and not determinable. P = arXiv 2502.02483 v3 PDF page; tex = `main.tex` line. Details: `audit.md`.

| # | Setting | Paper | Ours | Status | Source |
|---|---|---|---|---|---|
| 1 | GMM target | 0.5 N((3,3), 0.5² I) + 0.5 N((−3,3), 0.5² I) | same | verified (test) | §6.1 P7, App. I.1 P32 |
| 2 | GMM training set | 102,400 points | 102,400, data seed 0 | matched | App. I.1 P32 |
| 3 | Data order | not stated | per-epoch shuffle | approximated | — |
| 4 | Checkerboard generator | "see Galashov et al. (2025)"; not specified there | FFJORD `inf_train_gen("checkerboard")` (×2 scale, [−4,4]²) | approximated | App. K.1 P35; G App. (tex 2149); F `toy_data.py` l.106–110 |
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
| 37 | Fig. 3 right std | 8 draws, std over ξ, average over 4096 x_t | per-coordinate std (ddof 0), mean over coords | approximated (vector→scalar reduction) | App. I.1 P33 |
| 38 | Analytic posterior | mixture formula App. I.1 | same; tested vs importance sampling | verified (test) | App. I.1 P33 |
| 39 | Fig. 4 | x_0 ∈ {μ_1, μ_2}, t ∈ {0.99, 0.88, 0.45, 0.23, 0.01}, 4096 samples, one x_t per panel | same | matched | Fig. 4 P7, App. I.1 |
| 40 | Table 2 metric | "MMD", nothing else stated | same estimator family as Fig. 3, reported with its definition; numeric agreement unverified | unknown | App. K.1 P35 |
| 41 | Seeds | not stated (single run) | seeds 0, 1, 2 if budget allows; seed 0 is the primary run | approximated | — |
| 42 | Hardware | TPU/GPU (JAX) | CPU, PyTorch 2.14.1 float32 | approximated | `_rules.md` GPU blocker |
