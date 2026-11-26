# E3 plan: CIFAR-10 trial (ddm2dx X2, 2026-11-26)

Spec: `/scratch/wg2381/claude_jobs/ddm2dx/prompts/_task.md`, E3 (and E4 for `cifar_B_pH`, `cifar_C_dino`). Short trial, one seed each:
does the method train on CIFAR-10 at all, and does β (or ρ) change anything? Not a replication of the paper's 1e6-step numbers.

## Runs
All configs are in `cifar/configs/`. They are identical except the columns below (checked by diff).

| run | arm | target | m | ρ | β | λ | feature term | priority |
|---|---|---|---|---|---|---|---|---|
| `cifar_B` | B | x_0 | 4 | ‖d‖^β | 2 | 0 | – | 1 |
| `cifar_C` | C | x_0 | 4 | ‖d‖^β | 0.1 | 1 | – | 2 |
| `cifar_A` | A | velocity x_0 − x_1 | 1 | – | – | – | – | 3 |
| `cifar_B_b1` | B | x_0 | 4 | ‖d‖^β | 1 | 0 | – | 4 |
| `cifar_B_pH` | B | x_0 | 4 | √(‖d‖² + c²) − c, c = 0.00054·√3072 = 0.02993 | – | 0 | – | 5 |
| `cifar_C_dino` | C | x_0 | 4 | ‖d‖^β | 0.1 | 1 | + λ_f · ES(φ(x̂), φ(x_0)), φ = DINOv2 ViT-S/14 CLS, β_f = 1, λ = 1 | 6 |

Shared settings: U-Net of App. I.2 (copied from transport_map `src/tmap/unet.py` 4b9e8f9, 140.0 M params), class-conditional;
Adam lr 1e-4 (0.9, 0.999, 1e-8), clip 1, cosine warmup 100 then constant; batch 128 data points (B/C: 512 network evaluations);
30k steps; α = 1 − t, σ = t, t ~ U[0.01, 0.99]; sigmoid weighting bias 0 for every arm; bf16 autocast (losses in fp32); hflip;
dropout 0; EMA 0.9999 with warm-up min(0.9999, (1+k)/(10+k)); seed 0 for every run (same images, flips, t, X_1 at every step).

## Our choices where paper or spec leave room (the same for every run)
| choice | value | why |
|---|---|---|
| weighting | sigmoid, bias 0 | the 2D runs use it; the paper sweeps bias ∈ {−2..2} or none |
| A's loss | w_t ‖v̂ − v‖² (sum over pixels) | velocity target as the paper's diffusion baseline; same w_t and the same squared-norm scale as B at β = 2 |
| last conv init | PyTorch default (not zero) | with a zero last conv the B/C particles would all be equal at step 0; ddm2d's 2D MLP also uses default init |
| time input | sinusoidal(1000 t) | as ddm2d's 2D model |
| EMA warm-up | min(0.9999, (1+k)/(10+k)) | without it 37 % of the 10k-step EMA would be the random init |
| safety | 0.01 (train t range, sampler clamp) | ddm2d's value |
| λ_f (C_dino) | weight 1 × s, s = mean ‖y−y′‖^0.1 / mean ‖φ(y)−φ(y′)‖^1 over same-class pairs of 200 train images per class | transport_map's `scale: auto` rule, adapted to the pixel β = 0.1 [G] |
| feature weighting | w_t on the feature term too | one weight per data point |

## Evaluation protocol (identical for every run; run at the end of each training job by `scripts/train_gpu.slurm`)
- FID-10k, clean-fid `legacy_tensorflow` vs CIFAR-10 train stats (bundled file = transport_map's, sha256 checked in every JSON),
  labels i mod 10 (1000 per class), sampler seed 0, EMA weights.
- Checkpoint selection: EMA at 10k and 20k, FID-10k at NFE 10 **and** 100 for every arm (the spec asks NFE 10 for B/C and 100 for A;
  doing both for all arms keeps the table symmetric).
- Final (30k) EMA: FID-10k at NFE {2, 4, 8, 10, 15, 30, 100}; 10 × 10 grids (row = class) at NFE 10 and 100.
- B/C runs: per-pixel std over 8 ξ draws of x̂(t, x_t, ξ), 256 test images, t ∈ {0.02, 0.05, 0.1, …, 0.9, 0.98}.
- Sampler: Algorithm 2 (Eq. (4) bridge, churn 1), uniform grid, one network call per NFE; A uses x̂ = x_t + t v̂.

## Cost estimate and budget (≤ 75 GPU-h for the chain)
Prior (before the bench), from transport_map's UNet bench (`claude_jobs/standups/unet_S1.md`: forward+backward over 512 images ≈ 1.0 s,
30.6 GiB on an L40S): B / C ≈ 1 s/step → 8.3 h per 30k; A ≈ 0.3 s/step → 2.5 h; C_dino ≈ 2 s/step → 17 h. Evaluation ≈ 0.5–1 h per run.
Total ≈ 55–60 GPU-h. **The bench (`bench_<jobid>/bench.json`) replaces these numbers; see the section below once it is in.**

## Bench result and final plan
(pending: bench job 24750418)
