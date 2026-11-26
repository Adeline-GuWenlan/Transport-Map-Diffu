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

## Bench result (job 24750418, L40S 46 GB, 20 min = 0.34 GPU-h; `bench_24750418/bench.json`) [E]
Full-size model, batch 128, 5 warm-up + 20 timed steps, real data, bf16.

| case | grad_chunk | s/step | peak allocated GiB |
|---|---|---|---|
| A | 0 (one pass) | 0.250 | 9.5 |
| B (β = 2) | 0 | 0.992 | 30.6 |
| B (β = 2) | 256 | 1.292 | 18.1 |
| C (β = 0.1, λ = 1) | 0 | 0.998 | 31.7 |
| B_pH | 0 | 1.008 | 31.7 |
| C_dino | 0 | OOM (> 43.5) | – |
| C_dino | 256 | 1.816 | 25.0 |
| C_dino | 128 | 1.672 | 14.6 |

FID-10k (sampling 10k images in batches of 1000 + clean-fid call): NFE 10: 70.7 + 19.4 = 90 s; NFE 100: 669.5 + 15.8 = 685 s.
So sampling costs ≈ 6.7 s per NFE per 10k images, and one FID call ≈ 18 s. λ_f for C_dino from the `auto` rule: s = 0.02547
(pixel mean ‖y − y′‖^0.1 = 1.431, DINOv2 mean ‖φ − φ′‖ = 56.2).

## Final plan (no cuts)
- grad_chunk: 0 for A, B, B_b1, B_pH, C; **128 for C_dino** (exact two-pass gradient; one pass does not fit). grad_chunk only changes
  memory, not the loss or its gradient (tests/test_cifar.py::test_chunked_gradient_exact). It is the one config field, besides the
  intended ones, that differs between runs.
- Steps: 30k for every run (no reduction needed).
- Evaluation per run: snapshots 10k, 20k at NFE 10 + 100: 2 × (90 + 685) s = 1550 s; final at NFE {2, 4, 8, 10, 15, 30, 100}: 169 × 6.7 s
  + 7 × 18 s = 1257 s; ξ-spread ≈ 20 s. Total ≈ 0.79 GPU-h per run.

| run | s/step (bench) | train GPU-h | eval GPU-h | + overhead | total | `--time` (×1.25) |
|---|---|---|---|---|---|---|
| cifar_A | 0.250 | 2.09 | 0.79 | 0.1 | 2.98 | 03:45:00 |
| cifar_B | 0.992 | 8.27 | 0.79 | 0.1 | 9.16 | 11:40:00 |
| cifar_B_b1 | 0.992 (as B) | 8.27 | 0.79 | 0.1 | 9.16 | 11:40:00 |
| cifar_B_pH | 1.008 | 8.40 | 0.79 | 0.1 | 9.29 | 11:40:00 |
| cifar_C | 0.998 | 8.32 | 0.79 | 0.1 | 9.21 | 11:40:00 |
| cifar_C_dino | 1.672 | 13.93 | 0.79 | 0.1 | 14.82 | 18:30:00 |
| **sum** | | 49.3 | 4.7 | 0.6 | **54.6** | |

With the bench (0.34) the chain projects **≈ 55 GPU-h of 75**, leaving ≈ 20 GPU-h for X4 (relaunches, at most two eval-only jobs).
Overhead = checkpoint writes (latest.pt ≈ 2.2 GB every 1000 steps) and start-up [G]. The jobs are requeue-safe, so `--time` only
bounds a single segment.
