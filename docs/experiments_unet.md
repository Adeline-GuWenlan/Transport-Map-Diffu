# UNet vs DiT backbone: one-step energy-score transport map (unet chain, started 2026-11-20)

Goal: train a UNet exactly like the finished DiT-B/2 Experiments A/B (`docs/experiments_AB.md`), then compare.
The task and fairness contract are in `claude_jobs/unet/prompts/_task.md`. Labels: **[E]** = evidence in a committed
file, **[G]** = guess or assumption.

## Source: De Bortoli et al. 2025, App. I.2
"Distributional Diffusion Models with Scoring Rules", ICML 2025, https://arxiv.org/abs/2502.02483 (v3 PDF). It is the
published energy-score model on pixel CIFAR-10 with a U-Net. Checked in the v3 PDF on 2026-11-20:
- **App. I.2 "Image space experiments", paragraph "Architecture.", p. 33.** The paragraph sets the channel width
  (256), the multipliers (1, 2, 2, 2), 2 residual blocks per level, and attention at the second level ("resolution 16
  in the case of CIFAR-10") and in the bottleneck. A residual block is "normalizing the input, applying a 3 × 3
  convolution block and applying a non-linearity. This is then followed by a dropout layer and a 3 × 3 convolution
  block. Finally we add the input (processed through a 1 × 1 convolutional block)". The paragraph also says:
  RMSnorm, GELU, attention "after the convolutional residual block", 8 heads, "sinusoidal embedding followed by a MLP
  layer" for time, an "embedding matrix" for the CIFAR-10 class, the two conditionings summed, and "adaptive RMSnorm
  ... the scale and bias of the normalization layers are obtained by passing the conditioning vector through a MLP
  layer". Their input is [x_t, ξ] (2c channels), and half of the 2c output channels are dropped.
- **App. I.2 "Training and evaluation details.", p. 34:** 1e6 steps, batch 256 (CIFAR-10), Adam, clip norm 1, lr 1e-4,
  cosine warmup over the first 100 iterations, population size m = 4. No dropout rate, EMA or parameter count.
- **App. J "Computational complexity", Table 1, p. 35:** steps/s on CIFAR-10. For example, H100 at batch 64: diffusion
  14.3, distributional m = 4: 4.15. These are paper numbers (JAX); they are not comparable with ours.
- The abstract page links no code. [E: arXiv abstract page, 2026-11-20]

What is ours, not the paper's: the one-step setting (input z only, 3 channels, constant t = 999), and the A/B training
recipe (AdamW, betas (0.9, 0.95), 5k warmup, clip 2, EMA 0.9999, N = 512). Both are taken unchanged from the DiT runs.

## Implementation: `src/tmap/unet.py`
Written from the paper text. No third-party U-Net code was read or cloned. The down/up layout follows the common DDPM
U-Net (Ho et al. 2020) from memory. `build_model` dispatches on `model.backbone: unet`
(`src/tmap/model.py`, additive). `UNetTransportMap` has the `TransportMap` interface: `forward(z, c)`, `sample`,
`in_channels`, `out_channels`, `input_size`, `num_classes`. Checkpoints already store the full config, so
`tmap.evaluate` and `tmap.sample` rebuild either backbone from it [E: `tests/test_unet.py::test_checkpoint_round_trip`].

### Design decisions where App. I.2 is silent
| item | choice | why |
|---|---|---|
| downsampling | 3×3 conv, stride 2, after each level but the last | DDPM default (`conv_resample`) [G] |
| upsampling | nearest ×2, then 3×3 conv | DDPM default [G] |
| skip connections | every down-path block output is kept; each up-path block takes concat(h, skip); 2 blocks per level on both paths (8 skips) | the paper says "2 residual blocks" per level; DDPM's third up block (which also eats the downsample skips) is dropped to keep 2 [G] |
| channels per level | 256 / 512 / 512 / 512 at 32 / 16 / 8 / 4 px | App. I.2 multipliers |
| attention placement | after each residual block at 16 px, on both paths (2 down + 2 up), plus the bottleneck | paper: "second level" and "after the convolutional residual block" [G: the up path too] |
| bottleneck | ResBlock → attention → ResBlock | DDPM middle block [G] |
| norm in the attention block | pre-norm: x + proj(MHSA(AdaRMSNorm(x))); qkv and proj are linear layers with bias | [G] |
| RMSNorm axis | over channels at each spatial position, computed in fp32 | [G]; per-position channel norm, like RMSNorm on the last axis in NHWC (JAX) code |
| adaptive RMSNorm | x / rms(x) · (1 + γ(e)) + β(e), [γ, β] = Linear(GELU(e)); no separate learned gain | paper: "scale and bias ... MLP"; MLP = GELU + one linear [G] |
| norms per residual block | **one** adaptive norm, at the block input (as the paper describes); the second conv has none | follows the text literally |
| adaptive norm everywhere | yes: residual blocks, attention blocks and the output head | paper: "replacing the RMSnorm with an adaptive RMSnorm" |
| 1×1 skip conv | always, even when C_in = C_out | paper: "processed through a 1 × 1 convolutional block" |
| head | conv_in 3×3 (3 → 256); output AdaRMSNorm → GELU → 3×3 conv (256 → 3) | DDPM-style head [G] |
| output init | last conv zero (weights and bias), so F ≡ 0 at step 0 as for DiT's adaLN-Zero | fairness contract [G] |
| other inits | PyTorch defaults (incl. the modulation linears, so the class matters after one step) | [G] |
| embedding width | e ∈ R^1024 (4 × base channels); sinusoidal dim 256; time MLP Linear(256, 1024) → GELU → Linear(1024, 1024); class `nn.Embedding(10, 1024)`; summed | DDPM's 4× convention [G] |
| t | constant 999 (as the DiT map) | the one-step map |
| input / output | z, 3×32×32 → 3×32×32; no [x_t, ξ] concat | the one-step map |
| dropout | 0 | paper gives no rate; DiT uses none |
| `model.patch_size: 2` | not used by the UNet; it is the grid of `tmap.evaluate`'s blockiness metric, equal to DiT-B/2's patch, so the metric is comparable | avoids editing `evaluate.py` |

## Fairness checks
- **Configs** [E: `docs/unet_config_diff.txt`, `tests/test_unet.py::test_configs_differ_from_dit_only_in_model_name_and_grad_chunk`].
  `configs/expA_cifar10_unet_pixel.yaml` and `configs/expB_cifar10_unet_pixel-dinov2.yaml` differ from their DiT twins
  only in `name`, comments, the `model:` block and `train.grad_chunk`.
  - `grad_chunk`: **512 for A** (= N, so one pass) and **256 for B** (exact two-pass). DiT used 128 for both.
    This was S1's plan; the 4 runs actually used **256 for all** (S2, see "Training runs").
  - The gradient is exact either way [E: `tests/test_unet.py::test_chunked_two_pass_gradient_equals_one_pass`]. It
    changes speed only. With chunk = N, the UNet A run skips the extra forward pass that DiT A paid (about +1/3
    compute), so wall time is not a like-for-like backbone cost.
- **Same batches, flips and noise** [E: `runs/determinism_unet_S1.json` (seed 0), `runs/determinism_unet_S1_seed1.json` (seed 1),
  `scripts/determinism_check.py`]. The real `tmap.train.train()` ran 3 steps from the A config for both backbones (small
  models, `n_per_class` 4). For both seeds the logged sampler class choices, image indices, hflip masks, batch
  checksums, the z passed to `forward_backward` (sha256) and the fixed grid noise are identical across backbones.
  Reason: z comes from `zgen` (seed + 1) with shape (N, 3, 32, 32), the sampler has its own numpy RNG, and model
  init uses only the global torch RNG.
- **DiT unchanged** [E: `tests/test_dit_unchanged.py`]. Golden losses, gradient norms and parameter checksums after 3 real
  training steps of a tiny DiT (pixel loss, and a random-conv feature term for the B path), recorded on 8a15563
  before any edit. It still passes with the UNet code in place (`runs/pytest_unet_S1.log`: 35 passed).
- Not changed: optimizer, lr 1e-4, warmup, EMA, clip, seeds, eval seeds, `scripts/exp_gpu.slurm`.

## Size and cost (3×32×32) [E: `runs/cost_unet_S1.json`, `scripts/cost_unet.py`]
| | UNet (App. I.2) | DiT-B/2 | UNet / DiT |
|---|---|---|---|
| params | 140.0 M | 129.7 M | 1.08 |
| forward GMACs/sample (`torch.utils.flop_counter`) | 20.04 | 21.79 | 0.92 |
| saved activations/sample, bf16 autocast | 53.4 MiB | 110.2 MiB | 0.48 |
| + DINOv2-S/14 at 224 px (exp B) | 81.3 MiB | 138.1 MiB | |
| CPU s/step, batch 64, fp32, 7 threads (chunk 16) | 9.96 | 14.89 | 0.67 |
| `grad_chunk` by the 30 GiB rule, A / B | 512 / 256 | 256 / 128 | |

Notes:
- The params ratio is 1.08, inside the contract's 0.6×–1.6× band.
- DiT-B/2's config quotes 23.0 GMACs, the DiT paper's count. The same counter that gives 20.04 for the UNet gives
  21.79 for DiT-B/2, so compare the 20.04 against 21.79.
- **Memory method.** Bytes of the distinct storages packed for backward (`saved_tensors_hooks`) at batch 2 and 6,
  slope per sample. The slope removes per-call constants such as bf16 weight copies. The same method gives
  110.2 MiB for DiT-B/2, against 111 MiB in `docs/experiments_AB.md` "Memory". DINOv2 gives 27.9 MiB against about
  26 MiB there.
- **CPU timing.** It uses the exact chunked path at chunk 16: one pass of DiT-B/2 at batch 64 in fp32 was OOM-killed
  in the 16 GB CPU allocation.
- **Chunk rule.** Largest power of two with chunk × activations ≤ 30 GiB, capped at 512. A at 512 is about 26.7 GiB,
  plus about 2.8 GB of weights, EMA, gradients and AdamW state (140 M × 20 bytes). The GPU bench checks it.

## CPU smoke (not a result)
[E: `runs/smoke_unet_cpu/` (`command.txt`, `metrics.jsonl`, `eval.jsonl`, `samples/`)]. This is a code check only; it is never
compared with anything.
- Setup: `configs/smoke_cpu_cifar10.yaml` (CIFAR-10, batch 8 × 8, lr 2e-4, fp32, CPU, 7 threads) with a small UNet
  model block: base 64, multipliers (1, 2, 2), 1 block per level, attention at 16 px plus the bottleneck, 4 heads,
  emb 256. That gives 3.98 M params.
- Run: 1737 steps in 18.5 min of training, in two segments. Segment 1 was stopped by SIGTERM at step 897 (the agent
  tool's time limit), checkpointed, and resumed exactly.
- Training loss: 26.70 (mean of steps 1–50) → 18.66 (steps 1651–1700).
- Held-out pixel D1 (500/class): 17.80 at step 0 (F ≡ 0, the "zero" baseline) → 0.678 at 1000 → **0.621** at 1737
  (raw). EMA: 1.138. Raw diversity ratio: 1.015.
- Output: all finite. On the fixed grid noise, mean |x| = 0.45.

## GPU bench
Job 23951657, 1 × L40S (46068 MiB), gl042, 7.9 min, commit dc12830, `scripts/bench_gpu_unet.slurm`
[E: `runs/slurm-tmap_bench_unet-23951657.out`, `runs/bench_gpu_unet_23951657/`]. 60 steps each; the median s/step is
taken over steps 30–60. This is a bench result only, not a training result.

| UNet run | grad_chunk | s/step | peak mem | loss step 10 → 60 |
|---|---|---|---|---|
| A | **512** (one pass) | **1.012** | 30.6 GiB | 27.283 → 27.064 |
| A | 256 | 1.297 | 17.1 GiB | 27.283 → 27.064 |
| B | **256** | **1.906** | 23.9 GiB | 69.536 → 65.437 |
| B | 512 | OOM: 42.9 GiB allocated by PyTorch, out of 44.4 | – | – |

- The two A chunk sizes give the same losses, as the exact gradient should.
- At the first logged step, UNet A starts at the same loss as DiT A's bench (27.283). F ≡ 0 for both and the batch is
  the same.
- B: λ = 0.6595, the same scale as DiT B's runs (fixed per run from the first 200 train images per class).
- Evaluation path (`tmap.evaluate` at 100/class, FID-2k, Inception and DINOv2 D1, nn grid, then `tmap.sample` for EMA
  and raw): exit 0 in 57 s.
- **Chosen (configs unchanged):** `grad_chunk` 512 for A and 256 for B.

For calibration [E: `runs/slurm-tmap_bench-22605880.out`, `runs/expA_s*/metrics.jsonl`, `runs/expB_s*/metrics.jsonl`], the
DiT bench predicted the real DiT runs within 2%:

| DiT run | bench s/step | real run (median of `steps_per_s`) |
|---|---|---|
| A (chunk 128) | 1.463 | 1/0.673 = 1.486 |
| B (chunk 128) | 1.982 | 1/0.5046 = 1.982 |

### Projection per 30k-step run [G, from the measurements above]
- **Training:** 30000 × s/step × 1.03. The 3% covers startup, in-loop pixel D1 every 5k, checkpoints and one requeue.
  DiT A s0's measured overhead was 0.5%: 44817 s of training wall against 30000 / 0.673 = 44577 s.
- **Evaluation:** post-training evaluation of the 4 snapshots took 15.8–16.7 min in the DiT runs (`runs/slurm-tmap_*.out`:
  "tmap.train exited with 0" → "done"). It is dominated by FID-10k and feature D1, and should not depend much on the
  backbone. Assume 0.28 h.

| run | s/step | training | + eval | per run | × 2 seeds |
|---|---|---|---|---|---|
| UNet A | 1.012 | 8.69 h | 0.28 h | **8.97 GPU-h** | 17.9 |
| UNet B | 1.906 | 16.36 h | 0.28 h | **16.64 GPU-h** | 33.3 |
| total, 4 runs | | | | | **51.2 GPU-h** |

- Every run is under the 30 GPU-h stop rule, and the total is within the 120 GPU-h budget. GPU-h used by S1: 0.13
  (the bench).
- Suggested `--time` for S2: A 10:00:00 and B 18:00:00 (about 10% margin). `l40s_public` has no wall limit
  (`sinfo`: infinite). `scripts/exp_gpu.slurm` requeues safely if a limit is hit anyway.
- **Wall time is not a like-for-like backbone cost.** UNet A runs one pass, while DiT A used chunk 128 and paid the
  extra forward. At the same chunk (256), UNet A costs 1.297 s/step against DiT A's 1.494 s/step in the DiT bench.

## Training runs (S2, submitted 2026-11-21 09:18 EST)
Submitted with `scripts/exp_gpu.slurm` (unchanged), in the DiT pattern from `docs/experiments_AB.md`
[E: `claude_jobs/unet/jobids/S2.txt`]:

| run | job | config | seed | `TRAIN_SET` | `--time` |
|---|---|---|---|---|---|
| `runs/unetA_s0` | 23968737 | `configs/expA_cifar10_unet_pixel.yaml` | 0 | `seed=0 train.grad_chunk=256` | 14:00:00 |
| `runs/unetB_s0` | 23968738 | `configs/expB_cifar10_unet_pixel-dinov2.yaml` | 0 | `seed=0` | 20:00:00 |
| `runs/unetA_s1` | 23968739 | `configs/expA_cifar10_unet_pixel.yaml` | 1 | `seed=1 train.grad_chunk=256` | 14:00:00 |
| `runs/unetB_s1` | 23968740 | `configs/expB_cifar10_unet_pixel-dinov2.yaml` | 1 | `seed=1` | 20:00:00 |

```bash
CONFIG=configs/expA_cifar10_unet_pixel.yaml RUN=runs/unetA_s0 TRAIN_SET="seed=0 train.grad_chunk=256" \
  sbatch --job-name=tmap_uA_s0 --time=14:00:00 scripts/exp_gpu.slurm
CONFIG=configs/expB_cifar10_unet_pixel-dinov2.yaml RUN=runs/unetB_s0 TRAIN_SET="seed=0" \
  sbatch --job-name=tmap_uB_s0 --time=20:00:00 scripts/exp_gpu.slurm
# and the same for seed 1
```

**`grad_chunk` = 256 for all 4 runs.** The S2 prompt asks for one value for all 4 runs. B does not fit at 512 (bench OOM), so the value
is 256. The A config still says 512, so the A runs override it in `TRAIN_SET`. Each run's own `config.yaml` records the value used.
The gradient is exact at any chunk size, so this only changes speed. With it, UNet A runs two-pass like DiT A did (DiT used
chunk 128), so the A wall times are closer to like-for-like. The cost is about +2.2 GPU-h per A run.

**Projection at the chosen chunk** [G: bench s/step × 30000 × 1.03, plus 0.28 h for evaluation, same method as above]:

| run | bench s/step | training | + eval | per run | `--time` = ⌈1.2 × training + eval⌉ |
|---|---|---|---|---|---|
| UNet A (chunk 256) | 1.297 | 11.13 h | 0.28 h | 11.41 GPU-h | ⌈13.64⌉ = 14 h |
| UNet B (chunk 256) | 1.906 | 16.36 h | 0.28 h | 16.64 GPU-h | ⌈19.91⌉ = 20 h |
| total, 4 runs | | | | **56.1 GPU-h** | |

Every run is below the 30 GPU-h stop line, and the total is below the 120 GPU-h chain budget (S1 used 0.13).

### Health check (S2, 2026-11-21 14:14 EST)
[E: `runs/unet*/metrics.jsonl`, `runs/unet*/eval.jsonl`, `runs/slurm-tmap_u*-*.out`, snapshots committed with this section; `sacct`:
all 4 RUNNING, restart count 0.] The s/step is the median of `1/steps_per_s` from the 6th log line on. The finish time is
[G]: remaining steps at that speed, plus 0.28 h of evaluation.

| run | job | node | step | s/step (bench) | max_mem | loss at 1k → now | grad norm at 1k → now | held-out pixel D1 raw / EMA at 5k, 10k | projected finish (EST) |
|---|---|---|---|---|---|---|---|---|---|
| unetA_s0 | 23968737 | gl055 | 13400 | 1.308 (1.297) | 17.1 GiB | 18.475 → 18.347 | 46.2 → 8.3 | 0.132 / 8.991, 0.125 / 3.663 | ≈ 20:30 Nov 21 |
| unetA_s1 | 23968739 | gl035 | 13300 | 1.301 (1.297) | 17.1 GiB | 18.501 → 18.344 | 51.8 → 6.9 | 0.104 / 8.355, 0.116 / 3.021 | ≈ 20:30 Nov 21 |
| unetB_s0 | 23968738 | gl037 | 9000 | 1.929 (1.906) | 23.9 GiB | 37.804 → 37.068 | 88.2 → 15.2 | 0.291 / 9.702 (5k only) | ≈ 01:45 Nov 22 |
| unetB_s1 | 23968740 | gl029 | 8800 | 1.930 (1.906) | 23.9 GiB | 37.837 → 37.025 | 88.9 → 16.1 | 0.153 / 10.187 (5k only) | ≈ 01:55 Nov 22 |

- All losses and grad norms are finite. The loss drops fast in the first 1k steps, then slowly. The grad norm falls steadily.
  Real speed is within 1.3% of the bench. Memory equals the bench peak (A chunk 256: 17.1 GiB, B: 23.9 GiB).
- Every finish falls well inside its `--time` (A ends at about 11.2 h of 14 h, B at about 16.5 h of 20 h), so no requeue is expected.
- Projected GPU-h for the 4 runs: about 55 [G]. Used by 14:14 EST: about 19.4 (4 jobs × about 4.85 h).
- One observation, not a result: at matched steps the UNet's grad norm is about 2–3× DiT-B/2's (for example DiT A s0 at
  13.4k: 3.2 [E: `runs/expA_s0/metrics.jsonl`]). It falls steadily, and clip 2 is the same for both backbones. S3 compares the
  finished runs.

## Results (S3, 2026-11-22)
All 4 UNet runs finished: 30k steps, no requeue, evaluation of the 5k/10k/20k/30k snapshots complete (EMA and raw)
[E: `runs/unet{A,B}_s{0,1}/eval/step*.json`, `runs/slurm-tmap_u*-*.out`, `sacct`: all COMPLETED, exit 0]. Nothing diverged.
All 4 ran at `grad_chunk` 256 (S2); the DiT runs used 128. Both are exact, so this changes speed only.

The full tables (7 metrics × 4 steps × 8 runs, EMA and raw) are generated by `scripts/summarize_backbones.py` into
`runs/backbone_summary_ema.md` and `runs/backbone_summary_raw.md`. DiT numbers are read from the committed
`runs/expA_*`/`runs/expB_*` eval JSONs, not retyped. Figures: `runs/backbone_metrics_ema.png` (metric vs step; DiT solid,
UNet dashed, A blue, B orange) and `runs/backbone_samples_ema.png` (30k EMA grids, 4 arms × 2 seeds).

### Headline, 30k EMA, seed mean ± half-range [E: `runs/backbone_summary_ema.md`]
| arm | FID-10k | pixel D1 | Inception D1 | DINOv2 D1 | blockiness | 1-NN class agreement | diversity |
|---|---|---|---|---|---|---|---|
| DiT-B/2 A (pixel ES) | 321.6 ± 27.4 | 0.792 ± 0.220 | 12.957 ± 1.052 | 43.391 ± 0.268 | 11.89 ± 0.48 | 0.203 ± 0.007 | 0.93 ± 0.05 |
| **UNet A (pixel ES)** | **158.1 ± 17.9** | 0.878 ± 0.726 | 5.044 ± 0.260 | 27.431 ± 0.329 | 1.02 ± 0.02 | 0.235 ± 0.003 | 0.86 ± 0.06 |
| DiT-B/2 B (+ DINOv2 ES) | 60.4 ± 8.6 | 0.209 ± 0.053 | 1.453 ± 0.292 | 0.427 ± 0.005 (circular) | 1.52 ± 0.02 | 0.201 ± 0.001 | 0.92 ± 0.00 |
| **UNet B (+ DINOv2 ES)** | **26.9 ± 1.9** | 0.464 ± 0.221 | 0.540 ± 0.075 | 1.398 ± 0.387 (circular) | 0.96 ± 0.00 | 0.243 ± 0.007 | 0.85 ± 0.02 |

References [E: same file]: real test images have blockiness 1.01 and 1-NN class agreement 0.288 (chance 0.10). Pixel D1
right-class / wrong-class real: 0.011 / 2.059; Inception D1 0.002 / 3.260; DINOv2 D1 0.005 / 15.672.

Raw weights at 30k [E: `runs/backbone_summary_raw.md`]: FID-10k DiT A 333.0 ± 2.3, UNet A 149.5 ± 19.0, DiT B 62.9 ± 7.0,
**UNet B 17.9 ± 1.0**. Raw blockiness: DiT A 20.56, UNet A 0.99, DiT B 1.62, UNet B 0.96.

### Over training [E: `runs/backbone_summary_ema.md`, `runs/backbone_metrics_ema.png`]
- **UNet B learns much earlier.** EMA FID-10k at 10k steps: UNet B 39.5 ± 1.5, DiT B 354.3 ± 7.5. DiT B only drops
  between 10k and 20k (to 85.9). From 20k to 30k UNet B still improves (31.1 → 26.9, −13%).
- **UNet A plateaus.** EMA FID-10k 160.3 → 130.9 → 170.5 → 158.1 at 5k/10k/20k/30k; not monotone (seed 0 went 128.6 →
  215.1 → 176.0). By the Experiment C rule of `docs/experiments_AB.md` this is "in between", and the grids (below) show
  no objects, so the rule does not send UNet A to 200k.
- **Blockiness.** The UNet is at 0.87–1.02 (real images: 1.01) at every snapshot, in both arms. DiT A is at 5.5–11.9,
  DiT B at 1.0–1.5 (seed means).
- **Pixel D1 does not track sample quality across backbones.** UNet A s1 has the lowest 30k EMA pixel D1 of all 8 runs
  (0.152), yet makes no objects. Pixel D1 has a large seed spread for the UNet (UNet A: 1.604 vs 0.152).
- **DINOv2 D1 (B's training space) favours DiT B** (0.43 vs 1.40), while FID and Inception D1, which B does not train
  on, favour UNet B. A guess [G]: DiT B fits the DINOv2 features more tightly than the image distribution behind them.
- **Grad norm at 30k** [E: last line of each `metrics.jsonl`]: UNet A 7.2 / 5.2, DiT A 32.7 / 119.3 (it grew late; at
  13.4k it was 3.2, see the S2 health check); UNet B 8.6 / 9.5, DiT B 4.5 / 4.0. So the S2 observation (UNet 2–3× DiT)
  holds for B only. All are above clip 2, so clipping is active for every run. Its effect was not tested [G].

### The grids, 30k EMA [E: `runs/backbone_samples_ema.png`, `runs/*/eval/step0030000_samples_{ema,raw}.png`; descriptions are mine]
| arm | what the 10 × 10 grids show (rows = classes) |
|---|---|
| DiT A, both seeds | No objects. Every cell is the same kind of 2×2-pixel colour mosaic around a blurry centre blob; rows differ only in overall tint. This is the DiT A failure (blockiness 11.9). |
| UNet A s0 | No mosaics, but no objects either. Smooth flat backgrounds with one dark or light blob with a bright rim, many near-empty cells. Low variety within a row (EMA diversity 0.80). Class shows only as colour (blue rows for airplane/ship). |
| UNet A s1 | No mosaics, no objects. Smooth two-tone "camouflage" shapes, more varied than s0 (diversity 0.92). The raw grid looks the same. |
| DiT B, both seeds | Objects, but blurry. Vehicle rows (airplane, ship, truck) and animal rows are distinct, yet the animal rows mix classes (the deer row shows dog- and cat-like shapes). Some cells have white blocky patches at the patch scale. This fits DiT B's numbers in `runs/expAB_summary_ema.md` (FID 51.9 / 69.0, blockiness about 1.5). |
| UNet B s0 | Clear objects, nearly all class-correct: airplanes, red cars, birds, cats, deer with antlers, dogs, frogs, horses, ships, trucks. Smooth, no patch artefacts. A few wrong cells (animal faces in the ship row). Good variety in pose and colour. |
| UNet B s1 | Same quality as s0, varied. About 5 of 100 cells are degenerate: a flat dark square with a white blob (visual count [G]). |

So the comparison with DiT [E for the numbers, the grid reading is mine]:
- DiT A makes mosaics; UNet A makes smooth blobs. Neither makes objects.
- DiT B and UNet B both make objects; UNet B's are cleaner and its FID is about half.

## Does the backbone matter, given the loss?
**Yes, by a large factor, at equal data, seeds, loss, optimizer and budget.**

- **Pixel-only A: does a UNet escape DiT A's mosaic failure?** Partly.
  - The mosaic artefact is gone. UNet A's blockiness is 1.02 against DiT A's 11.89 [E], and its FID-10k halves
    (158.1 vs 321.6 [E]).
  - But UNet A still produces no recognisable objects at 30k, only smooth blobs [grid reading]. Its FID has plateaued
    around 130–170 since 5k [E].
  - So the 2×2 mosaic was the DiT-B/2 patch backbone's failure mode under the pixel energy score. "No objects with
    pixel ES alone at 30k" holds for both backbones.
- **B: is the UNet better or worse than DiT at equal budget?** Better.
  - 30k EMA FID-10k 26.9 ± 1.9 vs 60.4 ± 8.6 (raw: 17.9 vs 62.9). Inception D1 0.54 vs 1.45. 1-NN class agreement 0.243
    vs 0.201 (real 0.288) [E].
  - Both seeds of UNet B beat both seeds of DiT B on FID and Inception D1 at 10k, 20k and 30k [E].
  - The UNet reaches DiT B's 30k FID before 10k steps (39.5 at 10k vs 60.4).
  - Cost is not higher: UNet B 1.924 s/step and 16.31 GPU-h per run, DiT B 1.982 s/step and 16.85 GPU-h [E, see below].
  - Where UNet B is worse: DINOv2 D1 (its training feature, circular), pixel D1 (0.46 vs 0.21) and diversity ratio
    (0.85 vs 0.92) [E].
- **The loss matters more than the backbone.** Adding the DINOv2 term moves FID from 158 to 27 for the UNet and from 322 to
  60 for DiT [E]. Switching DiT for the UNet roughly halves FID within each loss.

## Measured cost [E: `runs/backbone_summary_ema.md` "Measured cost"]
| arm | grad_chunk | s/step (median) | GPU-h per run (train + eval) | peak mem | params | GMACs/sample |
|---|---|---|---|---|---|---|
| DiT-B/2 A | 128 | 1.487 | 12.74 | 17.7 GiB | 129.7 M | 21.79 |
| UNet A | 256 | 1.302 | 11.15 | 17.1 GiB | 140.0 M | 20.04 |
| DiT-B/2 B | 128 | 1.982 | 16.85 | 21.4 GiB | 129.7 M | 21.79 |
| UNet B | 256 | 1.924 | 16.31 | 23.9 GiB | 140.0 M | 20.04 |

- GPU-h are wall hours of the allocation segments in each run's slurm `.out`. Each DiT run had one requeue; its restart
  overhead is included and the gap between allocations is not.
- Wall time is not a clean backbone cost: the chunk sizes differ (128 vs 256), and the B step time is dominated by
  DINOv2 at 224 px.
- GPU-h used by this chain: bench 0.13 + 4 UNet runs 54.93 = **55.1 GPU-h** of the 120 budget. S3 submitted no job.

## Caveats
- **2 seeds.** Mean ± half-range of two values is not a confidence interval. Still, the B gap (26.9 vs 60.4) is far
  larger than either arm's seed spread, and in every snapshot from 10k on, both UNet B seeds beat both DiT B seeds.
- **FID-10k** against CIFAR-10 train statistics (clean-fid, legacy_tensorflow), 1000 samples per class. It is biased high
  against FID-50k and not comparable with paper FIDs.
- **30k steps only** (the paper trains 1e6). Rankings may change with longer training; DiT B was still improving at 30k
  (75.5 → 51.9 for s0).
- **Sizes not exactly matched.** UNet 140.0 M params / 20.04 GMACs; DiT-B/2 129.7 M / 21.79 (ratio 1.08 / 0.92).
- **Decided, not specified.** The UNet follows App. I.2, but many details are my choices (table "Design decisions where
  App. I.2 is silent"): one adaptive norm per residual block, attention also on the up path, 2 blocks per level on the up
  path, emb width 1024, zero-initialised last conv, dropout 0. Any of these could matter.
- **The recipe was tuned for neither backbone.** lr 1e-4, warmup, EMA 0.9999 and clip 2 come from the DiT runs.
  EMA 0.9999 has a half-life of about 6.9k steps, so the 5k/10k EMA numbers mostly reflect early weights. The UNet's
  raw weights are better than its EMA at 30k for B (FID 17.9 vs 26.9), which suggests the EMA is still lagging [G].
- **Grid descriptions** are one person's reading of 100 fixed-noise samples per run.

## What to run next (recommendations only; nothing launched)
1. **UNet B for longer**, e.g. 100k–200k steps from `runs/unetB_s0/ckpt/latest.pt` (raise `train.max_steps`). It was still
   improving 20k → 30k (−13% FID). This is the best one-step map so far. At 1.92 s/step, continuing to 100k costs about
   37 GPU-h per seed [G].
2. **Do not spend a long run on pixel-only A** with either backbone. Neither made objects at 30k, and UNet A's FID has
   plateaued since 5k.
3. **FID-50k and a third seed** for the 30k UNet B and DiT B snapshots before quoting final numbers (eval-only, about
   1 GPU-h each [G]).
4. **One ablation to explain the gap**: DiT with patch size 1 or a conv output head, or the UNet without attention, to
   see whether locality or the patch grid causes DiT's artefacts.
5. **Check DINOv2 over-fitting**: evaluate B in a held-out feature space not used in training (e.g. CLIP or SwAV) to test
   the [G] above.
