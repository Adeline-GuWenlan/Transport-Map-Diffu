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
SMOKE_PLACEHOLDER

## GPU bench
BENCH_PLACEHOLDER
