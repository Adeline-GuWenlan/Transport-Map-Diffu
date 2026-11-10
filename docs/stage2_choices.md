# Stage 2: backbone, optimizer and dataset

Written 2026-11-06 by the stage-2 agent (Slurm job 22047086, CPU node, no GPU). The loss and estimator
come from stage 1 (`docs/energy_score_loss.tex`, eqs. (final) and (estimator)): β = 1, per-class groups,
weights n_c/N, U-statistic interaction (diagonal excluded by index), double-where norm, m_c ≥ 2. This
document only designs around them.

Measurements made here are in `docs/stage2_checks/` (scripts, `results/*.json`, `logs/`). They ran in the
stage-1 helper env `tmap-tex` (CPU torch 2.14.1, 8 threads) with timm 1.0.31, torchvision 0.29.2 and
pyarrow 26.0.1 added in throwaway locations (see "Reproduce" at the end).

## Decision summary

| Choice | Decision | Alternatives considered | Why |
|---|---|---|---|
| Backbone code | `facebookresearch/DiT` `models.py`, commit `ed81ce2`, imported **unmodified** from `third_party/DiT` | SiT, fast-DiT, U-ViT, LightningDiT, JiT | CLAUDE.md asks for DiT; it is the reference and one 370-line file; the others add nothing a one-step map needs (details below) |
| Map F_θ(z, c) | `DiT(input_size=32, in_channels=C, learn_sigma=False, class_dropout_prob=0)`; z ~ N(0, I) of shape C×32×32; constant t = 999; F = DiT output directly | drop t; residual F = z + DiT(z); extra noise channels | no upstream edits needed; the t-embedder stays for the later per-noise-level variant; z already has full data dimension |
| Space | **pixel space**, 32×32, values in [−1, 1] | SD-VAE latents (DiT's 256/8 = 32×32×4 setup) | the SD-VAE turns a 32 px image into a 4×4×4 latent; latents need a 256 px dataset (ImageNet, account and terms) and a VAE in every eval; TBSM reports raw-pixel energy-distance generators working on small images |
| CPU smoke model | "DiT-T/4": depth 6, hidden 256, 4 heads, patch 4 (64 tokens). 7.39 M params, 0.32 GMACs/sample, **0.138 s/step** at batch 64 | DiT-S/4 (0.61 s/step), DiT-U/4 (0.06 s/step), DiT-T/2 (0.64 s/step) | well under 1 s; the smallest DiT-table size (S/4) is not |
| GPU model | **DiT-B/2** on 3×32×32: 129.7 M params, 23.0 GMACs/sample | DiT-S/2 (32.6 M, 6.05 GMACs) as the cheap fallback; L/2, XL/2 | the DiT-B/2 compute of the standard latent setup (same 256 tokens). Pixel CIFAR-10 at patch 2 is proven for ViTs: U-ViT-S/2 (embed 512, depth 12, patch 2) reaches FID 3.11 as a diffusion model (`baofff/U-ViT` README, `configs/cifar10_uvit_small.py`); B/2 gives the one-step map more capacity than that |
| Optimizer | AdamW, β = (0.9, 0.95), eps 1e-8, wd 0, linear warmup then constant lr, clip 2.0, EMA | Adam β2 = 0.999 (DiT, IMM, TBSM); wd 0.01 (Drifting); cosine decay | DiT's verified recipe, plus Drifting's β2 = 0.95, clip 2.0 and warmup (the closest one-step, kernel-loss, DiT setting) |
| lr / EMA / steps | GPU: lr 1e-4, warmup 5k, EMA 0.9999, 200k steps. CPU smoke: lr 2e-4, warmup 500, EMA 0.999, 60 min | lr 2e-4–4e-4 (Drifting) | 1e-4 is DiT's value for this backbone; the tiny smoke model gets Drifting's B/2 value to learn faster in 60 min |
| Batch (n × m) | class-stratified. GPU: 8 classes × 64 = N = 512, r = 1 (m_c = n_c = 64). CPU: 8 classes × 8 = 64, m_c = n_c = 8 | random batches; r = 2; per-data-point pairs (engression) | every class present has m_c ≥ 2 by construction; per-class pooling has lower variance than pairs at equal cost (stage 1 a4 and the measurement below); 64 per class as in Drifting |
| Smoke dataset | **Fashion-MNIST**, 28→32 zero-padded, 1 channel | MNIST (fallback), CIFAR-10 (optional second smoke config) | MIT; train and test from the same distribution (held-out floor 0.0004); class structure is a large part of its pixel energy distance, so class conditioning can be checked |
| Real dataset | **CIFAR-10** 32×32 RGB | CIFAR-100, ImageNet 32/64, Tiny ImageNet | open download without an account; standard FID reference stats; same 32×32, 10-class format as the smoke test |
| FID | `clean-fid` 0.1.35 (MIT), `mode="legacy_tensorflow"`, precomputed CIFAR-10 **train** stats; FID-10k while training, FID-50k at the end (EMA) | pytorch-fid, torch-fidelity, EDM's `cifar10-32x32.npz` | no reference-stat computation needed; legacy_tensorflow matches published numbers |

## 1. Backbone

### 1.1 The reference DiT code (read in full)
Cloned with `git clone --depth 1 https://github.com/facebookresearch/DiT.git third_party/DiT`,
commit **`ed81ce2229091fd4ecc9a223645f95cf379d582b`** (2023-06-02). License: **CC-BY-NC 4.0**
(`LICENSE.txt`; README "The code and model weights are licensed under CC-BY-NC"). Non-commercial research
use is allowed; anything we derive from it inherits NC.

What `models.py` does (line numbers at that commit):
- **Size table** (`DiT_XL_2` … `DiT_S_8`, ll. 328–363; `DiT_models` dict l. 365): XL depth 28 / hidden 1152 / 16 heads; L 24/1024/16;
  B 12/768/12; S 12/384/6; each with patch 2, 4, 8. The constructor accepts any depth/hidden/heads, so our
  tiny "T" and "U" sizes are plain keyword arguments, no code change.
- **Input**: `PatchEmbed(input_size, patch_size, in_channels, hidden)` (timm), plus a fixed 2-D sin-cos
  positional embedding (`requires_grad=False`).
- **Conditioning**: `TimestepEmbedder` (sinusoidal 256-d frequency embedding → MLP) and `LabelEmbedder`
  (an `nn.Embedding` of size `num_classes + 1` if `class_dropout_prob > 0`, the extra row being the CFG
  null class; size `num_classes` otherwise). The two embeddings are **summed** (`c = t + y`, l. 243), and the
  sum drives every block's adaLN.
- **adaLN-Zero**: each `DiTBlock` regresses shift/scale/gate for attention and MLP from `SiLU(c)`
  (`adaLN_modulation`, a `6·hidden` linear). `initialize_weights` (ll. 182–216) zeroes those linears in every
  block and the final layer's modulation **and** the final linear. So every block starts as the identity and
  the network output is exactly 0 at step 0. Stage 1 showed this needs the double-where norm (check f).
- **Output**: `FinalLayer` → `unpatchify`; `out_channels = in_channels * 2 if learn_sigma else in_channels` (l. 165).
  `learn_sigma=True` (the default) adds a predicted-Σ half trained with the variational bound
  (paper §3.1, "train Σθ with the full L"). We set **`learn_sigma=False`**: there is no Σ in our loss, and the
  output then has exactly the data's shape.
- **Latent setup** (`train.py`): `latent_size = image_size // 8` (l. 141), SD-VAE
  `stabilityai/sd-vae-ft-{ema,mse}` (l. 151), latents scaled by `0.18215` (l. 203), so 256 px → 32×32×4.
- **Training recipe** (`train.py`): `AdamW(lr=1e-4, weight_decay=0)` (l. 155), global batch 256 (l. 262),
  EMA decay 0.9999 (`update_ema`, l. 40), EMA initialised to the model (l. 184), random horizontal flip
  only (l. 160), no clipping, no warmup, fp32.

Paper (Peebles & Xie, ICCV 2023, https://arxiv.org/abs/2212.09748; text read at
https://ar5iv.labs.arxiv.org/html/2212.09748): "We train all models with AdamW … constant learning rate of
1×10⁻⁴, no weight decay and a batch size of 256", "we did not find learning rate warmup nor regularization
necessary", "The only data augmentation we use is horizontal flips", EMA "with a decay of 0.9999". All four
values the stage prompt asked about are **confirmed** in both paper and code. The paper's Table 4 Gflops
(S/B/L/XL at patch 2: 6.06 / 23.01 / 80.71 / 118.64) are reproduced by our counter (below), so our
"GMACs" column is in the paper's units.

### 1.2 Alternatives

| Code base | License (checked) | What it is | Suits us better? |
|---|---|---|---|
| SiT, https://github.com/willisma/SiT (`cbde832`) | MIT (repo) | interpolant/flow training around the same DiT net. Its `models.py` is DiT's `models.py` with 3 lines changed (we diffed it: the Meta copyright header is gone, and `forward` drops the Σ half when `learn_sigma`) | No. Same network, and the MIT label on a near-verbatim copy of CC-BY-NC code is not a clean licence. Its flow-matching losses are not ours |
| fast-DiT, https://github.com/chuanyangjin/fast-DiT | CC-BY-NC 4.0 (`LICENSE.txt`) | DiT plus pre-extracted VAE features, mixed precision, gradient checkpointing | No. Its speed-ups target latent training; we start in pixel space |
| U-ViT, https://github.com/baofff/U-ViT | MIT | ViT with long skips; time and class as tokens, no adaLN; has a pixel CIFAR-10 config | Not as the base: CLAUDE.md fixes DiT, and adaLN-Zero is what makes the t-conditioner reusable later. A good second backbone if DiT under-fits in pixel space |
| LightningDiT, https://github.com/hustvl/LightningDiT | MIT | modernised DiT (SwiGLU, RMSNorm, RoPE, qk-norm) for VA-VAE latents | Later, for the latent ImageNet test. More deps and latent-specific |
| JiT, https://github.com/LTH14/JiT | MIT | plain pixel-space ViT with large patches (TBSM uses JiT-B/16) | Not for 32 px; relevant if we go to 256 px pixels |

**Public code for one-step generators trained with energy distance, scoring rules or MMD:**
- **TBSM**, "Three-Body Scattering for Generative Modeling", Sun et al., arXiv 2607.18198 (Jul 2026),
  https://arxiv.org/abs/2607.18198, code https://github.com/sp12138/TBSM (`d1d43f5`), **Apache-2.0** except
  `imgnet/networks/backbones/pixeldit/` (NVIDIA NSCLv1, non-commercial). It recasts the energy distance as a
  per-sample attraction to a real sample and repulsion from an independent generated one, and trains
  one-step generators. ImageNet-256 NFE = 1: FID 1.63 (latent DiT-XL/2), 2.23 (pixel PixelDiT-XL). But:
  ImageNet runs **initialise from a pretrained diffusion/flow model** and compute the energy distance on
  **frozen encoder features** (ResNet-18, SigLIP2-B, MAE-B), not raw pixels. Its paper says "raw-pixel
  scattering for low-dimensional data and small images"; MNIST, Fashion-MNIST and CIFAR-10 appear only as
  qualitative samples from small U-Nets. `minimal/configs/identity.yaml` is a raw-pixel MNIST run (32 px,
  batch 128, 10k steps, generator lr 1e-5). Optimiser (Table 4): AdamW lr 1e-5, β = (0.9, 0.999), EMA
  0.999, batch 128–256, bf16; it is fine-tuning, so the low lr does not transfer to training from scratch.
- **Drifting Models**, Deng, Li, Li, Du, He, arXiv 2602.04770 (Feb 2026),
  https://arxiv.org/abs/2602.04770 (read v2 HTML). A one-step DiT-style generator f(ε) on SD-VAE latents
  32×32×4 (patch 2) or pixels (patch 16), with a kernel k(x, y) = exp(−‖x − y‖/τ) attraction–repulsion
  field. It is related to MMD (their App. C.2), not exactly the energy distance. The loss is computed on
  **frozen features** (latent-MAE by default): "we were unable to make our method work on ImageNet without
  a feature encoder". The batch is grouped by class: N_c = 64 labels × 64 positives × 64 generated
  negatives (B = 4096). Optimiser (Table 8): AdamW, β = (0.9, 0.95), wd 0.01, lr 2e-4 (B/2 ablation) to
  4e-4, warmup 5k–10k, clip 2.0, EMA 0.999 (ablation) or tuned in {0.999 … 0.9999}. No official code
  found; an unofficial one exists at https://github.com/aengusng8/DriftingModel (no licence).
- **Inductive Moment Matching (IMM)**, Zhou, Ermon, Song, ICML 2025, https://arxiv.org/abs/2503.07565,
  code https://github.com/lumalabs/imm (`e124a9d`), **CC BY-NC-SA 4.0**. Few-step generator on DiT
  (ImageNet-256) and DDPM++ (CIFAR-10) trained with an MMD (Laplace kernel) between groups of M = 4
  particles; M = 1 or 2 collapsed (their §7.2). `configs/im256.yaml`: AdamW lr 1e-4, β = (0.9, 0.999),
  wd 0, batch 4096, EMA 0.9999; `configs/cifar10.yaml`: RAdam lr 1e-4, same β, batch 4096, EMA 0.9999.
- **Distributional diffusion with scoring rules**, De Bortoli et al., ICML 2025,
  https://arxiv.org/abs/2502.02483. Generalised energy score per noise level; the noise ξ is concatenated
  with x_t along channels; m = 16 samples per data point in the robotics runs. Image hyperparameters are
  in App. I.2, which I could not retrieve (the HTML is truncated), so **unverified**. No code URL in the
  text I read.
- **Engression**, Shen & Meinshausen, https://github.com/xwshen51/engression (`eb6b83b`), **BSD-3**. MLP
  generators; `torch.optim.Adam`, default lr 1e-4 for continuous responses (`engression.py` ll. 132–133),
  m = 2 per observation.

None of these is a drop-in for us: TBSM and Drifting work in feature space and (TBSM) fine-tune, IMM is
NC-SA and multi-step, engression is MLP-only. We take their batch and optimiser evidence, not their code.
The common thread matters: **at ImageNet scale, both one-step kernel/energy methods needed a frozen feature
space**. That is the main risk for us, and the fallback is noted in §6.

### 1.3 DiT as a transport map F_θ(z, c)
- **Input z**: z ~ N(0, I) with the image's shape, C×32×32 (C = 1 for Fashion-MNIST, 3 for CIFAR-10), drawn
  fresh for every generated sample. The noise has the full data dimension, so no extra noise channels are
  needed and no bottleneck exists.
- **Timestep**: **keep** DiT's `t_embedder` and feed a **constant** t = 999 to every sample (DiT's
  index for pure noise). It then acts as a learned bias in `c = t_emb + y_emb`. This keeps the upstream file
  untouched (dropping t would mean editing `forward`), costs < 1% of the parameters and no FLOPs worth
  counting, and leaves the input ready for the next step (conditional energy score at several noise levels,
  stage-1 note §8), where t becomes a real conditioner.
- **Output head**: `learn_sigma=False`, so `out_channels = in_channels` and F(z, c) is the DiT output
  itself (`output: direct`). We keep DiT's zero-initialised head: F ≡ 0 at step 0 is safe with the
  double-where norm (stage 1 check f). We do not use a residual F = z + DiT(z): the network would have to
  learn to cancel z exactly to remove the noise.
- **Class conditioning**: `class_dropout_prob = 0`: no null class, no classifier-free guidance in the first
  test. (One-step CFG needs training-time support, as Drifting does; out of scope.)
- **Extra noise channels**: none now. For the later diffusion variant the input is (x_t, ξ) concatenated
  along channels (De Bortoli et al.), so in_channels = 2C but out_channels = C, which DiT's constructor
  cannot express. Stage 4's `model.py` should therefore take `in_channels` and `out_channels` separately. With
  `out_channels != in_channels`, it replaces `model.final_layer` with a new `FinalLayer(hidden, p, C)`
  (zero-initialised) after construction. That is a wrapper, not an edit of `third_party/`.
- **Upstream code**: import `DiT` and `FinalLayer` from `third_party/DiT/models.py` by putting that
  directory on `sys.path` (it needs `timm` for `PatchEmbed`, `Attention`, `Mlp`). Do not copy or modify it.

### 1.4 Pixel space, not latent space, for the first test
1. **Resolution.** DiT's latent setup is the SD-VAE at f = 8 on 256 px images. On a 32 px dataset the latent
   would be 4×4×4, which is useless; latents only make sense with a 256 px dataset. The only standard
   labelled one is ImageNet, which needs an account and terms (§3).
2. **Cost and dependencies.** Every evaluation sample would have to go through the VAE decoder (and the
   training set through the encoder once). On this CPU node that dominates the smoke test, and it adds
   `diffusers` plus a 335 MB checkpoint (`stabilityai/sd-vae-ft-ema`, safetensors 334.6 MB) before we know whether the loss works at all.
3. **Interpretability.** Pixel space is where the held-out energy distance, nearest neighbours and sample
   grids are directly readable, and it is the space the stage-1 estimator was costed for (d = 3072).
4. **Evidence that pixel space can work at this size.** TBSM uses "raw-pixel scattering for … small images"
   and shows MNIST, Fashion-MNIST and CIFAR-10 samples (qualitative only). Drifting and TBSM needed feature
   spaces for **ImageNet**, not for small images.

The risk, quantified (§4.3): in CIFAR-10 pixel space the class structure is a small part of the energy
distance, so class conditioning may be learned slowly. Latent (ImageNet-256, SD-VAE) is the natural
second test once the pixel test works. Feature-space energy distance is the fallback if it does not.

### 1.5 Sizes: measured parameter counts, FLOPs and CPU step time
`docs/stage2_checks/time_dit.py` → `results/dit_sizes.json`. FLOPs are from `torch.utils.flop_counter`
plus the attention matmuls 2·T²·D per layer, added analytically because the counter does not see timm's
`scaled_dot_product_attention` on CPU. GMACs = multiply-adds, forward, per sample. A step is forward,
loss (eq. 9, per class), backward, clip and AdamW, fp32, batch 64 (10 classes), 8 threads, median of 10
after 3 warm-up steps.

| Model | Input | Tokens | Params (M) | Fwd GMACs | DiT paper Gflops | Step @ b64 (s) |
|---|---|---|---|---|---|---|
| DiT-XL/2 | 4×32×32 (latent) | 256 | 675.1 | 118.62 | 118.64 | – |
| DiT-B/2 | 4×32×32 (latent) | 256 | 130.5 | 23.00 | 23.01 | – |
| DiT-S/2 | 4×32×32 (latent) | 256 | 33.0 | 6.05 | 6.06 | – |
| DiT-XL/2 | 3×32×32 | 256 | 674.0 | 118.61 | | – |
| DiT-L/2 | 3×32×32 | 256 | 457.1 | 80.69 | | – |
| **DiT-B/2 (GPU)** | 3×32×32 | 256 | **129.7** | **23.00** | | – |
| DiT-S/2 (GPU fallback) | 3×32×32 | 256 | 32.6 | 6.05 | | – |
| DiT-B/4 | 3×32×32 | 64 | 129.6 | 5.56 | 5.56 | – |
| DiT-S/4 | 3×32×32 | 64 | 32.5 | 1.41 | 1.41 | 0.595 |
| DiT-S/4 | 1×32×32 | 64 | 32.5 | 1.41 | | 0.607 |
| DiT-T/2 (ours) | 1×32×32 | 256 | 7.43 | 1.41 | | 0.637 |
| **DiT-T/4 (ours, CPU smoke)** | 1×32×32 | 64 | **7.39** | **0.318** | | **0.138** |
| DiT-T/4 (ours) | 3×32×32 | 64 | 7.41 | 0.319 | | 0.138 |
| DiT-U/4 (ours) | 1×32×32 | 64 | 2.85 | 0.121 | | 0.061 |
| DiT-T/8 (ours) | 1×32×32 | 16 | 7.40 | 0.079 | | 0.047 |

T = depth 6, hidden 256, 4 heads; U = depth 4, hidden 192, 3 heads (head dim 64, as in DiT). Our FLOP counts
match the paper's Table 4 to within 0.02 Gflops, which confirms the counting. DiT-T/4 gives about 7 steps/s
on the node: ~26k steps of batch 64 in the 60 min smoke run. Patch 4 rather than 2 because T/2 costs 4.6×
more per step for the same parameters; patch 8 (16 tokens of 8×8 px) is too coarse for 32 px images.

**GPU cost estimate (assumption, not measured; no GPU here).** DiT-B/2: fwd + bwd ≈ 3 × 23.0 = 69 GMACs
= 138 GFLOP per generated sample, × 512 = 71 TFLOP per step. At an assumed ~100–150 TFLOP/s sustained in
bf16 on one L40S-class GPU, that is ~0.5–0.7 s/step, so 200k steps ≈ 28–39 h. The GPU job must be
requeue-safe (stage 4 writes resume). DiT-S/2 is ~4× cheaper if a faster first answer is wanted.

## 2. Optimizer

### 2.1 What others used (all checked at the cited source)
| Work | Loss | Optimiser | lr | β | wd | Batch | EMA | Clip / warmup |
|---|---|---|---|---|---|---|---|---|
| DiT (paper + `train.py`) | diffusion MSE + VLB | AdamW | 1e-4 const | default (0.9, 0.999) | 0 | 256 | 0.9999 | none / none |
| Drifting (Table 8) | kernel drift, one-step | AdamW | 2e-4 (B/2 abl.), 4e-4 | (0.9, 0.95) | 0.01 | 4096–8192 (64/class) | 0.999–0.9999 | 2.0 / 5k–10k |
| IMM (`configs/im256.yaml`) | MMD, Laplace kernel | AdamW | 1e-4 | (0.9, 0.999) | 0 | 4096 (groups of 4) | 0.9999 | – |
| TBSM (Table 4; fine-tune) | energy distance in features | AdamW | 1e-5 | (0.9, 0.999) | – | 128–256 | 0.999 | – |
| Engression (code) | energy score, MLP | Adam | 1e-4 | default | – | – | – | – |
| U-ViT-S/2 CIFAR-10 (`configs/cifar10_uvit_small.py`) | diffusion, pixel ViT | AdamW | 2e-4 | (0.99, 0.999) | 0.03 | 128 | – | 500k steps |

### 2.2 Our choice
- **AdamW, β1 = 0.9, β2 = 0.95, eps = 1e-8, weight decay 0.** β2 = 0.95 follows Drifting, the closest setting:
  a one-step DiT generator with a class-grouped attraction–repulsion loss. A shorter second-moment memory
  tracks the gradient scale, which shrinks as P_θ → Q (below). Weight decay 0 as DiT and IMM, to keep the
  first test free of extra regularisation.
- **lr**: 1e-4 on GPU (DiT's verified value for DiT-B; IMM uses the same at a 16× larger batch). 2e-4 for
  the 7 M-parameter CPU smoke model (Drifting's B/2 value), to make progress in 60 min.
- **Schedule**: linear warmup (5000 steps GPU, 500 CPU), then **constant**. DiT found warmup unnecessary
  for diffusion, but our loss is new and starts from F ≡ 0, and warmup is cheap insurance (Drifting uses
  it). Constant lr plus EMA (as DiT and IMM) makes runs extendable and resume trivial.
- **Gradient clipping** at global norm 2.0 (Drifting). Measured on the smoke model, the pre-clip norm is
  3–67, so this binds on every step. It then acts as gradient normalisation and made no difference to the
  loss over 400 steps (§2.5). Log the pre-clip norm.
- **EMA** of the weights, initialised to the model at step 0: 0.9999 on GPU (DiT, IMM; a ~10k-step horizon,
  fine for 200k steps), 0.999 for the ~25k-step CPU smoke run (a 0.9999 EMA would still be mostly initial
  weights). Evaluate both raw and EMA weights in the smoke run.
- **Precision**: fp32 on CPU; bf16 autocast for the network on GPU, but compute **distances in fp32**.
- **Augmentation**: horizontal flip for CIFAR-10 (DiT's only augmentation); none for Fashion-MNIST, so the
  smoke target is exactly the data distribution.

### 2.3 Batch structure (n × m)
- **Class-stratified sampler**: each step picks `classes_per_batch` classes uniformly without replacement,
  then `n_per_class` training images of each. With balanced datasets the expected loss is
  Σ_c w_c S(P_c, Q_c) with uniform w_c = π_c, the same minimiser (stage-1 Prop. 15 and remark on weights).
- **r = 1 generated sample per data point**: m_c = n_c, so the network cost equals an ordinary batch of N.
- GPU: 8 × 64 → N = M = 512. CPU: 8 × 8 → N = M = 64. (The timing in §1.5 used batch 64 spread over 10
  classes; the network cost is the same, and the per-class distance cost is negligible either way.)
- Per class, the loss pairs all m_c samples with all n_c data points (confinement) and all m_c(m_c − 1)
  ordered sample pairs (interaction). The per-class blocks are small (64 × 64 × 3072), so compute them
  with the exact difference form `a[:, None] - b[None]` per class, which avoids the cancellation error of
  ‖a‖² + ‖b‖² − 2⟨a, b⟩. Memory: 64·64·3072·4 B = 50 MB per class block, fine; chunk if needed.

### 2.4 Specific to the energy score
**Gradient scale is free of the data scale (β = 1).** For one generated sample x_j of class c, the gradient
of eq. (9) with respect to x_j is (1/(N m_c)) Σ_i u(x_j − y_i) − (n_c/(N m_c (m_c − 1))) Σ_{k≠j} u(x_j − x_k),
where u(v) = v/‖v‖ is a **unit vector**. Each term has norm ≤ n_c/(N m_c), so with m_c = n_c the total is
≤ 2/N, whatever the pixel range or dimension. The loss is 1-homogeneous: rescaling data and outputs by s scales the loss by s and leaves
the output-gradients unchanged. With Adam, pixel ([−1, 1]) and latent (×0.18215) scaling therefore need no
lr change. What differs between spaces is the geometry, i.e. which directions dominate the distances.
Near the optimum, the attraction and repulsion unit-vector averages cancel, so gradients shrink: another
reason for β2 = 0.95 and for logging grad norms.

**Loss value vs. learning signal.** At P = Q the expected loss equals ½E‖Y − Y′‖ (per class), which is a
large constant. The learning signal is ½D₁ on top of it (measured numbers in §4.3). So **do not judge
training by the raw loss**: log the held-out D̂₁ (unbiased, can be < 0 on a batch).

**Gradient variance as m changes.** Stage 1 derived Var ≈ σ²_Y/n + τ²/m with τ² → 0 at the optimum: extra
generated samples only help early in training. It also found per-class pooling up to 19× less variable
than per-data-point groups at equal cost (2-D). The DiT-scale measurement is in §2.5.

### 2.5 Measured: gradient noise vs r on the smoke DiT (Fashion-MNIST)
`docs/stage2_checks/grad_var_m.py` → `results/grad_var_m.json`. The smoke model (DiT-T/4, Fashion-MNIST,
batch 8 classes × 8) was first trained 400 steps (57 s; batch loss 27.7 → 10.1) so that the interaction
term is active (at step 0, F ≡ 0 and its gradient is exactly 0). At that θ: K = 24 gradient draws each, the
trace of the covariance of the full parameter gradient (7.4 M entries); ‖mean‖² ≈ 230 throughout.

| Gradient noise source (fixed θ) | r = m_c / n_c | tr Cov | noise / signal | s per gradient |
|---|---|---|---|---|
| fresh z only, data fixed, per-class pooled | 1 | 11.6 | 0.050 | 0.14 |
| ″ | 2 | 6.4 | 0.027 | 0.33 |
| ″ | 4 | 2.9 | 0.013 | 0.72 |
| ″ | 8 | 1.8 | 0.008 | 2.11 |
| fresh z only, **per-data-point pairs** (engression: n_g = 1, m_g = 2) | 2 | 15.9 | 0.070 | 0.35 |
| fresh data **and** z, per-class pooled | 1 | 39.3 | 0.133 | 0.15 |

Reading (single seed, K = 24, so ±~30% on each trace):
- z-noise falls like 1/r (11.6 → 6.4 → 2.9 → 1.8), as stage 1 predicted.
- At r = 1 the generated-sample noise is **~30% of the total** (11.6 of 39.3); data sampling is the rest.
  Doubling r doubles the cost and removes ~5 of 39 (−13%). Doubling N at the same cost should roughly halve
  the total (expected from the 1/n and 1/m terms; not measured). So **spend compute on more data points per class, not on more samples per data point: r = 1.**
- Per-data-point pairs at the same cost as r = 2 are **2.5× noisier** than per-class pooling (15.9 vs 6.4),
  and noisier even than pooled r = 1 at half the cost. This confirms stage 1's (a4) at DiT scale.

**Gradient norms and clipping** (`results/warm_clip_{0,2.0}.json`, same 400 steps): the pre-clip global
norm was 5–67 (median 25) in the first 100 steps and 3–28 (median 10–12) in the last 100. Drifting's clip
of 2.0 therefore **binds on every step**. With and without it, the loss curves are the same to ±0.2
(loss per 100 steps 20.0/11.5/11.1/10.2 vs 20.3/11.7/11.1/10.5), as expected: Adam is invariant to the
gradient's overall scale. We keep `grad_clip_norm: 2.0`, which then works as per-step gradient
normalisation that also caps spikes, and the setting that was tested. Stage 4 should not read "clip
active on 100% of steps" as a fault.

## 3. Dataset

### 3.1 Comparison
Sizes and class counts were checked on the downloaded files where marked ✓ (`results/data_scale_*.json`).

| Dataset | Train / test | Res. | Classes | License / terms | Download (no account?) | Standard FID ref. |
|---|---|---|---|---|---|---|
| MNIST | 60k / 10k ✓ (unbalanced: 5421–6742 per class ✓) | 28×28 gray | 10 | no licence on the original page (http://yann.lecun.com/exdb/mnist/, unreachable from here); HF card `ylecun/mnist` says MIT | torchvision `MNIST` (mirror https://ossci-datasets.s3.amazonaws.com/mnist/, 200 OK ✓); HF `ylecun/mnist` ungated ✓ | none standard |
| **Fashion-MNIST** | 60k / 10k ✓ (6000 / 1000 per class ✓) | 28×28 gray | 10 | **MIT** (https://github.com/zalandoresearch/fashion-mnist) | torchvision `FashionMNIST` (http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/, 200 OK ✓, md5-checked by torchvision); HF `zalando-datasets/fashion_mnist` (MIT, ungated ✓) | none standard |
| **CIFAR-10** | 50k / 10k ✓ (5000 / 1000 per class ✓) | 32×32 RGB | 10 | no licence stated; the page asks to cite Krizhevsky 2009 (https://www.cs.toronto.edu/~kriz/cifar.html); HF card "unknown". Universally used for research | HF `uoft-cs/cifar10` parquet (ungated, 143 MB in 2 s ✓); torchvision `CIFAR10` (Toronto, now redirects to cave.cs.toronto.edu, measured ~140 kB/s → ~20 min) | **yes**: clean-fid `cifar10` train/test (✓ reachable); EDM `cifar10-32x32.npz` (✓) |
| CIFAR-100 | 50k / 10k (500 / 100 per class) | 32×32 RGB | 100 fine / 20 coarse | as CIFAR-10 | HF `uoft-cs/cifar100` (ungated ✓); torchvision `CIFAR100` (✓ 200) | not in clean-fid; none widely standard |
| ImageNet 32/64 (Chrabaszcz et al. 2017, https://arxiv.org/abs/1707.08819) | 1.28M / 50k (ImageNet-1k; not checked here) | 32 or 64 RGB | 1000 | ImageNet terms (non-commercial research) | official: image-net.org, needs an account (not re-checked: the page did not render via curl); HF `ILSVRC/imagenet-1k` gated (✓ `gated: auto`); ungated re-uploads `benjamin-paine/imagenet-1k-{32x32,64x64}` (1.9 GB for 64; licence "imagenet-agreement", ✓) | 64×64: EDM `imagenet-64x64.npz` (✓ reachable); guided-diffusion batches |
| Tiny ImageNet | 100k / 10k val (500 / 50 per class; standard description, not checked) | 64 RGB | 200 | ImageNet subset, ImageNet terms | http://cs231n.stanford.edu/tiny-imagenet-200.zip (248 MB, 200 OK ✓) | none standard |

### 3.2 Recommendation
- **CPU smoke: Fashion-MNIST**, padded to 32×32 (2 px of background on each side), 1 channel, in [−1, 1].
  It has an MIT licence from its authors, train and test come from the same distribution (held-out floor
  D̂₁ = 0.0004, vs 0.036 for MNIST), and its class structure is large relative to within-class spread
  (wrong-class D̂₁ = 12.7 vs E‖Y−Y′‖ = 22.7), so the per-class metric can confirm the conditioning.
  **MNIST** is the fallback if Fashion-MNIST is too hard in 60 min (identical format).
- **Real test: CIFAR-10**, 32×32 RGB, hflip. It is the standard small benchmark with FID reference stats,
  downloadable without an account, in the same format as the smoke test: 32×32, 10 balanced classes, the
  same DiT token grid. Only `in_channels` changes. `configs/smoke_cpu_cifar10.yaml` is an optional CPU
  bridge run.
- Not chosen: CIFAR-100 (100 classes × 500 images makes per-class groups small: at N = 512 only ~5 per class
  if all are present; harder to judge), ImageNet 32/64 (account and terms, 1.28M images; the ungated mirrors
  re-host ImageNet under its agreement; save for later), Tiny ImageNet (no standard FID; ImageNet terms).

### 3.3 Download commands (for stage 3)
Each download goes in its own empty directory. Check hashes. Parse with numpy, pyarrow or PIL; no pickle
needed. Suggested on-disk form for stage 4: `data/<name>/{train,test}_{images,labels}.npy` (uint8
N×C×H×W at native resolution, int64 labels). Pad Fashion-MNIST/MNIST to 32 in the loader.

```bash
# Fashion-MNIST (MIT). sha256 measured here:
mkdir -p data/fashion_mnist/raw && cd data/fashion_mnist/raw
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  curl -sSLO http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/$f; done
# 3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84  train-images-idx3-ubyte.gz
# a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845  train-labels-idx1-ubyte.gz
# 346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073  t10k-images-idx3-ubyte.gz
# 67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5  t10k-labels-idx1-ubyte.gz
# (equivalently: torchvision.datasets.FashionMNIST(root, download=True), which checks md5)

# MNIST (fallback):
mkdir -p data/mnist/raw && cd data/mnist/raw
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  curl -sSLO https://ossci-datasets.s3.amazonaws.com/mnist/$f; done
# 440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609  train-images-idx3-ubyte.gz
# 3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c  train-labels-idx1-ubyte.gz
# 8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6  t10k-images-idx3-ubyte.gz
# f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6  t10k-labels-idx1-ubyte.gz

# CIFAR-10 from the HF mirror, pinned to revision 0b2714987fa478483af9968de7c934580d0bb9a2:
mkdir -p data/cifar10/raw && cd data/cifar10/raw
R=https://huggingface.co/datasets/uoft-cs/cifar10/resolve/0b2714987fa478483af9968de7c934580d0bb9a2/plain_text
curl -sSLO $R/train-00000-of-00001.parquet; curl -sSLO $R/test-00000-of-00001.parquet
# 8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883  train-00000-of-00001.parquet (50000 rows)
# 841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de  test-00000-of-00001.parquet  (10000 rows)
# columns: img {bytes: PNG, path}, label int64. Decoder: load_cifar10_parquet in docs/stage2_checks/data_scale.py
# Slow official alternative (~20 min here): torchvision.datasets.CIFAR10(root, download=True), md5 c58f30108f718f92721af3b95e74349a
# Cross-check: the class counts must be 5000/1000 per class.

# FID reference statistics (clean-fid downloads these itself on first use; pre-fetch to be safe):
#   https://www.cs.cmu.edu/~clean-fid/stats/cifar10_legacy_tensorflow_train_32.npz   (32 MB)
#   https://www.cs.cmu.edu/~clean-fid/stats/cifar10_clean_train_32.npz               (32 MB)
#   Inception weights: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt (96 MB)
# clean-fid caches stats inside its package dir (cleanfid/stats/), i.e. in the env on /scratch, not HOME.
# Cross-check stats: https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz (EDM; mu/sigma, 50k train)
```

### 3.4 Evaluation plan
1. **Held-out energy distance (primary, cheap, exact target of the loss).** Per class, the unbiased D̂₁
   (stage-1 note §7) between n generated samples and n **test** images, n = 1000 (500 during smoke
   training); report per class and the class mean, β = 1, pixel space. Always print the **baselines**
   (measured here, mean over classes, n = 1000; `results/data_scale_*.json`):

   | D̂₁ to test set, per class (mean) | Fashion-MNIST | MNIST | CIFAR-10 |
   |---|---|---|---|
   | N(0, I) noise ("ceiling") | 22.02 | 26.03 | 9.22 |
   | point mass at 0 (DiT at step 0) | 38.01 | 44.22 | 17.85 |
   | point mass at the class mean (collapse) | 7.31 | 7.57 | 14.78 |
   | train images of the **wrong** class (c + 1) | 12.68 | 6.09 | 2.14 |
   | train images of the right class ("floor") | 0.0004 | 0.036 | 0.005 |
   | E‖Y − Y′‖ within class (2 × optimal loss value) | 17.43 | 17.92 | 36.41 |

   A model that beats "wrong class" has learned class-specific structure; the floor's spread (|D̂₁| up to
   ~0.03 per class at n = 1000) is the noise level of the metric. On GPU, also compute D̂₁ in Inception
   pool3 space, from the features already computed for FID.
2. **FID** (CIFAR-10): `clean-fid` 0.1.35, `fid.compute_fid(gen_dir or gen=fn, dataset_name="cifar10",
   dataset_res=32, dataset_split="train", mode="legacy_tensorflow")`, i.e. against the 50k train set, the
   convention of published CIFAR-10 FIDs. Also report `mode="clean"`. FID-10k every 10k steps, FID-50k at
   the end, EMA weights, 5k per class. On CPU (smoke CIFAR-10 bridge run only): FID-5k, labelled
   **indicative** (biased upward at small n; compare only like with like). Estimate, not measured:
   InceptionV3 ≈ 5.7 GFLOPs per 299² image, so ~10k images take several minutes on 8 cores. Fashion-MNIST:
   no FID (no standard reference); use D̂₁ only.
3. **Per-class sample grids**: 10 classes × 10 samples, the same fixed z at every checkpoint (shows
   progress, and whether different z give different images, i.e. collapse), raw and EMA weights.
4. **Memorisation**: for 1000 (CPU) / 10k (GPU) generated samples, the nearest training image in pixel L2
   (and Inception space on GPU). Compare the distribution of these NN distances with the same statistic for
   **test** images against the train set (unseen data as reference). Generated NN distances much smaller
   than test-to-train ones signal copying. Show the 20 closest generated–train pairs as a grid.
5. **Diversity / collapse check**: mean within-class pairwise distance of generated samples vs. E‖Y − Y′‖
   within class (the table above). Much smaller means under-dispersion or collapse.

## 4. Measurements behind the choices

### 4.1 Timing and FLOPs: §1.5.
### 4.2 Gradient noise vs r: §2.5.
### 4.3 Loss scale and class signal in pixel space (`docs/stage2_checks/data_scale.py`)
Scaled to [−1, 1], d = 1024 (padded MNIST-type) or 3072 (CIFAR-10). Numbers in the §3.4 table.
- The loss value at the optimum is ½E‖Y − Y′‖ ≈ 8.7 (Fashion-MNIST) or 18.2 (CIFAR-10) per class. At step
  0 (F ≡ 0) the excess ½D₁ is 19.0 (Fashion-MNIST) and 8.9 (CIFAR-10). A good model should get the excess
  below ~0.05, i.e. the informative part is < 1% of the logged loss near convergence. Log D̂₁.
- **Class signal.** CIFAR-10 images of the *wrong* class are only D̂₁ = 2.14 from the right class's test
  set (0.66 for bird → cat, 0.70 for cat → deer), against 9.2 for Gaussian noise and E‖Y − Y′‖ ≈ 38 over all classes (36 within a class). In
  Euclidean pixel space, CIFAR-10's class identity is ~6% of the distance scale (2.14 / 37.9), so a pixel-space energy
  score will learn global image statistics long before class-specific content. Fashion-MNIST's class
  identity is ~56% of the scale (12.7 / 22.7, all-class E‖Y − Y′‖). This is why the smoke test uses Fashion-MNIST, and why
  feature-space energy distance is the planned fallback for CIFAR-10 (§6).
- On CIFAR-10, N(0, I) noise (D̂₁ 9.2) is closer to the data than the zero-initialised DiT (17.9) or a
  class-mean point mass (14.8). Point masses are heavily penalised, as a strictly proper score should.

## 5. Package list for stage 3 (env `/scratch/wg2381/.conda/envs/tmap`)
- conda-forge: `python=3.11`, `pip`. (Use `--override-channels -c conda-forge`.)
- pip, CUDA build of PyTorch that also runs on CPU nodes, from `https://download.pytorch.org/whl/cu126`:
  `torch==2.14.1 torchvision==0.29.1`. Both cp311 manylinux_2_28 wheels exist there (✓). cu126 is chosen over
  cu130 because it needs an older NVIDIA driver (CUDA 12.x minor-version compatibility; **assumption:**
  the GPU nodes' driver supports CUDA 12.6; check `nvidia-smi` in the GPU job). There is no cu128 wheel for
  2.14.x on the index (✓).
- pip, pinned: `timm==1.0.31` (DiT imports `PatchEmbed`, `Attention`, `Mlp`; tested here with DiT
  `ed81ce2` and torch 2.14.1), `numpy`, `scipy`, `pyyaml`, `pillow`, `matplotlib`, `tqdm`, `pyarrow`
  (CIFAR-10 parquet), `pytest`, `requests`.
- pip: `clean-fid==0.1.35` (MIT). Install with `--no-deps` after torch, because its requirements list
  unpinned `torch`/`torchvision`. Its other deps (numpy, scipy, tqdm, pillow, requests) are above.
- Not needed now: `diffusers`/`accelerate` (SD-VAE, only for a later latent test), `torch-fidelity`,
  `pytorch-fid`.
- timm imports `huggingface_hub` and `safetensors` (pulled in by a normal `pip install timm`). With
  `--no-deps` they must be added by hand. Here an incomplete huggingface_hub printed a harmless
  "Error importing huggingface_hub.hf_api" warning.

## 6. Risks and open questions
1. **Pixel-space energy distance may ignore class content on CIFAR-10** (§4.3). Drifting could not train
   without a feature encoder on ImageNet; TBSM uses frozen encoders for natural images. If CIFAR-10 samples
   match global statistics but not classes, the next step is the same loss on frozen features. Two
   options: (a) a TBSM/Drifting-style frozen encoder, which needs a pretrained network and changes what
   "proper" means: proper only for the feature distribution; (b) a multi-scale pixel loss (energy
   distances on downsampled/patch views, summed: each term proper for its marginal). Decide after the
   smoke and GPU results.
2. **Slow fine detail.** Stage 1 (c2) needed ~20k steps to separate 8 modes in 2-D. The smoke run gets ~25k
   steps; do not read blur at 60 min as failure of the loss.
3. **β = 1 vs β ∈ (1, 2)**: untested on images; keep β configurable.
4. **GPU throughput** is an estimate (§1.5). The first GPU job should time 100 steps and adjust `max_steps`.
5. **CIFAR-10 licence** is unstated by its authors; acceptable for research use as everywhere else, but
   note it if results are published.
6. **DiT is CC-BY-NC**: our code that imports it is research-only. If this ever matters, U-ViT (MIT) is
   the closest permissive backbone (SiT's MIT `models.py` is a relabelled DiT copy).

## Reproduce the measurements
```bash
cd /scratch/wg2381/transport_map
eval "$(conda shell.bash hook)" && conda activate /scratch/wg2381/.conda/envs/tmap-tex
# throwaway extras used here (not the project env):
#   conda install -p /scratch/wg2381/.conda/envs/tmap-tex --override-channels -c conda-forge torchvision pytorch=2.14.1
#   python -m pip install --target /scratch/wg2381/.cache-home/stage2_tmp/pylib --no-deps timm pyyaml huggingface_hub safetensors
#   python -m pip install --target /scratch/wg2381/.cache-home/stage2_tmp/pylib_pa --no-deps pyarrow
TIMM_DIR=/scratch/wg2381/.cache-home/stage2_tmp/pylib python -I docs/stage2_checks/time_dit.py           # ~10 min
D=/scratch/wg2381/.cache-home/stage2_tmp/dl
python -I docs/stage2_checks/data_scale.py --mnist $D/mnist --fmnist $D/fmnist --out docs/stage2_checks/results/data_scale_mnist.json
python -I docs/stage2_checks/data_scale.py --cifar10-parquet $D/cifar10_hf --pyarrow-dir /scratch/wg2381/.cache-home/stage2_tmp/pylib_pa --out docs/stage2_checks/results/data_scale_cifar10.json
TIMM_DIR=/scratch/wg2381/.cache-home/stage2_tmp/pylib python -I docs/stage2_checks/grad_var_m.py --fmnist $D/fmnist --out docs/stage2_checks/results/grad_var_m.json
```
