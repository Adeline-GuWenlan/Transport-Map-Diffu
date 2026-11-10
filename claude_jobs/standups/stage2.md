Status: DONE

# Stage 2 standup: backbone, optimizer, dataset

Written 2026-11-06 ~08:05 EST by the stage-2 agent (Slurm job 22047086). Attempt 1 (06:15) hit the usage
limit before doing anything; all work below is from attempt 2 (07:23–08:05). Stage 1 is DONE
(`claude_jobs/standups/stage1.md`); I depended on its loss and estimator unchanged. Full reasoning, citations
and tables: **`docs/stage2_choices.md`**. Measurements: `docs/stage2_checks/` (scripts, `results/*.json`, `logs/`).
Configs: **`configs/smoke_cpu.yaml`**, **`configs/base_gpu.yaml`**, plus optional `configs/smoke_cpu_cifar10.yaml`.

## Decision table
| Choice | Decision | Alternatives considered | Why |
|---|---|---|---|
| Backbone code | facebookresearch/DiT `models.py` @ `ed81ce2`, imported unmodified from `third_party/DiT` (CC-BY-NC 4.0) | SiT (MIT, but its `models.py` is a 3-line-diff copy of DiT's), fast-DiT (CC-BY-NC, latent speed-ups), U-ViT (MIT), LightningDiT (MIT), JiT (MIT) | CLAUDE.md fixes DiT; it is one file and needs only timm; the others add nothing a one-step pixel map needs |
| Map F_θ(z, c) | `DiT(input_size=32, in_channels=C, learn_sigma=False, class_dropout_prob=0)`; z ~ N(0, I) of shape C×32×32; t-embedder kept, constant t = 999; output used directly (zero-init head: F ≡ 0 at step 0, so use stage 1's double-where norm); no extra noise channels | drop t (needs an upstream edit); residual z + DiT(z); CFG | no edits to upstream; t stays available for the per-noise-level follow-up. For that follow-up (input [x_t, ξ]) stage 4's wrapper should allow `out_channels != in_channels` by replacing `final_layer` |
| Space | pixel, [−1, 1], 32×32 | SD-VAE latents (32×32×4 at 256 px) | SD-VAE turns 32 px into 4×4×4; latents need ImageNet-256 (account) and a VAE in every eval; TBSM reports raw-pixel energy distance working on small images |
| CPU smoke model | "DiT-T/4" (depth 6, hidden 256, 4 heads, patch 4): **7.39 M params, 0.318 GMACs/sample, 0.138 s/step at batch 64** on 8 threads | DiT-S/4 0.61 s, DiT-T/2 0.64 s, DiT-U/4 0.06 s | well under 1 s; S/4 (smallest table size) is not |
| GPU model | **DiT-B/2** on 3×32×32: **129.7 M params, 23.0 GMACs/sample** | DiT-S/2 (32.6 M, 6.05 GMACs) fallback | DiT-B/2 latent-setup compute (256 tokens); our FLOP counter reproduces DiT Table 4 (S/2 6.05 vs 6.06, B/2 23.00 vs 23.01, XL/2 118.62 vs 118.64) |
| Optimizer | AdamW β = (0.9, 0.95), eps 1e-8, wd 0, linear warmup then constant, clip 2.0, EMA | DiT: AdamW 1e-4, wd 0, b256, EMA 0.9999, no clip/warmup (**all verified** in paper + `train.py`); Drifting: β2 0.95, wd 0.01, lr 2e-4–4e-4, clip 2, warmup; IMM: AdamW 1e-4, b4096, EMA 0.9999; TBSM: 1e-5 (fine-tune) | DiT recipe + Drifting's β2/clip/warmup (closest: one-step DiT, class-grouped kernel loss) |
| lr / EMA | GPU 1e-4, warmup 5k, EMA 0.9999, 200k steps, bf16 net + fp32 distances. CPU 2e-4, warmup 500, EMA 0.999, 60 min | | DiT value for DiT-B; faster for the 7 M smoke model |
| Batch n × m | class-stratified; r = 1 (m_c = n_c). GPU 8 classes × 64 = 512. CPU 8 × 8 = 64 | r = 2, 4, 8; engression pairs; random batches | measured: z-noise is only ~30% of gradient noise at r = 1, so more data beats more samples; pairs are 2.5× noisier than per-class pooling at equal cost |
| Smoke data | **Fashion-MNIST** (MIT), 28→32 zero-pad, 1 ch; MNIST fallback | MNIST, CIFAR-10 | held-out floor 0.0004 (MNIST 0.036); class identity is ~56% of its pixel energy-distance scale (CIFAR-10: ~6%) |
| Real data | **CIFAR-10** 32×32 RGB, hflip | CIFAR-100, ImageNet 32/64 (account/terms), Tiny ImageNet | open, no account, standard FID stats, same format as smoke |
| FID | clean-fid 0.1.35, `mode="legacy_tensorflow"`, precomputed `cifar10` **train** 32 stats; FID-10k during, FID-50k at end (EMA) | pytorch-fid, torch-fidelity, EDM npz (cross-check) | nothing to compute on CPU; matches published convention |

## Key measurements (all on this node; details in docs/stage2_choices.md)
- **Timing/FLOPs** (`results/dit_sizes.json`): table above. ~7 steps/s for the smoke model, so ~25k steps in 60 min.
- **Loss scale / baselines** (`results/data_scale_*.json`; held-out unbiased D̂₁ per class, n = 1000, mean over classes):

  | | Fashion-MNIST | MNIST | CIFAR-10 |
  |---|---|---|---|
  | N(0, I) noise | 22.02 | 26.03 | 9.22 |
  | zero (DiT at step 0) | 38.01 | 44.22 | 17.85 |
  | class-mean point mass | 7.31 | 7.57 | 14.78 |
  | wrong-class train images | 12.68 | 6.09 | 2.14 |
  | right-class train images (floor) | 0.0004 | 0.036 | 0.005 |
  | E‖Y−Y′‖ within class | 17.43 | 17.92 | 36.41 |

  The raw loss sits near ½E‖Y−Y′‖ at the optimum. Log held-out D̂₁, not the loss.
- **Gradient noise vs r** (`results/grad_var_m.json`, DiT-T/4 after 400 warm-up steps; tr Cov of the param
  gradient): z-only r = 1/2/4/8 → 11.6 / 6.4 / 2.9 / 1.8 (≈ 1/r); engression pairs at r = 2: 15.9; data + z at
  r = 1: 39.3. Mean-gradient ‖·‖² ≈ 230.
- **Clipping** (`results/warm_clip_*.json`): the pre-clip norm is 3–67, so clip 2.0 binds on every step. The
  loss curve with and without it is the same (Adam is scale-invariant). Kept as normalisation/spike cap.
  Stage 4: a clip rate of 100% is expected, not a bug.

## Packages stage 3 must install (env `/scratch/wg2381/.conda/envs/tmap`)
```
conda create -p /scratch/wg2381/.conda/envs/tmap --override-channels -c conda-forge python=3.11 pip
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1 torchvision==0.29.1   # cp311 wheels verified on the index
pip install timm==1.0.31 numpy scipy pyyaml pillow matplotlib tqdm pyarrow pytest requests          # timm pulls huggingface_hub, safetensors
pip install --no-deps clean-fid==0.1.35                                                             # --no-deps so it cannot touch torch
```
- Why cu126: older driver requirement than cu130. No cu128 wheel exists for 2.14.x. **Assumption:** the GPU
  nodes' driver supports CUDA 12.6.
- timm 1.0.31 + DiT `ed81ce2` + torch 2.14.1 were tested together here.
- Not needed now: diffusers, accelerate (SD-VAE only for a later latent test).
- Importing `third_party/DiT/models.py` writes `third_party/DiT/__pycache__/`. Set `PYTHONDONTWRITEBYTECODE=1`
  in `env/activate.sh` to keep the clone clean (`git status --porcelain` empty). I deleted the cache I created.

## Repos
| Repo | Commit | License | Status |
|---|---|---|---|
| https://github.com/facebookresearch/DiT | `ed81ce2229091fd4ecc9a223645f95cf379d582b` | CC-BY-NC 4.0 | **cloned** at `third_party/DiT` (depth 1, clean); recorded in `third_party/README.md` |

No other clone is needed. For reference only (not to clone): TBSM `sp12138/TBSM@d1d43f5` (Apache-2.0 + NSCLv1
part), IMM `lumalabs/imm@e124a9d` (CC BY-NC-SA 4.0), engression `xwshen51/engression@eb6b83b` (BSD-3), SiT
`willisma/SiT@cbde832` (MIT), clean-fid `GaParmar/clean-fid@e88c4d6` (MIT; installed from PyPI instead).

## Datasets: ids and download commands (sha256 measured here; full block in docs/stage2_choices.md §3.3)
```
# Fashion-MNIST (smoke) -> data/fashion_mnist/raw/   (also: torchvision.datasets.FashionMNIST, HF zalando-datasets/fashion_mnist)
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  curl -sSLO http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/$f; done
#  3aede38d…a84 train-images | a04f1713…845 train-labels | 346e55b9…073 t10k-images | 67da17c7…dd5 t10k-labels
# MNIST (fallback) -> data/mnist/raw/: same 4 names from https://ossci-datasets.s3.amazonaws.com/mnist/
# CIFAR-10 (real) -> data/cifar10/raw/: HF uoft-cs/cifar10 @ 0b2714987fa478483af9968de7c934580d0bb9a2
R=https://huggingface.co/datasets/uoft-cs/cifar10/resolve/0b2714987fa478483af9968de7c934580d0bb9a2/plain_text
curl -sSLO $R/train-00000-of-00001.parquet   # sha256 8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883
curl -sSLO $R/test-00000-of-00001.parquet    # sha256 841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de
# FID refs (clean-fid fetches these itself; pre-fetch): https://www.cs.cmu.edu/~clean-fid/stats/cifar10_legacy_tensorflow_train_32.npz,
#   …/cifar10_clean_train_32.npz; Inception: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt
```
- Checked: 60k/10k with 6000/1000 per class (Fashion-MNIST), 50k/10k with 5000/1000 per class (CIFAR-10).
- HF downloads 143 MB in 2 s. The official Toronto server (torchvision `CIFAR10`) redirects to
  cave.cs.toronto.edu and ran at ~140 kB/s (~20 min); avoid it.
- Parquet rows are `{img: {bytes: PNG}, label}`; a numpy/pyarrow/PIL decoder is `load_cifar10_parquet` in
  `docs/stage2_checks/data_scale.py`. No pickle is needed anywhere.
- Suggested on-disk form: `data/<name>/{train,test}_{images,labels}.npy` (uint8 N×C×H×W at native size, int64
  labels); pad 28→32 in the loader.

## Evaluation plan (docs/stage2_choices.md §3.4)
- Per-class held-out D̂₁ against the test set, with the baselines above.
- FID: CIFAR-10 only; CPU numbers are indicative.
- Per-class 10×10 grids with fixed z, raw and EMA weights.
- Nearest-train-neighbour distances vs test-to-train distances, plus a grid of the closest pairs.
- Within-class diversity vs E‖Y−Y′‖.

## Open questions / risks
1. **Class signal in CIFAR-10 pixel space is weak**: a wrong-class D̂₁ of 2.1, against a noise scale of 9–36.
   Drifting could not train on ImageNet without a feature encoder; TBSM uses frozen encoders for natural
   images. If CIFAR-10 samples get global statistics right but not classes, the next step is a feature-space
   or multi-scale energy distance (doc §6).
2. GPU step time is an estimate (~0.5–0.7 s/step for DiT-B/2 at N = 512, so 200k steps ≈ 1–1.5 days). The GPU
   job must resume after requeue and time its first 100 steps.
3. β = 1 vs β ∈ (1, 2) on images is untested; keep it configurable.
4. De Bortoli et al.'s image hyperparameters (App. I.2) could not be retrieved (truncated HTML): unverified.
5. CIFAR-10 has no stated licence (cite Krizhevsky 2009). DiT is CC-BY-NC, so this project's code is
   research-only.

## Throwaway items outside the project (not needed by later stages)
- `/scratch/wg2381/.cache-home/stage2_tmp/`: timm and pyarrow `--target` dirs, downloaded data, and the SiT
  `models.py` used for the diff.
- torchvision 0.29.2 was added to the stage-1 helper env `tmap-tex` (conda-forge).
