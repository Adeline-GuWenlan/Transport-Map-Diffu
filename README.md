# transport_map

One-step generative transport maps trained with the **energy score**, on a **DiT** backbone.

A class-conditional map F_θ(z, c) pushes Gaussian noise z ~ N(0, I) (image-shaped) to the data distribution of
class c in one forward pass. There is no denoising loss. F_θ is trained with the strictly proper energy score
(β = 1), whose expected value is ½·(energy distance to the data) + const:

    L(θ) = E_(Y,C) E_(Z,Z' iid N(0,I)) [ ‖F(Z,C) − Y‖ − ½ ‖F(Z,C) − F(Z',C)‖ ]

The derivation, and why the originally proposed loss collapses, are in `docs/energy_score_loss.{tex,pdf}`
(stage 1). The backbone, optimizer and dataset choices are in `docs/stage2_choices.md` (stage 2). The environment
and data are in `env/` and `data/README.md` (stage 3). The pipeline history is in `claude_jobs/standups/`.

**Status (2026-11-06, CPU only):** the pipeline runs end to end on a CPU node: tests, an overfit check, a
68-min smoke run on Fashion-MNIST and a 30-min CIFAR-10 look. See [Results](#results-cpu-2026-11-06). The real
CIFAR-10 run on GPU is written (`scripts/train_gpu.slurm`) but has not been run.

**In short:**
- The corrected energy-score loss trains the DiT map stably, with no collapse and no mode dropping.
- On Fashion-MNIST it gets class, silhouette and spread right: held-out D̂1 is 0.08, against 7.3 for the class
  mean.
- The pixel-space loss is nearly blind to fine structure. Samples carry a 4×4 patch mosaic. On CIFAR-10, images
  without any spatial structure already score better than real images of the wrong class.
- An energy distance on frozen Inception features reverses that ranking. Adding a feature-space term is the
  suggested next step (see Results).

## Layout
```
src/tmap/energy_score.py  loss: class-grouped unbiased U-statistic estimator (stage 1 eq. (9)), double-where norm;
                          held-out energy-distance estimators (float64 Gram form)
src/tmap/model.py         TransportMap: DiT (third_party/DiT, unmodified) with constant t = 999, F = DiT(z, t, c)
src/tmap/data.py          in-memory uint8 arrays, 28->32 padding, [-1, 1], class-stratified sampler (resumable)
src/tmap/train.py         YAML config + --set overrides, seed, AdamW + warmup + clip + EMA, checkpoint/resume,
                          metrics.jsonl (loss, confinement, interaction, grad norm, lr, it/s, img/s),
                          eval.jsonl (held-out per-class D1, raw + EMA), sample grids
src/tmap/sample.py        per-class sample grids with fixed z
src/tmap/evaluate.py      held-out D1 + baselines, nearest-train neighbours, diversity, class-confusion D1 and 1-NN
                          class agreement, patch blockiness, radial power spectrum, optional clean-fid FID
tests/                    pytest (24 tests)
configs/                  smoke_cpu.yaml (Fashion-MNIST, DiT-T/4), base_gpu.yaml (CIFAR-10, DiT-B/2),
                          smoke_cpu_cifar10.yaml, overfit_cpu.yaml (stage 4: one class, 256 images)
scripts/train_gpu.slurm   the real run (L40S, requeue-safe); not submitted
scripts/plot_run.py       curves and sample montages for a run directory
scripts/stage4_cpu_runs.sh the exact CPU runs of stage 4 (smoke, overfit, CIFAR-10, evals, blind-spot check, plots)
scripts/patch_shuffle_check.py  D1 of block-shuffled real images: what the pixel energy distance cannot see
runs/                     run outputs
```

## Setup
```bash
cd /scratch/wg2381/transport_map
bash env/setup_env.sh --lock               # only if /scratch/wg2381/.conda/envs/tmap is missing (stage 3)
source env/activate.sh                     # conda env tmap, caches on /scratch, PYTHONPATH=src, TMAP_DIT_DIR
python -m pip install -e . --no-deps --no-build-isolation
python -I scripts/prepare_data.py          # only if data/*/*.npy are missing (raw files: data/README.md)
python scripts/check_env.py                # versions, DiT clone pinned + clean, forward/backward, data, FID assets
```
Do not run `source env/activate.sh` under `set -u`.

## Test
```bash
source env/activate.sh
python -m pytest -q                        # 24 tests, ~40 s
```
The tests cover:
- `test_energy_score.py`: the U-statistic matches the closed-form 1-D Gaussian energy distance (4 cases), and the
  loss matches its closed-form expectation. Both estimators are unbiased at P = Q, while the V-statistic is
  biased. Grouping is per class with n_c/N weights and is order-invariant. The gradient is finite at F ≡ 0. The
  loss equals stage 1's reference `es_core.torch_es_loss_grouped`.
- `test_model.py`: shapes, F ≡ 0 at init (adaLN-Zero), 7.39 M params for the smoke model, and gradients reach
  every trainable parameter. `pos_embed` is DiT's frozen sin-cos table.
- `test_data.py`: padding, range, stratified batches, and sampler state restored exactly.
- `test_train.py`: 6 steps straight equal 3 steps + resume + 3 steps (model and EMA). It also checks the log fields.
- `test_evaluate.py`: baselines order correctly, the class-identity checks recognise real data, and blockiness
  detects a patch grid.
- `test_toy2d.py`: stage 1's 2-D 8-Gaussians toy (MLP, 3000 steps) reaches D1 < 0.03 in < 60 s (got 0.0077 in 17 s).

## Train
```bash
source env/activate.sh
# overfit check: one class (Sneaker), 256 fixed images; reference set = those images; 12 min
OMP_NUM_THREADS=7 python -m tmap.train --config configs/overfit_cpu.yaml --out runs/2026-11-06_overfit_fmnist_sneaker256 \
    --set num_threads=7
# CPU smoke run (Fashion-MNIST, DiT-T/4, 7.39 M params), 68 min wall clock, 7 threads
OMP_NUM_THREADS=7 python -m tmap.train --config configs/smoke_cpu.yaml --out runs/2026-11-06_smoke_fmnist_dit-t4 \
    --set train.time_limit_min=68 num_threads=7
# CIFAR-10 on the same small model, 30 min (an early look at the real dataset)
OMP_NUM_THREADS=7 python -m tmap.train --config configs/smoke_cpu_cifar10.yaml --out runs/2026-11-06_smoke_cifar10_dit-t4 \
    --set train.time_limit_min=30 num_threads=7
# all of the above plus the evaluations and plots, as run: bash scripts/stage4_cpu_runs.sh smoke overfit eval_smoke ...
# GPU (not run yet): CIFAR-10, DiT-B/2, 200k steps, bf16, requeue-safe. Recommended first: a 30k-step pilot
TRAIN_SET="train.max_steps=30000" RUN=runs/gpu_pilot_cifar10_dit-b2 sbatch scripts/train_gpu.slurm
sbatch scripts/train_gpu.slurm
```
- `--set key.sub=value` overrides any config entry; values are parsed as YAML.
- Re-running the same command with the same `--out` resumes from `<out>/ckpt/latest.pt`. SIGTERM or SIGUSR1
  writes a checkpoint and exits with code 3.
- Outputs per run: `config.yaml`, `train.log`, `metrics.jsonl`, `eval.jsonl`, `samples/step*_{raw,ema}.png`
  (rows = classes, 8 fixed z each), and `ckpt/latest.pt`.
- The gradient-norm column is measured before clipping. With clip 2.0 it binds on every step, as stage 2 measured
  and expected.
- **CPU threads:** do not run anything else in the same 8-CPU allocation while training with 8 threads. One
  competing process cut throughput from 4.9 to 0.75 it/s. The smoke run used 7 threads and left one core free.

## Sample
```bash
python -m tmap.sample --ckpt runs/2026-11-06_smoke_fmnist_dit-t4/ckpt/latest.pt \
    --out runs/2026-11-06_smoke_fmnist_dit-t4/samples_final_ema.png --weights ema --n-per-class 10
```

## Evaluate
```bash
python -m tmap.evaluate --ckpt runs/2026-11-06_smoke_fmnist_dit-t4/ckpt/latest.pt \
    --out runs/2026-11-06_smoke_fmnist_dit-t4/eval_final.json --n-per-class 1000 \
    --nn-grid runs/2026-11-06_smoke_fmnist_dit-t4/nearest_neighbours.png
python scripts/plot_run.py runs/2026-11-06_smoke_fmnist_dit-t4
# CIFAR-10 with an indicative CPU FID from 2000 samples (clean-fid, legacy_tensorflow, train stats)
python -m tmap.evaluate --ckpt runs/2026-11-06_smoke_cifar10_dit-t4/ckpt/latest.pt \
    --out runs/2026-11-06_smoke_cifar10_dit-t4/eval_final.json --n-per-class 1000 --fid-num-gen 2000
```
- **Metric:** the held-out unbiased energy distance D̂1 per class in pixel space, against the test split
  (first n images of each class), averaged over classes. A single estimate can be slightly < 0; its expectation is
  ≥ 0, with equality iff model = data.
- **Baselines:** N(0, I) noise (ceiling), the zero image (the model at step 0), a class-mean point mass, train
  images of the wrong class, and train images of the right class (floor).
- **Also reported:** nearest-train-neighbour distances of samples vs of test images (memorisation check), and the
  diversity ratio U-mean‖x−x'‖ / U-mean‖y−y'‖ per class (1 = data-like spread, ≪ 1 = collapse).
- **FID** (CIFAR-10 only, clean-fid `legacy_tensorflow` vs train stats): pass `--fid-num-gen N`. On CPU with small
  N it is indicative only. Fashion-MNIST has no standard FID reference, so FID is off for the smoke run.

## Results (CPU, 2026-11-06)
All numbers come from this CPU node (8 CPUs, no GPU) and are reproducible with `scripts/stage4_cpu_runs.sh`.
D̂1 is the held-out per-class unbiased energy distance in pixel space ([−1, 1], 32×32), averaged over classes.

### Fashion-MNIST smoke run: `runs/2026-11-06_smoke_fmnist_dit-t4/`
DiT-T/4 (7.39 M params). 68.3 min, 16,624 steps of batch 8 classes × 8, 4.3 it/s median on 7 threads, AdamW 2e-4,
EMA 0.999.

| | held-out D̂1 (1000/class) |
|---|---|
| zero image (model at step 0) | 38.01 |
| N(0, I) noise | 22.03 |
| wrong-class train images | 12.61 |
| class-mean point mass | 7.31 |
| **model, EMA, step 16624** | **0.082** |
| model, raw weights | 0.160 |
| right-class train images (floor) | 0.0006 |

- **Training lowers the held-out energy distance.** Values at 500/class during training (raw / EMA):
  38.06 → 0.55 / 6.08 (step 2k) → 0.37 / 0.41 (4k) → 0.18 / 0.19 (6k) → 0.19 / 0.12 (10k) → 0.15 / 0.068
  (16.6k). It was still falling at the end.
- **Loss.** 27.1 at the start, 8.81 at the end (mean of the last 500 steps). The optimum is ½·E‖Y−Y′‖ = 8.71, so the
  gap ≈ ½·D on training batches.
- **Pre-clip gradient norm.** It settles at about 4, so clip 2.0 binds on every step, as stage 2 measured.
- **Classes are right.** The D̂1 matrix of generated class × test class has its row minimum on the right class for
  10/10 classes. A pixel 1-NN classifier, fitted on 1000 train images per class, assigns 72.9% of samples to their
  conditioning class (real test images: 81.0%).
- **No collapse, no memorisation.**
  - The within-class spread ratio U-mean‖x−x′‖ / U-mean‖y−y′‖ is 0.97–1.02 for every class.
  - Per-class D̂1 ranges from 0.044 (Sneaker) to 0.116 (Bag).
  - A sample's median distance to its nearest train image is 10.6, against 7.0 for real test images; no sample is
    closer than the 5th percentile of test images.
  - In `nearest_neighbours.png` the samples match their neighbours in silhouette and brightness but are not
    copies.
- **Patch artifacts: the main failure.** Samples have the right silhouettes but their interiors are a 4×4-patch
  mosaic (DiT patch size 4).
  - Blockiness (mean pixel step across patch boundaries / inside patches) is 2.20 for samples and 1.05 for test
    images. It fell from 3.28 at step 4000.
  - The power spectrum (gen / data) by band, in cycles per image: 1–4: 0.97, 4–8: 1.22, 8–12: 1.11, 12–17: 0.84.
  - 33% of generated pixels lie outside [−1, 1] (mostly the black background, which is exactly −1 in the data).
    The output is not squashed.
- Figures:
  - `fig_curves.png`: loss with its optimum, pre-clip gradient norm, held-out D̂1 vs baselines.
  - `fig_samples_over_steps_{ema,raw}.png`.
  - `fig_final_vs_real_ema.png`: final EMA samples beside real test images, same layout.
  - `samples_final_ema_10x10.png`, `nearest_neighbours.png`, `samples/step*_{raw,ema}.png`.
  - `eval_final.json`, `eval.jsonl`, `metrics.jsonl`.

### Overfit check: `runs/2026-11-06_overfit_fmnist_sneaker256/`
One class (Sneaker), 256 fixed training images, batch 64 drawn from them, 12 min (2802 steps). The reference set
for D̂1 is the same 256 images.

| | D̂1 vs the 256 images |
|---|---|
| zero image (step 0) | 44.72 |
| N(0, I) noise | 27.49 |
| class-mean point mass | 5.71 |
| **model, raw, step 2802** | **0.079** |
| model, EMA | 0.58 (EMA 0.999 lags in a 2.8k-step run) |
| the 256 images themselves | −0.11 (U-statistic with identical sets) |

- The loss drops clearly, from 28.3 to 7.01; the optimum is ½·E‖Y−Y′‖ = 6.95.
- The samples approach the data in shape: right silhouette and brightness, and the within-set spread ratio is 0.99.
- They do **not** reproduce individual images in 12 min. The median distance to the nearest of the 256 targets
  is 9.7, against 0 for a copy. The same 4×4 patch mosaic is there: blockiness 3.08, data 1.08.
- Figure: `fig_raw_samples_vs_nearest_target.png`, rows 1 and 3 samples, rows 2 and 4 their nearest targets.

### What the pixel energy distance can and cannot see: `runs/2026-11-06_loss_blindspot/`
`scripts/patch_shuffle_check.py` involves no model. It builds images whose p×p blocks each come from an
independent random train image of the same class. Every block's marginal distribution is exact, and all
dependence between blocks (image structure) is destroyed. Their held-out D̂1 (1000/class):

| block size p | 16 | 8 | 4 | 2 | 1 (independent pixels) | wrong-class real images | right-class real images |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.190 | 0.240 | 0.272 | 0.289 | 0.295 | 12.61 | 0.0006 |
| CIFAR-10 | 0.214 | 0.327 | 0.397 | 0.422 | 0.431 | 2.06 | 0.011 |

- On CIFAR-10, images with no spatial structure at all score **5× better than real images of the wrong class**.
- Once per-pixel (per-class) marginals are right, the remaining signal for structure is small: D̂1 ≈ 0.43 →
  0.01, against a loss value of about 18. So the pixel energy score gives a weak gradient toward image structure.
- The Fashion-MNIST smoke model (0.082) is **below** the shuffled levels (0.27–0.30), so it has learned real
  dependence between patches. A CIFAR-10 model needs D̂1 well below about 0.2 before its samples can be expected
  to have structure.

**Pixel vs Inception-feature energy distance** (`scripts/feature_space_check.py`, CIFAR-10, 100/class, reference =
test set; Inception = clean-fid's legacy_tensorflow pool3 extractor, 2048-d; output
`runs/2026-11-06_loss_blindspot/cifar10_feature_space.json`):

| set | pixel D̂1 | Inception D̂1 |
|---|---|---|
| right-class real images | 0.024 | −0.002 |
| wrong-class real images | 1.99 | 3.27 |
| 4×4-block shuffled | 0.42 | 13.07 |
| pixel shuffled | 0.47 | 15.82 |
| CIFAR-10 CPU model (EMA, clamped) | **0.18** | **18.22** |

The ranking flips. Pixel space rates the structureless model as the best of the fakes. Inception space rates it
the worst and prefers real images of the wrong class. A feature-space energy-score term would see exactly what the
pixel term misses.

### CIFAR-10 on the small model (CPU, 30 min): `runs/2026-11-06_smoke_cifar10_dit-t4/`
`configs/smoke_cpu_cifar10.yaml` (DiT-T/4, 3 channels, hflip), 6,921 steps. This is an early look at the real
dataset, **not** the real run (that is DiT-B/2, patch 2, on GPU).

| | held-out D̂1 (1000/class) |
|---|---|
| zero image (step 0) | 17.85 |
| class-mean point mass | 14.78 |
| N(0, I) noise | 9.22 |
| wrong-class train images | 2.06 |
| independent-pixel shuffle (above) | 0.43 |
| 16×16-block shuffle (above) | 0.21 |
| **model, EMA, step 6921** | **0.181** |
| model, raw | 0.202 |
| right-class train images (floor) | 0.011 |

- D̂1 was still falling at the end (500/class, raw): 0.478 (2k) → 0.478 (4k) → 0.283 (6k) → 0.189 (6.9k).
  The loss is 18.4, against an optimum of 18.20.
- **Samples have no object structure.** They are 4×4-patch colour mosaics with a coherent per-image palette and
  weak per-class colour tendencies; see `fig_final_vs_real_ema.png`.
  - Blockiness 7.3 (data 1.02).
  - Power ratio gen/data, 12–17 cycles/image: 3.3.
  - **FID-2k (clean-fid legacy_tensorflow, CPU, indicative only): 419 (EMA), 391 (raw).**
- **Class signal is weak but present.** The D̂1-confusion row argmin is right for 9/10 classes (EMA) and 10/10
  (raw). Pixel 1-NN agreement is 20% (real test images: 28.8%; chance: 10%).
- Pre-clip gradient norm: about 3.5 until step 4.5k, then about 6.
- **Reading:** the model beats the block-shuffle thresholds on D̂1 while its samples look structureless. It has
  matched per-pixel colour statistics and whole-image palette coherence, which block shuffling destroys. On
  CIFAR-10, pixel D̂1 is not a usable quality measure on its own. Whether DiT-B/2 with patch 2 and 200k steps gets
  past this is the open question for the GPU run.



## Licences
- `third_party/DiT`: CC-BY-NC 4.0, so this project's code is research-only.
- Fashion-MNIST: MIT. CIFAR-10: no stated licence (cite Krizhevsky 2009). See `data/README.md`.
