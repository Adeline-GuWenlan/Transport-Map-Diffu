Status: DONE

# Stage 4 standup: the pipeline runs end to end

Written 2026-11-06 ~19:05–19:55 EST by the stage-4 agent (Slurm job 22047092, cs636, 8 CPUs, no GPU). An earlier
attempt of this stage left nothing on disk; all work is from this attempt (17:23 onward).

**Earlier stages.** Stage 3's standup says INCOMPLETE (it is a draft), but everything stage 4 depends on is on disk
and works, so stage 4 redid none of it:
- the `tmap` env and `env/activate.sh`;
- `third_party/DiT` at ed81ce2 (still clean after all runs: `git status --porcelain` empty);
- `data/*/{train,test}_*.npy`, with baselines that reproduce stage 2's numbers;
- the FID references (the clean-fid path was exercised on CPU).

Still open from stage 3: `data/README.md` contains `CROSSCHECK_PLACEHOLDER` (the HF vs official CIFAR-10
comparison; `data/cifar10/official_crosscheck.json` exists) and `FID_PLACEHOLDER`. Stage 4 does not depend on
either and did not edit stage 3's file.

Stages 1 and 2 are DONE. The loss and estimator are implemented exactly as stage 1 boxed them, with stage 2's model,
optimizer and data choices.

## Does it run? Yes
```bash
cd /scratch/wg2381/transport_map && source env/activate.sh && python -m pip install -e . --no-deps --no-build-isolation
python -m pytest -q
OMP_NUM_THREADS=7 python -m tmap.train --config configs/smoke_cpu.yaml --out runs/2026-11-06_smoke_fmnist_dit-t4 \
    --set train.time_limit_min=68 num_threads=7
python -m tmap.evaluate --ckpt runs/2026-11-06_smoke_fmnist_dit-t4/ckpt/latest.pt \
    --out runs/2026-11-06_smoke_fmnist_dit-t4/eval_final.json --n-per-class 1000 --nn-grid runs/2026-11-06_smoke_fmnist_dit-t4/nearest_neighbours.png
python scripts/plot_run.py runs/2026-11-06_smoke_fmnist_dit-t4
```
Every CPU run, as run, is in `scripts/stage4_cpu_runs.sh`. The README has setup, test, train, sample and evaluate
commands.

What was built (`src/tmap/`, installable with `pip install -e .`):
- **`energy_score.py`**
  - The loss is the class-grouped unbiased U-statistic of stage 1 eq. (9): β = 1, interaction weight ½,
    same-class pairs only, the diagonal excluded by index, n_c/N weights, the double-where norm, fp32 distances.
    It equals stage 1's `es_core.torch_es_loss_grouped` to 1e-14.
  - Eval estimators use a float64 Gram form.
- **`model.py`**: `TransportMap` = unmodified DiT with constant t = 999 and learn_sigma off. It supports
  `out_channels != in_channels` via a new zero-init head, for the later noise-level variant.
- **`data.py`**: in-RAM uint8 arrays and a class-stratified sampler with resumable RNG.
- **`train.py`**:
  - YAML config plus `--set` overrides, seeds;
  - AdamW, warmup, clip, EMA, bf16 autocast on GPU;
  - atomic checkpoints and exact resume (tested);
  - SIGTERM/USR1 → checkpoint → exit 3 (tested);
  - `metrics.jsonl` (loss, confinement, interaction, pre-clip grad norm, lr, it/s, img/s), `eval.jsonl`
    (held-out D̂1, raw + EMA), sample grids.
- **`sample.py`**: per-class grids with fixed z.
- **`evaluate.py`**:
  - held-out per-class D̂1 against 5 baselines;
  - nearest-train neighbours and a neighbour grid;
  - diversity ratio;
  - a generated-class × test-class D̂1 confusion matrix and pixel 1-NN class agreement;
  - patch blockiness and a radial power spectrum;
  - clean-fid FID (CIFAR-10 only; CPU = indicative).
- **`scripts/plot_run.py`** and **`scripts/train_gpu.slurm`** (not submitted).

## Tests: 24 passed
`python -m pytest -v` → **24 passed in 126 s** (8 threads; log `runs/pytest_2026-11-06.log`).
- **Estimator:**
  - matches the closed-form 1-D Gaussian energy distance in 4 cases;
  - the loss's expectation matches its closed form (1-D, and 2-D at P = Q with m = 3), so it is unbiased;
  - the U-statistic is unbiased at P = Q while the V-statistic is visibly biased;
  - per-class grouping, n_c/N weights, order invariance, not pooled;
  - r > 1 samples per data point;
  - finite gradient at F ≡ 0;
  - equals the stage 1 reference;
  - Gram-form eval distances are exact.
- **Model:** shapes (1 and 3 channels, out ≠ in), F ≡ 0 at init, 7.39 M params, gradients reach every trainable
  parameter (`pos_embed` is DiT's frozen sin-cos table), and the output depends on c and z.
- **Data:** padding to −1, range, stratification, exact sampler resume, subset and hflip.
- **Train:** 6 steps equal 3 steps + resume + 3 steps (model and EMA), and the log fields are present.
- **Evaluate:** the baselines order correctly, the class checks recognise real data, and blockiness detects a
  patch grid.
- **2-D toy** (stage 1 check c): D̂1 = 0.0077 in 17 s on 1 thread. The source Gaussian scores 0.248; stage 1 got
  0.0098.

## Smoke run: `runs/2026-11-06_smoke_fmnist_dit-t4/`
Fashion-MNIST, DiT-T/4 (7.39 M params), batch 8 classes × 8, AdamW 2e-4, EMA 0.999. **68.3 min, 16,624 steps** at
4.3 it/s median, 7 threads.

Held-out D̂1, test set, mean over 10 classes:

| | D̂1 |
|---|---|
| zero image = model at step 0 | 38.01 |
| N(0, I) noise (ceiling) | 22.03 |
| wrong-class train images | 12.61 |
| class-mean point mass | 7.31 |
| **model EMA, final (1000/class)** | **0.082** |
| model raw, final | 0.160 |
| right-class train images (floor) | 0.0006 |

During training (500/class, raw / EMA):

| step | 0 | 2k | 4k | 6k | 8k | 10k | 12k | 14k | 16.6k |
|---|---|---|---|---|---|---|---|---|---|
| raw | 38.06 | 0.548 | 0.373 | 0.185 | 0.221 | 0.189 | 0.186 | 0.142 | 0.148 |
| EMA | 38.06 | 6.08 | 0.408 | 0.186 | 0.133 | 0.117 | 0.101 | 0.080 | 0.068 |

- **Class identity:**
  - Class confusion by D̂1: the row argmin is the right class for 10/10 classes.
  - A pixel 1-NN classifier gives the conditioning class for 72.9% of samples (real test images: 81.0%).
- **Spread:**
  - The diversity ratio per class is 0.97–1.02.
  - The nearest-train distance has median 10.6 (test images: 7.0); 0% of samples are closer than the test
    images' 5th percentile.
- **Patch artifacts:** blockiness is 2.20 (data 1.05). The power ratio gen/data by band (cycles per image) is
  1–4: 0.97, 4–8: 1.22, 8–12: 1.11, 12–17: 0.84.
- **Loss:** 27.1 → 8.81, against an optimum of ½·E‖Y−Y′‖ = 8.71. The pre-clip grad norm settles at about 4, so the
  2.0 clip binds every step, as stage 2 predicted.
- **Figures** (all viewed):
  - `fig_curves.png`: loss with its optimum line, grad norm, held-out D̂1 vs the 5 baselines (log scale).
  - `fig_samples_over_steps_{ema,raw}.png`: grids at 6 steps.
  - **`fig_final_vs_real_ema.png`**: final EMA samples beside real test images, same layout, 3× upscaled.
  - `nearest_neighbours.png`, `samples_final_ema_10x10.png`, `samples/step*_{raw,ema}.png`.

## Overfit check: `runs/2026-11-06_overfit_fmnist_sneaker256/`
One class (Sneaker), 256 fixed images, batch 64, 12 min (2802 steps); `configs/overfit_cpu.yaml`.
- **D̂1 against the 256 images:** 44.72 at step 0, **0.079** (raw) at the end. Class-mean point mass: 5.71; the
  images themselves: −0.11.
- **Loss:** 28.3 → 7.01, against an optimum of 6.95.
- **Samples:** sneaker-shaped, with the right brightness and spread (ratio 0.99). They do **not** reproduce the
  256 images: the median distance to the nearest target is 9.7, where a copy would be 0. The same 4×4 mosaic is
  present: blockiness 3.08, data 1.08.
- Figure: `fig_raw_samples_vs_nearest_target.png`.

## CIFAR-10 on the small model (CPU, 30 min): `runs/2026-11-06_smoke_cifar10_dit-t4/`
`configs/smoke_cpu_cifar10.yaml` (DiT-T/4), 6,921 steps. An early look only; the real run is DiT-B/2 on GPU.
- **Held-out D̂1 (1000/class):** EMA **0.181**, raw 0.202. Baselines: noise 9.22, class mean 14.78, wrong class
  2.06, independent-pixel shuffle 0.43, 16×16-block shuffle 0.21, floor 0.011. D̂1 was still falling:
  0.48 → 0.28 → 0.19 at 2k/6k/6.9k.
- **Samples have no object structure.** They are 4×4-patch colour mosaics with a coherent per-image palette and
  weak per-class colour tendencies.
  - Blockiness 7.3 (data 1.02); high-band power 3.3× the data's.
  - **CPU FID-2k: 419 (EMA) / 391 (raw)**, indicative only.
- **Class signal:** the D̂1-confusion argmin is right for 9/10 classes (EMA) and 10/10 (raw). 1-NN agreement is 20%
  (real 28.8%, chance 10%).
- The pre-clip grad norm rose from about 3.5 to 6 after step 4.5k.
- **Reading:** below the shuffle thresholds on D̂1, yet visually structureless. Pixel D̂1 is not a quality measure
  on CIFAR-10 by itself, which confirms stage 2's risk 1.
- Figures: `fig_final_vs_real_ema.png` (viewed), `fig_curves.png` (viewed), `fig_samples_over_steps_*.png`,
  `nearest_neighbours.png`.

## What the results say about the loss
1. **The corrected energy-score loss works as a training signal for a one-step DiT map.**
   - It trains stably from the zero-init DiT: no NaN, thanks to the double-where norm.
   - The held-out energy distance falls by 2.7 orders of magnitude, to about 1% of the class-mean point mass
     (D̂1 ≈ 0.08 against 7.31) and 0.6% of the wrong-class baseline. It was still falling at the end of the run.
   - Class conditioning works (per-class grouping).
   - There is **no collapse and no mode dropping visible**: the diversity ratio is about 1 for every class, and
     samples within a class vary in shape and brightness, matching distinct train images.
   - So the failure stage 1 predicted for the user's literal formula (collapse to a point) does not happen.
   - On CIFAR-10 (30 min CPU, small model) the loss also trains stably and lowers D̂1 (17.9 → 0.18). Samples get
     colour statistics, not objects.
2. **It is weak on fine structure, as stage 1 warned** (the ‖t‖^(−d−β) weighting favours low frequencies):
   - Samples get silhouettes, brightness and class right, but their interiors are a 4×4-patch mosaic that matches
     DiT's patch size, not fabric or edges. The pixel energy distance is nearly blind to it: D̂1 ≈ 0.08 while the
     patch grid is obvious.
   - A plausible mechanism (my reading, not tested): each output patch is driven mostly by its own input-noise
     patch, and per-patch variance that "looks right" in L2 is cheap under the energy distance.
   - Blockiness was still falling (3.28 at step 4k, 2.20 at 16.6k), so more steps help, but slowly.
3. The samples are not blur in the regression-to-the-mean sense: the spread is right. They are wrong-texture noise
   at the patch scale.
4. **Measured blind spot of the pixel energy distance** (`runs/2026-11-06_loss_blindspot/`,
   `scripts/patch_shuffle_check.py`, no model involved).
   - Method: real images whose p×p blocks each come from an independent random same-class train image. Marginals
     are exact; structure is destroyed.
   - CIFAR-10 D̂1 by block size: p = 16: 0.214, 8: 0.327, 4: 0.397, 2: 0.422, 1 (independent pixels): 0.431.
     Wrong-class real images score 2.06; the floor is 0.011.
   - Fashion-MNIST: 0.190 / 0.240 / 0.272 / 0.289 / 0.295 (wrong class 12.61, floor 0.0006).
   - So on CIFAR-10, structureless images with the right per-pixel statistics score **5× better than real images
     of the wrong class**. Once marginals are right, the signal left for structure is a D̂1 gap of about 0.4,
     against a loss of about 18.
   - The Fashion-MNIST smoke model (0.082) is below the shuffled levels, and its silhouettes show it has learned
     real dependence between patches.
   - The CIFAR-10 CPU model (0.18) is also below every shuffled level, yet its samples have no visible structure.
     Whole-image palette coherence, which block shuffling destroys, is enough to beat these thresholds. On
     CIFAR-10, beating them is necessary but not sufficient.
5. **A frozen feature space sees what pixels miss** (`scripts/feature_space_check.py`, CIFAR-10, 100/class,
   Inception pool3 via clean-fid; `runs/2026-11-06_loss_blindspot/cifar10_feature_space.json`).

   | set | pixel D̂1 | Inception D̂1 |
   |---|---|---|
   | right-class real | 0.024 | −0.002 |
   | wrong-class real | 1.99 | 3.27 |
   | 4×4-block shuffled | 0.42 | 13.07 |
   | pixel shuffled | 0.47 | 15.82 |
   | CIFAR CPU model (EMA) | **0.18** | **18.22** |

   The ranking flips: in feature space the model is the worst set, and wrong-class real images beat every
   shuffle. This is direct support for next-step 2 below.
6. Pixel-space D̂1 is a good metric for coarse and class-level fit, not for sample quality. Compare it with the
   shuffle thresholds above, and report blockiness, spectra and FID next to it.

## Known issues
- **Patch mosaic** (above) is the main quality problem. Not fixed here.
- **Unbounded output.** 33% of generated Fashion-MNIST pixels fall outside [−1, 1], mostly around the exactly −1
  background. Grids clamp; D̂1 uses unclamped outputs. A tanh or clamp head is an untested option.
- **EMA 0.999 lags in short runs.** In the 2.8k-step overfit run the EMA is at 0.58 vs 0.079 raw. Fine at 16k steps.
- **Side jobs must also set `MKL_NUM_THREADS`.** `env/activate.sh` exports `MKL_NUM_THREADS=8`. A "1-thread" side
  job with only `OMP_NUM_THREADS=1` slowed the CIFAR run to about 1.8 it/s for about 4 min (19:07–19:11).
  `scripts/stage4_cpu_runs.sh` now sets both.
- **CPU thread oversubscription.** Any other process in the 8-CPU allocation next to an 8-thread run cut
  throughput from 4.9 to 0.75 it/s. The smoke run therefore restarted once at 17:37 with 7 threads (about 3 min
  lost; the run directory was cleared). Throughput was 4.3 it/s, against stage 2's 7.2 it/s with 8 threads on
  another node. The gap is not profiled: 7 vs 8 threads, node load average about 80, and EMA/log overhead.
- **The overfit check passes only partly.** The loss and D̂1 drop clearly and the samples match coarse shape, but
  12 min is not enough to reproduce individual images. A longer run, or patch size 2, would test whether exact
  overfitting is possible.
- **CPU FID** (CIFAR) uses 2000 samples, so it is biased upward and indicative only.
- **`scripts/train_gpu.slurm` is untested on a GPU.** Its signal → checkpoint → exit 3 → requeue logic was
  exercised on CPU (USR1 forwarded, checkpoint at step 771, exit 3), and resume is covered by pytest.
  - Assumption: QOS gpu48 (2-day max) covers l40s_public, so I used 24 h with requeue.
  - Fixed during writing: on GPU, `torch.load(map_location=cuda)` would put the CPU RNG state on the GPU. It is
    now `.cpu()`'d. This path has not been run on a GPU.
- The licence of `third_party/DiT` is CC-BY-NC 4.0, so this code is research-only.

## Suggested next experiment on GPU
1. **Short pilot first, not the full 200k steps.**
   - Submit `TRAIN_SET="train.max_steps=30000" RUN=runs/gpu_pilot_cifar10_dit-b2 sbatch scripts/train_gpu.slurm`
     (DiT-B/2, CIFAR-10, N = 512, bf16). `TRAIN_SET` is passed to `--set`; the run directory stays fixed across
     requeues.
   - Check the first-100-step timing in `metrics.jsonl` against stage 2's 0.5–0.7 s/step.
   - At 10k/20k/30k steps, look at the samples, FID-10k and blockiness first. Use D̂1 against the floor (0.011)
     and the shuffle thresholds (p = 16: 0.21) only as a necessary condition.
   - If the samples are still structureless colour mosaics, with FID in the hundreds, the pixel loss alone is not
     enough on CIFAR-10. Do not spend the 200k steps.
2. **The loss change most likely to matter**, given the measured blind spot: add a second strictly proper term,
   e.g. a per-class energy score on frozen-encoder features (Inception pool3 or a self-supervised encoder), as
   TBSM and Drifting use.
   - The sum of a strictly proper pixel term and a proper feature term is still strictly proper, so the minimiser
     is unchanged.
   - The shuffle check already shows that Inception pool3 ranks shuffled images and the CIFAR model far worse than
     wrong-class real ones (item 5 above). A self-supervised encoder should get the same check before use.
   - Caveat: training on Inception features and then reporting Inception FID is circular. Use a different encoder
     for the loss, or report FD-DINOv2 as well.
   - Also try a multi-scale pixel term (D_1 on 32/16/8-px average-pooled images).
3. **Cheap ablations on Fashion-MNIST** (CPU or a small GPU), at equal steps, measuring D̂1, blockiness, spectra
   and samples:
   - (a) DiT-T/2 vs T/4, the patch-size effect on the mosaic;
   - (b) z fed at a coarser resolution, or as a few noise tokens, to break patch-wise noise locality;
   - (c) β = 1.5;
   - (d) a tanh or clamp output head.

## Files written by stage 4
- `src/tmap/{__init__,energy_score,model,data,train,sample,evaluate}.py`, `pyproject.toml`.
- `tests/{conftest,test_energy_score,test_model,test_data,test_train,test_evaluate,test_toy2d}.py`.
- `configs/overfit_cpu.yaml` (new). Stage 2's configs are unchanged; overrides are passed by `--set`.
- `scripts/{plot_run.py,stage4_cpu_runs.sh,train_gpu.slurm,patch_shuffle_check.py,feature_space_check.py}`,
  `README.md`.
- `runs/2026-11-06_{smoke_fmnist_dit-t4,overfit_fmnist_sneaker256,smoke_cifar10_dit-t4,loss_blindspot}/`,
  `runs/pytest_2026-11-06.log`.
- Nothing was written outside the project except `/tmp` scratch, plus `/tmp/inception-2015-12-05.pt`, which
  clean-fid hard-codes.
