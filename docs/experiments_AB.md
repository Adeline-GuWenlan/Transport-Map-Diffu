# Experiments A / B / C: 30k-step GPU runs (submitted 2026-11-10)

## A: 30k GPU baseline
DiT-B/2 (129.7 M params, patch 2), CIFAR-10, the original pixel energy score (stage 1 loss, β = 1, class-grouped
U-statistic). Everything else is unchanged from `configs/base_gpu.yaml` (stage 2):
- batch N = 512 (8 classes × 64);
- AdamW 1e-4, betas (0.9, 0.95), 5k warmup, clip 2.0;
- EMA 0.9999, bf16 autocast.

Only `max_steps` = 30k and the evaluation schedule differ. Config: `configs/expA_cifar10_dit-b2_pixel.yaml`.

## B: feature-loss ablation
Same model, data, budget and seed as A, plus a frozen-encoder energy-score term. Config:
`configs/expB_cifar10_dit-b2_pixel-dinov2.yaml`.

L(θ) = L_pixel(θ) + λ · L_feat(θ),

where L_feat is the same class-grouped unbiased energy-score estimator applied to φ(F(z, c)) against φ(y).
- **Encoder φ:** DINOv2 ViT-S/14, the 384-d CLS token. Images are bicubic-upsampled 32 → 224 px; the parameters
  are frozen (`src/tmap/features.py`). It is self-supervised and is not the FID network, so FID and Inception D1
  stay non-circular for B. DINOv2 D1 *is* B's training space, so read it as circular for B.
  - Weights: timm `vit_small_patch14_dinov2.lvd142m`, Apache-2.0.
- **Why the minimiser is unchanged:** the feature term is a proper scoring rule of φ(X). Added to the strictly
  proper pixel term, the sum is still strictly proper, so the minimiser is still P_θ = Q.
- **λ:** λ = weight × s, with weight = 1.0 (**assumption**: equal footing). s = (mean same-class pixel distance) /
  (mean same-class feature distance), measured on the first 200 train images per class. It is computed once and
  stored in the checkpoint. A CPU dry run with 5 images per class gave s ≈ 0.68; the GPU runs log their own value
  in `feature_scale.jsonl`.

## Seeds
Seeds 0 and 1 for both A and B, giving 4 runs. A seed sets the init, the data sampler and the training noise.
Evaluation uses fixed noise in every run. A third seed can be added the same way.

## Memory: why the runs use an exact two-pass gradient
On CPU, under bf16 autocast, I summed the tensors saved for backward. DiT-B/2 saves about **111 MiB per
sample**, which is about **55 GiB at N = 512**. DINOv2-S/14 at 224 px adds about 26 MiB per sample, about 13 GiB.
One pass does not fit a 48 GB L40S. The A100/H100 partitions were draining on 2026-11-10; only the contested
H200 nodes would fit one pass.

`train.grad_chunk: 128` (`tmap.train.forward_backward`) solves this:
- **Pass 1:** compute F(z, c) and φ for the whole batch without a graph, then the loss and dL/dx, dL/dφ on
  detached copies.
- **Pass 2:** recompute each 128-sample chunk with a graph and backpropagate ⟨x, dL/dx⟩ + ⟨φ(x), dL/dφ⟩.

The parameter gradient is exactly the one-pass gradient: `tests/test_train.py` checks that, with and without the
feature term. The cost is one extra forward pass, about +1/3 compute. Activation memory drops to about 14 GiB
(A) or 17 GiB (B), plus about 3 GB of weights and optimizer state.

## Resources (per run)
1 × L40S (48 GB), 8 CPUs, 32 GB RAM, partition `l40s_public`. Time limits: **A 10 h, B 12 h**.

The limits come from an estimate, not a measurement. Per step, A costs 4 × 46 GFLOP × 512 ≈ 94 TFLOP, and B
costs about 117 TFLOP (the DINOv2 passes add about 22 TFLOP). At an assumed 100–180 TFLOP/s sustained bf16 on an
L40S, that is 0.5–0.9 s/step for A, i.e. 4.5–8 h for 30k steps, and about 25% more for B. Snapshot evaluation adds
about 20–30 min.

If a run needs longer, it checkpoints 5 min before the limit, requeues, and resumes, losing at most 1000 steps.
`scripts/bench_gpu.slurm` (job 22605880, 50 min) measures s/step and peak memory (chunk 128 and 256 for A, 128 for
B), runs the evaluation path once, and runs the DINOv2 block-shuffle check. If it shows the runs are faster, lower
the pending jobs' limits with `scontrol update JobId=<id> TimeLimit=<h:mm:ss>`.

| run | job | config | seed | time |
|---|---|---|---|---|
| `runs/expA_s0` | 22606246 | expA | 0 | 10 h |
| `runs/expB_s0` | 22606247 | expB | 0 | 12 h |
| `runs/expA_s1` | 22606248 | expA | 1 | 10 h |
| `runs/expB_s1` | 22606249 | expB | 1 | 12 h |

```bash
CONFIG=configs/expA_cifar10_dit-b2_pixel.yaml RUN=runs/expA_s0 TRAIN_SET="seed=0" \
  sbatch --job-name=tmap_A_s0 --time=10:00:00 scripts/exp_gpu.slurm
```

## What each run writes
- **During training:**
  - `metrics.jsonl` every 100 steps: loss and its pixel/feature parts, grad norm, lr, it/s, `max_mem_gib`.
  - `eval.jsonl`: held-out pixel D1 (raw and EMA, 1000/class) every 5k steps.
  - `samples/step*_{raw,ema}.png` every 5k steps.
  - `ckpt/latest.pt` every 1k steps.
  - `ckpt/step{5k,10k,20k,30k}.pt`: weights-only snapshots.
- **After training** (`eval/step*.json`, `.log`, `_samples_{ema,raw}.png`, `_nn.png`), for every snapshot and
  both weights:
  - **FID-10k**: clean-fid legacy_tensorflow vs CIFAR-10 train stats. It sits above the paper-standard FID-50k;
    use it only to compare runs and steps.
  - **pixel D1**, with baselines.
  - **Inception-pool3 D1** and **DINOv2 D1**, each with the right-class real (floor) and wrong-class real
    references in that space.
  - class checks (1-NN agreement, D1-confusion argmin), blockiness, radial power spectrum, diversity ratio and
    nearest-train neighbours.
- **Across runs:** `python scripts/summarize_ab.py` writes `runs/expAB_summary_ema.md` (table and the C rule),
  `runs/expAB_metrics_ema.png` (FID, pixel, Inception and DINOv2 D1 vs step; A blue, B orange, one line per seed)
  and `runs/expAB_samples_ema.png` (sample grids by step × run). Use `--weights raw` for the raw weights.
- **EMA lag:** EMA 0.9999 has a half-life of about 6.9k steps, so at 5k steps the EMA is mostly the
  early-training weights. Compare raw weights at 5k.

## C: decide whether to go to 200k
`scripts/summarize_ab.py` applies this rule to A's numbers (20k → 30k). The thresholds are **assumptions**, and the
sample grids must be looked at too.
- **Go to the 200k run (original goal):** all three must hold:
  - FID falls monotonically from 10k to 20k to 30k;
  - FID and Inception D1 each improve by ≥ 10% from 20k to 30k;
  - the grids show recognisable structure (cats, cars, birds), not 2×2 colour mosaics.

  Resubmit `configs/base_gpu.yaml` with `train.grad_chunk=128` (it also needs the chunked gradient on L40S), or
  continue A from its checkpoint with a larger `train.max_steps`.
- **Test new losses first:** pixel D1 still falls by ≥ 5% while FID and Inception D1 change by < 5%, and the grids
  show no structure. Then B's result decides the next loss: if B beats A on FID and Inception D1 in both seeds,
  develop the feature term (encoder, λ, multi-scale) before any long run.
- **Anything in between:** decide from the grids and from B.
