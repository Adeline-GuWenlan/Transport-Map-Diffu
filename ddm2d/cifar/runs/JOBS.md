# E3 GPU jobs (submitted by X2, 2026-11-26 18:37 UTC, code at 7d96e8f)

Each job: `RUN=<run> sbatch --export=ALL,RUN=<run> --job-name=ddmx_<run> --time=<t> cifar/scripts/train_gpu.slurm`
(`l40s_public`, 1 L40S, 8 CPUs, 32 GB, requeue-safe, USR1 at -300 s). Training writes `cifar/runs/<run>/` (`log.jsonl`, `train.log`,
`ckpt/` gitignored), then the evaluation protocol of `PLAN.md` writes `cifar/runs/<run>/eval/`. Slurm log:
`cifar/runs/slurm-ddmx_<run>-<jobid>.out`. Ids are also in `/scratch/wg2381/claude_jobs/ddm2dx/jobids/X2.txt`.

| run | job id | --time | projected GPU-h (PLAN.md) |
|---|---|---|---|
| cifar_B | 24752706 | 11:40:00 | 9.16 |
| cifar_C | 24752707 | 11:40:00 | 9.21 |
| cifar_A | 24752708 | 03:45:00 | 2.98 |
| cifar_B_b1 | 24752709 | 11:40:00 | 9.16 |
| cifar_B_pH | 24752710 | 11:40:00 | 9.29 |
| cifar_C_dino | 24752711 | 18:30:00 | 14.82 |

Earlier: bench 24750418 (COMPLETED, 00:20:16).

## Health checks
(X2 / X4 append here.)

### X2, 2026-11-26 14:28 EST (all six RUNNING; `health/health_2026-11-26T1430.jsonl`) [E]
All six started within 14 min of submission (13:37–13:51 EST). Losses finite at every logged step and falling.

| run | step | loss @100 | loss @1000 | loss now | grad norm (pre-clip) | frac clipped | s/step | peak GiB |
|---|---|---|---|---|---|---|---|---|
| cifar_B | 2900 | 67.20 | 28.67 | 26.15 | 170 | 1.00 | 0.988 | 30.6 |
| cifar_C | 2650 | 0.3316 | 0.3096 | 0.3021 | 0.32 | 0.00 | 1.003 | 30.6 |
| cifar_A | 10600 | 659.3 | 319.8 | 279.9 | 617 | 1.00 | 0.253 | 9.4 |
| cifar_B_b1 | 2300 | 5.532 | 3.3246 | 3.134 | 17.5 | 1.00 | 1.010 | 30.6 |
| cifar_B_pH | 2150 | 5.499 | 3.3242 | 3.148 | 21.0 | 1.00 | 1.003 | 30.6 |
| cifar_C_dino | 1250 | 0.875 | 0.630 | 0.614 | 0.87 | 0.22 | 1.653 | 13.5 |

Speeds match the bench (within 2 %). Projected training ends (EST, from these speeds; evaluation adds ≈ 0.8 h): A 15:50, B 21:50,
C 22:05, B_b1 / B_pH 22:15, C_dino 03:40 on Nov 27.

- cifar_A 10k EMA snapshot, CPU peek at NFE 10 (`health/cifar_A_snap_0010000_nfe10.png`, 100 images, seed 0; not an evaluation): every
  row shows its class (planes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks); images are blurry and low-contrast
  (pixel std 0.31), finite, 3e-6 of pixels outside [−1, 1].
- **Gradient clipping differs by arm** [E]: with clip 1 and squared or unit-power norms over 3072 pixels, A, B, B_b1, B_pH clip on
  every step (pre-clip norms 17–617); C never clips (0.3); C_dino clips on 22 %. Adam divides out a constant gradient scale, so this
  mostly acts as per-step normalisation, but it is a real difference between arms under the shared clip = 1 setting [G].
- **B_pH ≈ B_b1** [E for the losses, G for the reason]: their losses agree to 3 decimals so far. With c = 0.0299 and ‖x̂ − x_0‖ ≫ c over
  3072 pixels (even at t = 0.01 the noise alone gives ‖d‖ ≈ 0.55), √(‖d‖² + c²) − c ≈ ‖d‖ − c, i.e. the β = 1 loss. iCT's c was set
  for the gap between two consecutive consistency-model outputs, which is much smaller. Expect B_pH and B_b1 to end up close; that is
  a finding about this c, not a bug.
