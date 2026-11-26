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
