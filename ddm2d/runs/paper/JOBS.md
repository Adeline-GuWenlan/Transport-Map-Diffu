# Paper-scale training jobs (launched by S2, 2026-11-19)

Job script `scripts/train_job.sbatch` (commit `78a6e03`): `cpu_short`, account `torch_pr_527_general`, 6 CPUs, 12 GB, 5 h, `--requeue`.
Each job runs the 6 runs of one seed as 6 processes × 1 thread (`--threads 1`, `OMP_NUM_THREADS=1`), with `--deadline` = job end − 12 min.
A run that is not finished at the deadline checkpoints and exits 3; the job then submits a continuation of itself
(`--dependency=afterany:<job>`, up to 5 segments) and appends its id to `/scratch/wg2381/claude_jobs/ddm2d/jobids/S2.txt`.
Per-job events (start, host, exit code per run) are appended to `runs/paper/joblog.tsv`; per-run stdout is `runs/paper/<run>.out`;
Slurm stdout is `/scratch/wg2381/claude_jobs/logs/ddm2d_train_<jobname>-<jobid>.out`.

Command: `sbatch --job-name=ddm2d_train_s<seed> --export=ALL,RUNLIST=runs/paper/runlists/seed<seed>.txt scripts/train_job.sbatch` (from `ddm2d/`).

| run dir | config | seed | job (segment 1) |
|---|---|---|---|
| `runs/paper/gmm_A_s0` | `configs/paper/gmm_A.json` | 0 | 23769609 |
| `runs/paper/gmm_B_s0` | `configs/paper/gmm_B.json` | 0 | 23769609 |
| `runs/paper/gmm_C_s0` | `configs/paper/gmm_C.json` | 0 | 23769609 |
| `runs/paper/cb_A_s0` | `configs/paper/cb_A.json` | 0 | 23769609 |
| `runs/paper/cb_B_s0` | `configs/paper/cb_B.json` | 0 | 23769609 |
| `runs/paper/cb_C_s0` | `configs/paper/cb_C.json` | 0 | 23769609 |
| `runs/paper/gmm_A_s1` | `configs/paper/gmm_A.json` | 1 | 23769610 |
| `runs/paper/gmm_B_s1` | `configs/paper/gmm_B.json` | 1 | 23769610 |
| `runs/paper/gmm_C_s1` | `configs/paper/gmm_C.json` | 1 | 23769610 |
| `runs/paper/cb_A_s1` | `configs/paper/cb_A.json` | 1 | 23769610 |
| `runs/paper/cb_B_s1` | `configs/paper/cb_B.json` | 1 | 23769610 |
| `runs/paper/cb_C_s1` | `configs/paper/cb_C.json` | 1 | 23769610 |
| `runs/paper/gmm_A_s2` | `configs/paper/gmm_A.json` | 2 | 23769636 |
| `runs/paper/gmm_B_s2` | `configs/paper/gmm_B.json` | 2 | 23769636 |
| `runs/paper/gmm_C_s2` | `configs/paper/gmm_C.json` | 2 | 23769636 |
| `runs/paper/cb_A_s2` | `configs/paper/cb_A.json` | 2 | 23769636 |
| `runs/paper/cb_B_s2` | `configs/paper/cb_B.json` | 2 | 23769636 |
| `runs/paper/cb_C_s2` | `configs/paper/cb_C.json` | 2 | 23769636 |

Data seed is 0 for every run (the same 102,400-point training set per dataset, `configs/paper/*.json` `data_seed`); the run seed controls
model init, data order, t, forward noise and ξ. Checkerboard B is the optional extra (`_task.md` §2 lists A and C for the checkerboard).
Continuation job ids, if any, are in `jobids/S2.txt` and `runs/paper/joblog.tsv`.
