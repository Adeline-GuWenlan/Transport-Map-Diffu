# 2D training jobs (ddm2dx X1, launched 2026-11-26)

Job script `scripts/train_job.sbatch` (unchanged): `cpu_short`, account `torch_pr_527_general`, 6 CPUs, 12 GB, 5 h, `--requeue`.
6 runs per job, 1 thread each, `--deadline` = job end − 12 min; unfinished runs checkpoint and exit 3, and the job submits its own
continuation (`--dependency=afterany`, up to 5 segments) and appends its id to `/scratch/wg2381/claude_jobs/ddm2dx/jobids/X1.txt`.
Per-job events: `runs/ext/joblog.tsv`. Per-run stdout: `runs/ext/<run>.out`. Slurm stdout:
`/scratch/wg2381/claude_jobs/logs/ddm2d_train_ddm2dx_tr<NN>-<jobid>.out`.

Command (from `ddm2d/`, code at commit `6112f21`):
`sbatch --job-name=ddm2dx_tr<NN> --export=ALL,RUNLIST=runs/ext/runlists/list<NN>.txt,JOBIDS=/scratch/wg2381/claude_jobs/ddm2dx/jobids/X1.txt,JOBLOG=runs/ext/joblog.tsv scripts/train_job.sbatch`

| job (segment 1) | name | run list | runs |
|---|---|---|---|
| 24751248 | ddm2dx_tr01 | `runlists/list01.txt` | `gmm_B_b0.1_s0` `gmm_B_b0.1_s1` `gmm_B_b0.5_s0` `gmm_B_b0.5_s1` `gmm_B_b1_s0` `gmm_B_b1_s1` |
| 24751250 | ddm2dx_tr02 | `runlists/list02.txt` | `gmm_B_b1.5_s0` `gmm_B_b1.5_s1` `gmm_B_b1.99_s0` `gmm_B_b1.99_s1` `gmm_B_b3_s0` `gmm_B_b3_s1` |
| 24751251 | ddm2dx_tr03 | `runlists/list03.txt` | `gmm_B_b3.5_s0` `gmm_B_b3.5_s1` `gmm_C_b0.5_s0` `gmm_C_b0.5_s1` `gmm_C_b1_s0` `gmm_C_b1_s1` |
| 24751252 | ddm2dx_tr04 | `runlists/list04.txt` | `gmm_C_b1.5_s0` `gmm_C_b1.5_s1` `gmm_C_b1.99_s0` `gmm_C_b1.99_s1` `gmm_C_b2_s0` `gmm_C_b2_s1` |
| 24751254 | ddm2dx_tr05 | `runlists/list05.txt` | `gmm2w_A_s0` `gmm2w_A_s1` `gmm2w_B_s0` `gmm2w_B_s1` `gmm2w_C_s0` `gmm2w_C_s1` |
| 24751255 | ddm2dx_tr06 | `runlists/list06.txt` | `gmm9w_A_s0` `gmm9w_A_s1` `gmm9w_B_s0` `gmm9w_B_s1` `gmm9w_C_s0` `gmm9w_C_s1` |
| 24751256 | ddm2dx_tr07 | `runlists/list07.txt` | `gmm_C_L1b1_s0` `gmm_C_L1b1_s1` `gmm_C_L1b0.5_s0` `gmm_C_L1b0.5_s1` `gmm_B_L1b1_s0` `gmm_B_L1b1_s1` |
| 24751258 | ddm2dx_tr08 | `runlists/list08.txt` | `gmm_C_pH0.1_s0` `gmm_C_pH0.1_s1` `gmm_C_pH1_s0` `gmm_C_pH1_s1` `gmm_B_pH0.1_s0` `gmm_B_pH0.1_s1` |
| 24751259 | ddm2dx_tr09 | `runlists/list09.txt` | `gmm_C_imq_s0` `gmm_C_imq_s1` `gmm_C_rbf_s0` `gmm_C_rbf_s1` `gmm_C_lap_s0` `gmm_C_lap_s1` |
| 24751260 | ddm2dx_tr10 | `runlists/list10.txt` | `gmm9w_C_L1b1_s0` `gmm9w_C_L1b0.5_s0` `gmm9w_B_L1b1_s0` `gmm9w_C_pH0.1_s0` `gmm9w_C_pH1_s0` `gmm9w_B_pH0.1_s0` |
| 24751261 | ddm2dx_tr11 | `runlists/list11.txt` | `gmm9w_C_imq_s0` `gmm9w_C_rbf_s0` `gmm9w_C_lap_s0` `gmm9w_B_b1_s0` |

Submitted 2026-11-26 18:21 UTC. At submission the per-user `cpu_short` cap (32 CPUs; the X1 and X2 agents hold 16) let 2 jobs start
at once; the rest wait with reason `QOSMaxCpuPerUserLimit` and start as CPUs free up.

## Budget (≤ 400 CPU-h of 2D training for the chain)
Estimate (`PLAN.md`): 11 jobs × 6 CPUs × 4.5–5.8 h ≈ 297–383 CPU-h, charged by allocation (a job holds 6 CPUs until its slowest run
ends). The smoke speed (0.11–0.13 s/step, i.e. 3.1–3.6 h per run) suggests the low end. Actual use: `sacct` over the ids in
`jobids/X1.txt` (X3 records the final figure).
