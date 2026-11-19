# Job-script test (S2, 2026-11-19, smoke only)
`scripts/train_job.sbatch` run with `bash` inside the S2 allocation (no Slurm submission) on two 1500-step copies of the smoke configs
(gmm A, gmm C): segment 1 with `DEADLINE_OVERRIDE` = start + 25 s → both runs exit 3 after checkpointing (steps 91, 82) and the script calls
the (fake) `sbatch` for a continuation with `--dependency=afterany:<job>` and appends its id to `jobids.txt`; segment 2 resumes and finishes
(DONE.json, `n_segments` 2, 150 log lines); segment 3 skips both runs. Logs: `joblog.tsv`, `fake_sbatch_calls.txt`, `run_*.out`, `run_*/`.
Checkpoints deleted after the test.
