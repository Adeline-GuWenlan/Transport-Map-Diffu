# `_task.md` §5 checklist (smoke config only)

Done by **S1** on 2026-11-19 (cs607, Slurm job 23760197), at commit `e211297` plus the smoke files committed with this doc. These are smoke outputs:
none of them is a paper-scale result. S2 owns this checklist and may redo or extend any item (checkerboard arm B was run here too, since it is cheap).

Smoke runs: `configs/smoke/{gmm,cb}_{A,B,C}.json` = the paper model, optimizer and loss with 300 steps (ckpt every 100, log every 10).
Command, for each config: `PYTHONPATH=src python -m ddm2d.train --config configs/smoke/<c>.json --out runs/smoke/<c>_s0 --seed 0 --threads 1`
(6 runs in parallel, 1 thread each, 34–38 s each).

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Environment, GPU | done. Python 3 env `tmap`, torch 2.14.1+cu126, CPU only; **no GPU in this allocation**, and GPU is not used by design (user's `gpu48` QOS = 2 GPUs, both held by the user's other chains, `_rules.md`; CPU is fast enough, `docs/timing_s1.json`) | `runs/smoke/checklist/env.json` |
| 2 | Generate both datasets, plot ground truth | done. 102,400 points each, data seed 0; checkerboard: 100 % of points in the support | `runs/smoke/checklist/datasets.{png,pdf,json}` (sha256 of each training set and of eval set 0) |
| 3 | Instantiate all three variants | done. A: 8,569,858 params, x-branch in 2, out 2; B, C: 8,570,116 params, x-branch in 4, out 4 → prediction [n, m, 2] | `runs/smoke/checklist/models.json`; `tests/test_model.py` |
| 4 | Forward diffusion and bridge equations | done (unit tests): marginal moments, bridge marginal preservation (closed form + Monte Carlo) for churn 0, 0.5, 1, DDIM/DDPM limits, s = 0 | `tests/test_schedule.py`; `runs/smoke/checklist/pytest.txt` (71 passed) |
| 5 | Generalized energy loss vs explicit reference | done: Eq. (14) vs a double loop for β ∈ {0.1, 2}, λ ∈ {0, 1}, δ ∈ {0, 1e-12}, rel. tol 1e-12 | `tests/test_losses.py`; `pytest.txt` |
| 6 | Several forward/backward steps, finite losses and grads | done: all 6 smoke runs, every logged window finite (`"finite": true`); loss first → last window: gmm A 9.24 → 0.318, B 9.03 → 0.305, C 0.487 → 0.241; cb A 5.11 → 0.420, B 5.13 → 0.410, C 0.465 → 0.271. Pre-clip grad norm max 130–234 for A/B in the first steps (clipped to 1), ≤ 2.1 for C | `runs/smoke/<c>_s0/log.jsonl` |
| 7 | Smoke model per dataset and arm | done: 6 runs, each with `DONE.json`, `ckpt/final.pt` (gitignored), `ckpt_manifest.txt` | `runs/smoke/{gmm,cb}_{A,B,C}_s0/` |
| 8 | Save / reload incl. optimizer, EMA, RNG; continue = uninterrupted | done: `cb_C` smoke run stopped at step 150 (process exit), resumed to 300; model and EMA **bitwise equal** to the uninterrupted run (max abs diff 0.0), generator state equal, all 30 log lines equal. Unit test also covers a deadline stop and an epoch reshuffle for all arms | `runs/smoke/checklist/resume.json`; `runs/smoke/cb_C_s0_resumed/`; `tests/test_train.py` |
| 9 | Samples with the actual samplers | done: Algorithm 2 (B, C) and the ε = 1 bridge with deterministic x̂ (A), NFE 5, 20, 100 (GMM) and 5, 10, 100 (checkerboard), 4096 samples, EMA weights; NFE counted by the sampler | `runs/smoke/checklist/eval_{gmm,cb}/samples.{png,pdf}` |
| 10 | MMD and one end-to-end evaluation figure | done: unbiased MMD²_u, σ = 1, vs eval set 0, plus biased MMD² and D_ρ in the JSON; GMM also posterior std at 5 times | `runs/smoke/checklist/eval_{gmm,cb}/eval.json`, `samples.png` |

The 300-step smoke models are far from converged (MMD²_u 0.2–1.3), as expected with EMA 0.99 and a 100-step warmup. A longer sanity check
(paper config stopped at 3000 steps, `runs/smoke/sanity3k_gmm_*`) already gives MMD²_u at NFE 5 of 1.48e-2 (A), 1.41e-2 (B), 9.2e-4 (C)
(`runs/smoke/sanity3k_gmm_eval/eval.json`) — smoke numbers, not results.

Not done here (S2): the packed-process benchmark, the job script, and the paper-scale launch.

**S2 review (2026-11-19):** S2 accepted these items as they stand (S1 did them with the code S2 launched, except for the EMA-snapshot option
added in `78a6e03`, which is covered by `tests/test_train.py::test_ema_snapshots_and_resume_unchanged`; 74 tests pass). Item 1, GPU: the
S2 allocation (cs601, job 23766928) has no GPU either; GPU is not used because the user's `gpu48` QOS allows 2 GPUs and the user's other
chains hold them (`_rules.md`), and CPU suffices (`docs/benchmark.md`). Item 8 was re-checked at the job-script level (deadline exit 3 →
continuation → resume → DONE; `runs/smoke/jobtest/README.md`).
