# runs/ext PLAN (ddm2dx X1)

Spec: `/scratch/wg2381/claude_jobs/ddm2dx/prompts/_task.md` (E1, E2, E4). Every config is `configs/paper/gmm_{A,B,C}.json` with only
dataset / β / λ / ρ / ρ-parameter changed (`tests/test_ext_configs.py`). 100k steps, batch 128, m = 32 (A: m = 1). Run seed = training
seed (init, data order, t, noise, ξ); data seed 0 for every run. Evaluation (X3) uses the ddm2d protocol for every run.

ρ parameters: pseudo-Huber `rho_c` = c; IMQ `rho_c` = c (k = (‖d‖² + c)^(-1/2), main.tex Eq. imq_k); RBF `rho_c` = σ² = M² (γ = 1);
Laplace `rho_c` = σ = M (γ = 1). M², M: median of ‖x − x'‖², ‖x − x'‖ over 100 batches of 512 training points, seed 0
(`losses.median_sq_dist`, main.tex line 1747); values in `median_heuristic.json`. **Caveat [E]:** on the paper GMM the median falls
in the gap between within-mode and cross-mode distances, so it moves with the batch seed (M² = 12.5 for seed 0; 9.8 / 13.6 / 8.1 for
seeds 1–3). On `gmm9w` it is stable (34.6 ± 0.05).

s/step: smoke measurement (`smoke/summary.json`, 1 thread, 7 processes side by side on one node; steps 100→400). Smoke is not a
result. CPU-h estimate per run = 100k × s/step / 3600 × 1.3 (the ddm2d paper runs were 1.0–1.6× slower in packed jobs than alone).

| # | exp | run dir | dataset | arm | β | λ | ρ | ρ param | seed | list | smoke s/step | est. CPU-h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | E1a | `runs/ext/gmm_B_b0.1_s0` | gmm | B | 0.1 | 0.0 | l2 | – | 0 | 01 | 0.112 | 4.1 |
| 2 | E1a | `runs/ext/gmm_B_b0.1_s1` | gmm | B | 0.1 | 0.0 | l2 | – | 1 | 01 | 0.112 | 4.1 |
| 3 | E1a | `runs/ext/gmm_B_b0.5_s0` | gmm | B | 0.5 | 0.0 | l2 | – | 0 | 01 | 0.121 | 4.4 |
| 4 | E1a | `runs/ext/gmm_B_b0.5_s1` | gmm | B | 0.5 | 0.0 | l2 | – | 1 | 01 | 0.121 | 4.4 |
| 5 | E1a | `runs/ext/gmm_B_b1_s0` | gmm | B | 1.0 | 0.0 | l2 | – | 0 | 01 | 0.121 | 4.4 |
| 6 | E1a | `runs/ext/gmm_B_b1_s1` | gmm | B | 1.0 | 0.0 | l2 | – | 1 | 01 | 0.121 | 4.4 |
| 7 | E1a | `runs/ext/gmm_B_b1.5_s0` | gmm | B | 1.5 | 0.0 | l2 | – | 0 | 02 | 0.124 | 4.5 |
| 8 | E1a | `runs/ext/gmm_B_b1.5_s1` | gmm | B | 1.5 | 0.0 | l2 | – | 1 | 02 | 0.124 | 4.5 |
| 9 | E1a | `runs/ext/gmm_B_b1.99_s0` | gmm | B | 1.99 | 0.0 | l2 | – | 0 | 02 | 0.121 | 4.4 |
| 10 | E1a | `runs/ext/gmm_B_b1.99_s1` | gmm | B | 1.99 | 0.0 | l2 | – | 1 | 02 | 0.121 | 4.4 |
| 11 | E1a | `runs/ext/gmm_B_b3_s0` | gmm | B | 3.0 | 0.0 | l2 | – | 0 | 02 | 0.123 | 4.4 |
| 12 | E1a | `runs/ext/gmm_B_b3_s1` | gmm | B | 3.0 | 0.0 | l2 | – | 1 | 02 | 0.123 | 4.4 |
| 13 | E1a | `runs/ext/gmm_B_b3.5_s0` | gmm | B | 3.5 | 0.0 | l2 | – | 0 | 03 | 0.116 | 4.2 |
| 14 | E1a | `runs/ext/gmm_B_b3.5_s1` | gmm | B | 3.5 | 0.0 | l2 | – | 1 | 03 | 0.116 | 4.2 |
| 15 | E1b | `runs/ext/gmm_C_b0.5_s0` | gmm | C | 0.5 | 1.0 | l2 | – | 0 | 03 | 0.123 | 4.5 |
| 16 | E1b | `runs/ext/gmm_C_b0.5_s1` | gmm | C | 0.5 | 1.0 | l2 | – | 1 | 03 | 0.123 | 4.5 |
| 17 | E1b | `runs/ext/gmm_C_b1_s0` | gmm | C | 1.0 | 1.0 | l2 | – | 0 | 03 | 0.114 | 4.1 |
| 18 | E1b | `runs/ext/gmm_C_b1_s1` | gmm | C | 1.0 | 1.0 | l2 | – | 1 | 03 | 0.114 | 4.1 |
| 19 | E1b | `runs/ext/gmm_C_b1.5_s0` | gmm | C | 1.5 | 1.0 | l2 | – | 0 | 04 | 0.119 | 4.3 |
| 20 | E1b | `runs/ext/gmm_C_b1.5_s1` | gmm | C | 1.5 | 1.0 | l2 | – | 1 | 04 | 0.119 | 4.3 |
| 21 | E1b | `runs/ext/gmm_C_b1.99_s0` | gmm | C | 1.99 | 1.0 | l2 | – | 0 | 04 | 0.114 | 4.1 |
| 22 | E1b | `runs/ext/gmm_C_b1.99_s1` | gmm | C | 1.99 | 1.0 | l2 | – | 1 | 04 | 0.114 | 4.1 |
| 23 | E1b | `runs/ext/gmm_C_b2_s0` | gmm | C | 2.0 | 1.0 | l2 | – | 0 | 04 | 0.118 | 4.3 |
| 24 | E1b | `runs/ext/gmm_C_b2_s1` | gmm | C | 2.0 | 1.0 | l2 | – | 1 | 04 | 0.118 | 4.3 |
| 25 | E2 | `runs/ext/gmm2w_A_s0` | gmm2w | A | – | – | – (Eq. 6) | – | 0 | 05 | 0.115 | 4.1 |
| 26 | E2 | `runs/ext/gmm2w_A_s1` | gmm2w | A | – | – | – (Eq. 6) | – | 1 | 05 | 0.115 | 4.1 |
| 27 | E2 | `runs/ext/gmm2w_B_s0` | gmm2w | B | 2.0 | 0.0 | l2 | – | 0 | 05 | 0.113 | 4.1 |
| 28 | E2 | `runs/ext/gmm2w_B_s1` | gmm2w | B | 2.0 | 0.0 | l2 | – | 1 | 05 | 0.113 | 4.1 |
| 29 | E2 | `runs/ext/gmm2w_C_s0` | gmm2w | C | 0.1 | 1.0 | l2 | – | 0 | 05 | 0.113 | 4.1 |
| 30 | E2 | `runs/ext/gmm2w_C_s1` | gmm2w | C | 0.1 | 1.0 | l2 | – | 1 | 05 | 0.113 | 4.1 |
| 31 | E2 | `runs/ext/gmm9w_A_s0` | gmm9w | A | – | – | – (Eq. 6) | – | 0 | 06 | 0.111 | 4.0 |
| 32 | E2 | `runs/ext/gmm9w_A_s1` | gmm9w | A | – | – | – (Eq. 6) | – | 1 | 06 | 0.111 | 4.0 |
| 33 | E2 | `runs/ext/gmm9w_B_s0` | gmm9w | B | 2.0 | 0.0 | l2 | – | 0 | 06 | 0.113 | 4.1 |
| 34 | E2 | `runs/ext/gmm9w_B_s1` | gmm9w | B | 2.0 | 0.0 | l2 | – | 1 | 06 | 0.113 | 4.1 |
| 35 | E2 | `runs/ext/gmm9w_C_s0` | gmm9w | C | 0.1 | 1.0 | l2 | – | 0 | 06 | 0.118 | 4.2 |
| 36 | E2 | `runs/ext/gmm9w_C_s1` | gmm9w | C | 0.1 | 1.0 | l2 | – | 1 | 06 | 0.118 | 4.2 |
| 37 | E4 | `runs/ext/gmm_C_L1b1_s0` | gmm | C | 1.0 | 1.0 | l1 | – | 0 | 07 | 0.123 | 4.4 |
| 38 | E4 | `runs/ext/gmm_C_L1b1_s1` | gmm | C | 1.0 | 1.0 | l1 | – | 1 | 07 | 0.123 | 4.4 |
| 39 | E4 | `runs/ext/gmm_C_L1b0.5_s0` | gmm | C | 0.5 | 1.0 | l1 | – | 0 | 07 | 0.114 | 4.1 |
| 40 | E4 | `runs/ext/gmm_C_L1b0.5_s1` | gmm | C | 0.5 | 1.0 | l1 | – | 1 | 07 | 0.114 | 4.1 |
| 41 | E4 | `runs/ext/gmm_B_L1b1_s0` | gmm | B | 1.0 | 0.0 | l1 | – | 0 | 07 | 0.112 | 4.0 |
| 42 | E4 | `runs/ext/gmm_B_L1b1_s1` | gmm | B | 1.0 | 0.0 | l1 | – | 1 | 07 | 0.112 | 4.0 |
| 43 | E4 | `runs/ext/gmm_C_pH0.1_s0` | gmm | C | – | 1.0 | pseudo_huber | 0.1 | 0 | 08 | 0.116 | 4.2 |
| 44 | E4 | `runs/ext/gmm_C_pH0.1_s1` | gmm | C | – | 1.0 | pseudo_huber | 0.1 | 1 | 08 | 0.116 | 4.2 |
| 45 | E4 | `runs/ext/gmm_C_pH1_s0` | gmm | C | – | 1.0 | pseudo_huber | 1 | 0 | 08 | 0.118 | 4.2 |
| 46 | E4 | `runs/ext/gmm_C_pH1_s1` | gmm | C | – | 1.0 | pseudo_huber | 1 | 1 | 08 | 0.118 | 4.2 |
| 47 | E4 | `runs/ext/gmm_B_pH0.1_s0` | gmm | B | – | 0.0 | pseudo_huber | 0.1 | 0 | 08 | 0.122 | 4.4 |
| 48 | E4 | `runs/ext/gmm_B_pH0.1_s1` | gmm | B | – | 0.0 | pseudo_huber | 0.1 | 1 | 08 | 0.122 | 4.4 |
| 49 | E4 | `runs/ext/gmm_C_imq_s0` | gmm | C | – | 1.0 | imq | 1 | 0 | 09 | 0.114 | 4.1 |
| 50 | E4 | `runs/ext/gmm_C_imq_s1` | gmm | C | – | 1.0 | imq | 1 | 1 | 09 | 0.114 | 4.1 |
| 51 | E4 | `runs/ext/gmm_C_rbf_s0` | gmm | C | – | 1.0 | rbf | 12.51 | 0 | 09 | 0.111 | 4.0 |
| 52 | E4 | `runs/ext/gmm_C_rbf_s1` | gmm | C | – | 1.0 | rbf | 12.51 | 1 | 09 | 0.111 | 4.0 |
| 53 | E4 | `runs/ext/gmm_C_lap_s0` | gmm | C | – | 1.0 | laplace | 3.537 | 0 | 09 | 0.117 | 4.2 |
| 54 | E4 | `runs/ext/gmm_C_lap_s1` | gmm | C | – | 1.0 | laplace | 3.537 | 1 | 09 | 0.117 | 4.2 |
| 55 | E4 | `runs/ext/gmm9w_C_L1b1_s0` | gmm9w | C | 1.0 | 1.0 | l1 | – | 0 | 10 | 0.123 | 4.4 |
| 56 | E4 | `runs/ext/gmm9w_C_L1b0.5_s0` | gmm9w | C | 0.5 | 1.0 | l1 | – | 0 | 10 | 0.124 | 4.5 |
| 57 | E4 | `runs/ext/gmm9w_B_L1b1_s0` | gmm9w | B | 1.0 | 0.0 | l1 | – | 0 | 10 | 0.120 | 4.3 |
| 58 | E4 | `runs/ext/gmm9w_C_pH0.1_s0` | gmm9w | C | – | 1.0 | pseudo_huber | 0.1 | 0 | 10 | 0.125 | 4.5 |
| 59 | E4 | `runs/ext/gmm9w_C_pH1_s0` | gmm9w | C | – | 1.0 | pseudo_huber | 1 | 0 | 10 | 0.131 | 4.7 |
| 60 | E4 | `runs/ext/gmm9w_B_pH0.1_s0` | gmm9w | B | – | 0.0 | pseudo_huber | 0.1 | 0 | 10 | 0.117 | 4.2 |
| 61 | E4 | `runs/ext/gmm9w_C_imq_s0` | gmm9w | C | – | 1.0 | imq | 1 | 0 | 11 | 0.116 | 4.2 |
| 62 | E4 | `runs/ext/gmm9w_C_rbf_s0` | gmm9w | C | – | 1.0 | rbf | 34.59 | 0 | 11 | 0.117 | 4.2 |
| 63 | E4 | `runs/ext/gmm9w_C_lap_s0` | gmm9w | C | – | 1.0 | laplace | 5.882 | 0 | 11 | 0.126 | 4.6 |
| 64 | E2x | `runs/ext/gmm9w_B_b1_s0` | gmm9w | B | 1.0 | 0.0 | l2 | – | 0 | 11 | 0.122 | 4.4 |

Runs: 64; run lists: 11 (`runlists/list01..11.txt`, 6 runs each except the last).
Sum of per-run estimates: **272 CPU-h**. But a `train_job.sbatch` job holds 6 CPUs until its slowest run ends, so the
charge is ≈ 6 × (slowest run) per job: 11 jobs × 6 CPUs × 4.5–5.8 h ≈ 297–383 CPU-h
(budget ≤ 400 CPU-h of 2D training for the chain).

Not launched: the second E2 extra (C on `gmm9w` with the best E1b β). It needs the E1b evaluation (X3). Budget left for it is in
`JOBS.md`.
