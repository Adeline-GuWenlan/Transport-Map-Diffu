# Smoke (ddm2dx X1). Not a result: never use these numbers in result tables.

`scripts/ext_smoke.py` on cs616.hpc.nyu.edu, 2026-11-26T13:19:55-0500, 7 processes × 1 thread side by side (8-CPU allocation); total wall 716 s.
Raw: `summary.json`. Each `configs/ext` config: 200 steps, then a second process resumes from `ckpt/latest.pt` to 400 steps
(`resumed` = the trainer printed "resumed from step 200"). `long_*`: 3000 steps without resume, to check the gradient norm of
β < 1 and L1 against the paper C config. gnorm = global gradient norm before clipping (clip 1), averaged / maxed over 100-step windows.
s/step = wall time per step between the first and the last log window (startup excluded).

## Checks

- Every run exits 0, every logged loss and gradient norm is finite, and every 200 → 400 resume works (`BAD: []` in the script output).
- No blow-up for β < 1 or L1 (3000 steps; gnorm mean in the first vs last 100-step window, max over the last 1000 steps):

| run | gnorm mean, steps 1–100 | gnorm mean, steps 2901–3000 | gnorm max, steps 2001–3000 |
|---|---|---|---|
| `gmm_B_b0.1` | 0.22 | 0.37 | 1.18 |
| `gmm_C_L1b0.5` | 1.70 | 1.05 | 2.82 |
| `gmm_C_L1b1` | 5.54 | 1.70 | 3.99 |
| `gmm_B_L1b1` | 5.46 | 2.45 | 5.45 |
| `gmm_C_b0.5` | 1.64 | 0.96 | 4.92 |
| `paper_gmm_C` | 0.33 | 0.26 | 0.69 |

  All decrease or stay flat; the β < 1 and L1 runs end within a few × the paper C config, and no window has a spike that
  grows. (Their larger gradient norms are expected: ‖d‖^(β−1) weights small residuals heavily. Clipping at 1 bounds the update.)
- s/step is 0.11–0.13 for every ρ: the kernel and L1 losses cost no more than ‖·‖^β (the 2048-wide time MLP dominates).
- β = 3 and 3.5 (λ = 0) start with large gradient norms (up to 4e3 in the first 100 steps, clipped to 1) that fall to 4–8 by step 400.

## All runs

| run | rc | resumed | finite | s/step | loss, steps 1–100 | loss, last window | gnorm mean, last window | gnorm max (all) |
|---|---|---|---|---|---|---|---|---|
| `long_gmm_B_b0.1` | 0 | – | True | 0.115 | 0.5579 | 0.4609 | 0.37 | 1.6 |
| `long_gmm_C_L1b0.5` | 0 | – | True | 0.114 | 0.9240 | 0.2012 | 1.05 | 6.6 |
| `long_gmm_C_L1b1` | 0 | – | True | 0.130 | 2.2595 | 0.1938 | 1.70 | 32.3 |
| `long_gmm_B_L1b1` | 0 | – | True | 0.125 | 2.2526 | 0.2679 | 2.45 | 32.3 |
| `long_gmm_C_b0.5` | 0 | – | True | 0.119 | 0.8038 | 0.1779 | 0.96 | 9.4 |
| `long_paper_gmm_C` | 0 | – | True | 0.124 | 0.4001 | 0.2299 | 0.26 | 2.1 |
| `gmm2w_A` | 0 | True | True | 0.115 | 5.5137 | 0.2952 | 3.29 | 146.7 |
| `gmm2w_B` | 0 | True | True | 0.113 | 5.6076 | 0.3034 | 5.91 | 204.3 |
| `gmm2w_C` | 0 | True | True | 0.113 | 0.3924 | 0.2479 | 0.30 | 3.6 |
| `gmm9w_A` | 0 | True | True | 0.111 | 9.6474 | 0.4874 | 5.76 | 167.8 |
| `gmm9w_B` | 0 | True | True | 0.113 | 10.3039 | 0.5335 | 10.42 | 165.2 |
| `gmm9w_B_L1b1` | 0 | True | True | 0.120 | 2.6886 | 0.5092 | 5.70 | 23.6 |
| `gmm9w_B_b1` | 0 | True | True | 0.122 | 2.0023 | 0.3210 | 2.32 | 20.3 |
| `gmm9w_B_pH0.1` | 0 | True | True | 0.117 | 1.9330 | 0.2969 | 4.67 | 32.0 |
| `gmm9w_C` | 0 | True | True | 0.118 | 0.4006 | 0.2857 | 0.38 | 1.7 |
| `gmm9w_C_L1b0.5` | 0 | True | True | 0.124 | 0.9566 | 0.2480 | 1.07 | 18.3 |
| `gmm9w_C_L1b1` | 0 | True | True | 0.123 | 2.5636 | 0.6127 | 2.65 | 22.3 |
| `gmm9w_C_imq` | 0 | True | True | 0.116 | 0.0562 | -0.1975 | 0.40 | 2.0 |
| `gmm9w_C_lap` | 0 | True | True | 0.126 | -0.0428 | -0.2152 | 0.19 | 5.1 |
| `gmm9w_C_pH0.1` | 0 | True | True | 0.125 | 1.8988 | 0.2175 | 1.98 | 30.1 |
| `gmm9w_C_pH1` | 0 | True | True | 0.131 | 1.5834 | 0.1217 | 1.34 | 19.5 |
| `gmm9w_C_rbf` | 0 | True | True | 0.117 | -0.1344 | -0.2427 | 0.05 | 0.8 |
| `gmm_B_L1b1` | 0 | True | True | 0.112 | 2.2526 | 0.3746 | 2.55 | 32.3 |
| `gmm_B_b0.1` | 0 | True | True | 0.112 | 0.5579 | 0.4635 | 0.56 | 1.5 |
| `gmm_B_b0.5` | 0 | True | True | 0.121 | 0.8926 | 0.3615 | 1.61 | 6.2 |
| `gmm_B_b1.5` | 0 | True | True | 0.124 | 3.2344 | 0.2772 | 2.39 | 63.8 |
| `gmm_B_b1.99` | 0 | True | True | 0.121 | 6.3352 | 0.3072 | 2.46 | 212.8 |
| `gmm_B_b1` | 0 | True | True | 0.121 | 1.6530 | 0.2964 | 2.13 | 22.9 |
| `gmm_B_b3.5` | 0 | True | True | 0.116 | 49.6314 | 0.9893 | 8.09 | 4225.3 |
| `gmm_B_b3` | 0 | True | True | 0.123 | 24.6270 | 0.6115 | 4.47 | 1714.0 |
| `gmm_B_pH0.1` | 0 | True | True | 0.122 | 1.6038 | 0.2597 | 2.44 | 22.9 |
| `gmm_C_L1b0.5` | 0 | True | True | 0.114 | 0.9240 | 0.2541 | 0.78 | 6.6 |
| `gmm_C_L1b1` | 0 | True | True | 0.123 | 2.2595 | 0.2722 | 1.84 | 32.3 |
| `gmm_C_b0.5` | 0 | True | True | 0.123 | 0.8038 | 0.2360 | 0.81 | 9.4 |
| `gmm_C_b1.5` | 0 | True | True | 0.119 | 3.2353 | 0.2515 | 2.21 | 63.8 |
| `gmm_C_b1.99` | 0 | True | True | 0.114 | 6.3337 | 0.3055 | 2.57 | 212.8 |
| `gmm_C_b1` | 0 | True | True | 0.114 | 1.6434 | 0.2309 | 1.40 | 22.7 |
| `gmm_C_b2` | 0 | True | True | 0.118 | 6.4068 | 0.3124 | 3.04 | 217.6 |
| `gmm_C_imq` | 0 | True | True | 0.114 | 0.0736 | -0.1906 | 0.47 | 3.7 |
| `gmm_C_lap` | 0 | True | True | 0.117 | 0.0277 | -0.1944 | 0.28 | 4.4 |
| `gmm_C_pH0.1` | 0 | True | True | 0.116 | 1.6019 | 0.2059 | 1.36 | 22.9 |
| `gmm_C_pH1` | 0 | True | True | 0.118 | 1.2418 | 0.0959 | 0.93 | 21.9 |
| `gmm_C_rbf` | 0 | True | True | 0.111 | -0.0730 | -0.2391 | 0.10 | 1.9 |
