# CPU benchmark of the paper config (started by S1, 2026-11-19)

Node cs607, Slurm job 23760197 (8 CPUs, `cpu_short`). Paper config `configs/paper/gmm_<arm>.json`: batch 128, m = 32 (B, C), App. I.1 model
(8.57 M parameters). Each process: 5 warm-up steps, then timed training steps (forward, backward, clip, Adam, EMA); logging and checkpointing
excluded (logged every 100 steps, checkpoint every 5000: negligible). **[E]** raw files below.

## Isolated (one process, nothing else running) — `docs/timing_s1.json`, 40 timed steps
| arm | 1 thread | 2 threads | 4 threads | 8 threads |
|---|---|---|---|---|
| A | 0.0884 s | 0.0513 s | 0.0328 s | 0.0193 s |
| B | 0.0836 s | 0.0507 s | 0.0320 s | 0.0204 s |
| C | 0.0946 s | 0.0535 s | 0.0340 s | 0.0238 s |

## Packed (K processes × T threads = 8 CPUs, arms mixed C, A, B, …) — `docs/benchmark_packed_s1.json`, 150 timed steps
| layout | per-process median s/step | worst | h per 100k-step run | runs per 8-CPU-hour | CPU-h per run |
|---|---|---|---|---|---|
| 8 × 1 | 0.107–0.112 | 0.1122 | 3.12 | 2.57 | **3.12** |
| 4 × 2 | 0.060–0.062 | 0.0619 | 1.72 | 2.33 | 3.44 |
| 2 × 4 | 0.035–0.036 | 0.0355 | 0.99 | 2.03 | 3.94 |
| 1 × 8 | 0.0214 | 0.0214 | 0.59 | 1.68 | 4.75 |

Packing costs ≈ 20 % per process at 1 thread (0.11 vs 0.09 s/step) but 8 × 1 still has the highest throughput. Contrary to the 6× slowdown in
`_rules.md`, nothing collapses here because every process keeps to its own thread count.

## Estimate (for S2 to confirm on its job nodes)
- Runs: GMM A, B, C + checkerboard A, C (+ checkerboard B, cheap) × seeds 0, 1, 2 = 15 (18) runs × 100k steps.
- 8 × 1 layout: ≈ 3.1 h wall and 3.1 CPU-h per run; 18 runs ≈ 56 CPU-h, + evaluation; well inside 600 CPU-h. One 8-CPU job of 8 runs finishes
  in ≈ 3.2 h (< 6 h cap); three such jobs (24 CPUs, the chain cap) hold all 18 runs (24 slots) at once.
- 4 × 2 is ≈ 10 % less efficient but finishes each run in ≈ 1.7 h.
- Different `cpu_short` nodes may be slower; the job script must checkpoint and requeue anyway (`ddm2d.train --deadline`, exit code 3).

Reproduce: `python scripts/time_step.py --out docs/timing_s1.json`; `python scripts/bench_packed.py --out docs/benchmark_packed_s1.json`.

## S2 confirmation on a second node (cs601, Slurm job 23766928) — `docs/benchmark_packed_s2.json`, 150 timed steps
| layout | per-process median s/step (range) | worst | h per 100k-step run |
|---|---|---|---|
| 8 × 1 | 0.108–0.118 | 0.1183 | 3.29 |
| 6 × 1 | 0.110–0.114 | 0.1140 | 3.17 |
| 4 × 2 | 0.063–0.075 | 0.0749 | 2.08 |
| 2 × 4 | 0.042–0.044 | 0.0444 | 1.23 |
| 1 × 8 | 0.0305 | 0.0305 | 0.85 |

cs601 is ≈ 5 % slower than cs607 at 1 thread per process and ≈ 40 % slower at 8 threads, so 1-thread processes are also the most
node-robust choice. (The `cpu_h_per_run` field in the JSON divides by 8 CPUs; for 6 × 1 in a 6-CPU job it is 3.17 CPU-h.)

## Chosen layout and budget (S2)
- **3 jobs × 6 CPUs, each = one seed: 6 processes × 1 thread** (GMM A, B, C; checkerboard A, B, C). Arms of one seed share a node, so
  their wall times are directly comparable. 3 × 6 = 18 CPUs ≤ the chain's 24-CPU cap; with the 8-CPU agent job that is 26 of the user's
  32 `cpu_short` CPUs, so all three jobs can run at once.
- Expected: ≈ 0.11–0.12 s/step → ≈ 3.2–3.4 h per run, inside one 5 h job (deadline = end − 12 min; continuation if a node is slower).
- CPU-h (Slurm charges CPUs × elapsed): 3 jobs × 6 CPUs × ≈ 3.4 h ≈ **61 CPU-h** for all 18 runs (seeds 0, 1, 2; checkerboard B
  included, ≈ 10 CPU-h). Worst case with every job using its full 5 h: 90 CPU-h. Both are far inside the 600 CPU-h chain budget, so seeds
  1 and 2 and checkerboard B were launched together with seed 0.
- Rejected: 8 × 1 jobs (no spare cores, and 18 runs do not split into 8s); 4 × 2 (≈ 30 % more CPU-h per run on cs601).
