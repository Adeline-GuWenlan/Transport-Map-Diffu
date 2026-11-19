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
