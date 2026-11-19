# Health check of the paper-scale runs (S2, 2026-11-19)

Jobs 23769609 (seed 0, cs619), 23769610 (seed 1, cs619), 23769636 (seed 2, cs603), all started 17:57 EST (`runs/paper/JOBS.md`).
Tool: `PYTHONPATH=src python scripts/health_check.py --out runs/paper/health/<name>.json` (run in the S2 allocation, not on the job nodes).
It reads `log.jsonl` (100-step windows), loads `ckpt/latest.pt` into a fresh `Trainer` (asserts config, seed and data fingerprint; restores
model, EMA, Adam, step, generator, data order, global RNGs), and runs the EMA weights and the latest EMA snapshot on a fixed probe batch.

| check | when | file | result |
|---|---|---|---|
| h1 | +2 min, step 900–1200 | `health/h1_step900.json` | 18/18 logging, all windows finite |
| h2 | +12 min, step 5300–7200 | `health/h2_ckpt5000.json` | 18/18 finite; 18/18 `step_0005000` checkpoints load, EMA probe finite |
| h3 | +25 min, step 10500–14300 | `health/h3_ema10k.json` | 18/18 finite; 18/18 `latest.pt` (step 10000) load; 18/18 `ema_0010000.pt` load, probe finite |

**[E]** from `h3_ema10k.json`:
- s/step (log wall times, windows 2..last): seed 0 and 1 on cs619 0.128–0.143; seed 2 on cs603 0.105–0.127. The benchmark is 0.114
  (6 × 1, cs601). The 10–25 % slowdown on cs619 is plausibly because two of these jobs (12 processes) share that node **[G]**. The
  slowest projected finish is ≈ 3.55 h more after step 10.5k, i.e. ≈ 4.0 h in total, inside the 4.8 h job deadline (end − 12 min). If a
  run is slower, the job checkpoints and submits a continuation.
- Loss after 10k steps, 100-step window: GMM A 0.210–0.212, B 0.204–0.212, C 0.229–0.231; checkerboard A 0.406–0.409, B 0.396–0.405,
  C 0.233. The loss values of A/B (MSE-type) and C (energy score) are not comparable with each other.
- Memory: `sstat` MaxRSS 6.2–6.5 GB per job (6 processes), of the 12 GB requested. Disk: 7.6 GB under `runs/paper` at step ≈ 11k;
  expected ≈ 0.9 GB per run at the end (`latest`, 2 rolling `step_*`, `final`, 10 EMA snapshots of 34 MB).
- No `FAILED.txt`, no non-finite window, and every pre-clip grad norm is finite.

Verdict: all 18 runs are healthy. Nothing was relaunched.
