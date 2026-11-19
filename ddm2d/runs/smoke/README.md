# runs/smoke: SMOKE outputs only. Never use anything here as a paper-scale result.

- `sanity3k_gmm_{A,B,C}_s0/` (S1, 2026-11-19): the **paper** config `configs/paper/gmm_<arm>.json` stopped early with `--max-steps 3000`
  (3 processes × 2 threads, ~170 s each). Purpose: check that all three arms learn before S2 launches anything. Checkpoints gitignored.
  Command: `PYTHONPATH=src python -m ddm2d.train --config configs/paper/gmm_<arm>.json --out runs/smoke/sanity3k_gmm_<arm>_s0 --seed 0 --threads 2 --max-steps 3000`
- `sanity3k_gmm_eval/`: `scripts/smoke_eval.py` on those EMA weights (NFE 5, 20, 100; 4096 samples; churn 1; unbiased MMD², σ = 1;
  posterior std at 5 times with 1024 pairs).
