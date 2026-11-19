# runs/smoke: SMOKE outputs only. Never use anything here as a paper-scale result.

- `sanity3k_gmm_{A,B,C}_s0/` (S1, 2026-11-19): the **paper** config `configs/paper/gmm_<arm>.json` stopped early with `--max-steps 3000`
  (3 processes × 2 threads, ~170 s each). Purpose: check that all three arms learn before S2 launches anything. Checkpoints gitignored.
  Command: `PYTHONPATH=src python -m ddm2d.train --config configs/paper/gmm_<arm>.json --out runs/smoke/sanity3k_gmm_<arm>_s0 --seed 0 --threads 2 --max-steps 3000`
- `sanity3k_gmm_eval/`: `scripts/smoke_eval.py` on those EMA weights (NFE 5, 20, 100; 4096 samples; churn 1; unbiased MMD², σ = 1;
  posterior std at 5 times with 1024 pairs).
- `sanity20k_gmm_{A,C}_s0/` (S1): the paper config stopped at 20,000 steps (2 processes × 3 threads). `sanity20k_gmm_eval/`:
  - `eval.json`, `samples.png`: NFE 5, 10, 20, 50, 100. MMD²_u: A 1.39e-2, 5.5e-3, 2.0e-3, 1.1e-3, 1.7e-3; C −4e-5, −2.3e-4, −3e-5, 3.8e-4, 8.2e-4.
  - `nfe1000_clamp.json`: NFE 1000, MMD²_u with the low safety clamp (default) / without: A 1.44e-2 / 4.17e-2; C 6.6e-3 / 1.89e-2.
  - `denoiser_error_vs_t.json`: rmse of x̂ (A) and of the C particle mean (64 draws) against the exact posterior mean, and C's particle std vs
    the exact posterior std, at t ∈ [0.001, 0.99] (eval set k = 50).
  Smoke finding (not a result): at 20k steps, MMD² grows with NFE beyond ≈ 50 for both arms. Below t ≈ 0.02 the denoiser error (≈ 0.03; ≈ 0.1
  when extrapolated below the training range t < 0.01) exceeds the true posterior std (0.01–0.02), and C's particle std floors at ≈ 0.06, so
  many fine steps near t = 0 accumulate error. At t ≥ 0.9, C's std (1.16–1.19) is below the posterior's (1.69–1.77): the sigmoid weight
  (b = 0) is ≈ 1e-2 at t = 0.9 and ≈ 1e-4 at t = 0.99. Whether 100k steps fix either is for S2/S3 to check on the paper runs.
