# Stage 1 numerical checks

CPU-only checks for `docs/energy_score_loss.tex` (Section 9). Total wall time is about 15 min on 8 cores.

Environment: `/scratch/wg2381/.conda/envs/tmap-tex` (conda-forge: python 3.11, numpy, scipy, matplotlib,
CPU pytorch 2.14.1, tectonic, poppler, pypdf). Creation log: `logs/conda_create.log`.

    bash run_all.sh                       # runs every check, then make_tex_results.py
    cd .. && tectonic energy_score_loss.tex   # rebuild the PDF

| file | what it checks |
|---|---|
| `es_core.py` | estimators: U/V-statistic energy distance, energy-score loss, NaN-safe `torch_pow_dist`, grouped (per-class) minibatch loss, closed forms |
| `check_a_estimator.py` | (a1) unbiasedness, P = Q; (a2) 1-D Gaussian closed form; (a3) Fourier constant C(d, beta); (a4) variance vs m |
| `check_b_beta2.py` | (b) beta = 2 vs beta = 1 on N(0,1) vs N(0,4); (b3) weighted interaction lambda |
| `check_c_toy2d.py`, `plot_c.py` | (c) 2-D MLP map to 8 Gaussians: corrected loss vs readings of the proposed formula |
| `check_c2_long.py` | (c2) corrected loss, 3000 vs 20000 steps |
| `check_d_conditional.py` | (d) per-class vs pooled interaction term, 1-D two-class example |
| `check_d2_conditional_2d.py` | (d2) same, 2-D interleaved classes (secondary) |
| `check_e_grad_at_zero.py` | (e) PyTorch gradient of the norm at 0; NaN trap; cdist matmul-path accuracy |
| `check_f_zero_init.py` | (f) zero-initialised output layer (as in DiT): safe vs naive distance |
| `make_tex_results.py` | results/*.json -> results/results_macros.tex and tables used by the note |
