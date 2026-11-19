# ddm2d: reproducing the 2D experiments of Distributional Diffusion Models (De Bortoli et al., ICML 2025)

Status (S1, 2026-11-19): audit, plan, package, tests, smoke checklist and CPU benchmark done; paper-scale training not yet run.
S4 rewrites this README with the results.

- Audit of the paper and reference code: `docs/audit.md`. Plan and decisions: `docs/plan.md`. Settings table: `docs/paper_vs_implementation.md`.
- Spec: `docs/task_spec.md`. Smoke checklist: `docs/smoke_checklist.md`. CPU benchmark: `docs/benchmark.md`.

```bash
PY=/scratch/wg2381/.conda/envs/tmap/bin/python
cd ddm2d
OMP_NUM_THREADS=4 $PY -m pytest -q tests                       # 72 unit tests
# train one arm (A standard, B beta=2/lambda=0, C beta=0.1/lambda=1); resumable, exit 3 at --deadline
OMP_NUM_THREADS=1 PYTHONPATH=src $PY -m ddm2d.train --config configs/paper/gmm_C.json --out runs/paper/gmm_C_s0 --seed 0 --threads 1
# smoke evaluation (MMD vs NFE, posterior std, sample figure)
$PY scripts/smoke_eval.py --runs runs/smoke/gmm_C_s0 --dataset gmm --out runs/smoke/eval_example
```
Package `src/ddm2d/`: `schedule` (Eq. 2–4, sigmoid weight), `data` (GMM, FFJORD checkerboard), `model` (App. I.1 MLP), `losses` (Eq. 6, 14),
`samplers` (Algorithm 2), `mmd` (RBF MMD², float64, chunked), `posterior` (exact GMM posterior), `train`, `evaluate`.
