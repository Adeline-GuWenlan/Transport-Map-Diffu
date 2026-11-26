# ddm2d/cifar: CIFAR-10 trial of distributional diffusion (ddm2dx E3)

Short trial (30k steps, batch 128) of De Bortoli et al. 2025 (arXiv 2502.02483) on class-conditional CIFAR-10, pixel space,
with the App. I.2 U-Net (copied from transport_map `src/tmap/unet.py`, commit 4b9e8f9). Spec: `/scratch/wg2381/claude_jobs/ddm2dx/prompts/_task.md`, E3.

| path | what |
|---|---|
| `src/ddmcifar/unet.py` | U-Net + `DDMUNet` (A: 3 in / 3 out = velocity; B, C: 6 in = x_t ⊕ ξ, 6 out, first 3 dropped) |
| `src/ddmcifar/data.py` | CIFAR-10 `.npy` → [−1, 1], seeded resumable epoch sampler with hflip |
| `src/ddmcifar/losses.py` | A: w‖v̂ − v‖²; B / C: Eq. (14) over flattened images with ρ ∈ {‖·‖^β, pseudo-Huber} |
| `src/ddmcifar/features.py` | frozen DINOv2 ViT-S/14 (copied from transport_map `features.py`), `scale: auto` |
| `src/ddmcifar/sampler.py` | Algorithm 2 (Eq. (4) bridge, churn 1); A: x̂ = x_t + t v̂; ξ-spread vs t |
| `src/ddmcifar/train.py` | Adam, clip 1, cosine warmup 100, EMA 0.9999 (with warm-up), bf16, exact chunked gradient, resume, USR1 → exit 3 |
| `tests/` | CPU tests with a tiny U-Net |

Tests (from `ddm2d/`): `PYTHONPATH=src:cifar/src /scratch/wg2381/.conda/envs/tmap/bin/python -m pytest cifar/tests -q`.

## Runs (E3)
Plan, decisions and the bench-based cost: `runs/PLAN.md`. Job ids and health checks: `runs/JOBS.md`.

- Train + evaluate one run on an L40S: `RUN=cifar_B sbatch --export=ALL,RUN=cifar_B --job-name=ddmx_cifar_B --time=11:40:00 scripts/train_gpu.slurm`
  (requeue-safe; evaluation follows training in the same job, finished evaluations are skipped on restart).
- Evaluate by hand (GPU): `python -m ddmcifar.evaluate --run cifar/runs/cifar_B --ckpt snap_0030000 --nfe 2,4,8,10,15,30,100 --grid-nfe 10,100 --spread`.
- Bench: `sbatch --job-name=ddmx_cifar_bench scripts/bench_gpu.slurm` → `runs/bench_<jobid>/bench.json`.
- CPU smoke (tiny U-Net, proves the path only): `bash cifar/scripts/smoke_cpu.sh` from `ddm2d/` → `runs/smoke/`.
- Health: `python cifar/scripts/health.py` (log summary), `python cifar/scripts/snapshot_peek.py <run> snap_0010000` (CPU grid).
