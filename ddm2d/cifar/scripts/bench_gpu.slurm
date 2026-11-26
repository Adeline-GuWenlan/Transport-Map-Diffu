#!/bin/bash
# GPU bench for the E3 plan (cifar/scripts/bench.py). Submit from anywhere:
#   sbatch --job-name=ddmx_cifar_bench /scratch/wg2381/tmap-ddm2d/ddm2d/cifar/scripts/bench_gpu.slurm
#SBATCH --account=torch_pr_527_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/wg2381/tmap-ddm2d/ddm2d/cifar/runs/slurm-%x-%j.out
ROOT=/scratch/wg2381/tmap-ddm2d/ddm2d
cd "$ROOT" || exit 1
for v in $(env | grep -oE '^(CLAUDECODE|CLAUDE_CODE_[A-Z_]*)='); do unset "${v%=}"; done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source /scratch/wg2381/transport_map/env/activate.sh || exit 1    # not under `set -u`
export HF_HUB_OFFLINE=1 PYTHONPATH="$ROOT/src:$ROOT/cifar/src:$PYTHONPATH"
echo "[$(date)] job $SLURM_JOB_ID on $(hostname); git $(git -C "$ROOT" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python cifar/scripts/bench.py --out "cifar/runs/bench_$SLURM_JOB_ID"
echo "[$(date)] bench exit $?"
