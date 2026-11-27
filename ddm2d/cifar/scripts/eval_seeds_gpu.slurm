#!/bin/bash
# X4 eval-only job (E3): sampler-seed spread of FID-10k on the final EMA weights, plus (with REAL=1) the real-data FID-10k floor.
#   RUNS=cifar_A,cifar_B REAL=1 sbatch --job-name=ddmx_evseed1 --time=02:30:00 cifar/scripts/eval_seeds_gpu.slurm
# (pass RUNS with sbatch's inherited env, not --export=...,RUNS=a,b, which splits at the comma). Requeue-safe: finished JSONs are skipped.
# Same header as train_gpu.slurm (modelled on transport_map scripts/exp_gpu.slurm, main 4b9e8f9).
#SBATCH --account=torch_pr_527_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/wg2381/tmap-ddm2d/ddm2d/cifar/runs/eval/slurm-%x-%j.out
ROOT=/scratch/wg2381/tmap-ddm2d/ddm2d
: "${RUNS:?set RUNS}"
cd "$ROOT" || exit 1
for v in $(env | grep -oE '^(CLAUDECODE|CLAUDE_CODE_[A-Z_]*)='); do unset "${v%=}"; done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source /scratch/wg2381/transport_map/env/activate.sh || exit 1    # not under `set -u`
export HF_HUB_OFFLINE=1 PYTHONPATH="$ROOT/src:$ROOT/cifar/src:$PYTHONPATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[$(date)] job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} on $(hostname); runs $RUNS; git $(git -C "$ROOT" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
PID=""
trap 'kill "$PID" 2>/dev/null; echo "[$(date)] USR1: requeueing"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
EXTRA=; [ "${REAL:-0}" = 1 ] && EXTRA=--real
python cifar/scripts/eval_seeds.py --runs "$RUNS" --ckpt snap_0030000 --nfe 2,4,10,100 --seeds 1,2 $EXTRA &
PID=$!
while true; do wait "$PID"; CODE=$?; kill -0 "$PID" 2>/dev/null || break; done
echo "[$(date)] eval_seeds exited with $CODE"
