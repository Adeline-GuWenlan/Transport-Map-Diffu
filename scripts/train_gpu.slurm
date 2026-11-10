#!/bin/bash
# Real run: DiT-B/2 transport map on CIFAR-10 with the energy-score loss (configs/base_gpu.yaml). NOT submitted by
# the pipeline. Submit by hand:   sbatch scripts/train_gpu.slurm
# Optional:  RUN=runs/<name> sbatch scripts/train_gpu.slurm   (default run dir below; keep it fixed across requeues)
#            TRAIN_SET="train.max_steps=30000" sbatch ...         (space-separated config overrides for tmap.train --set)
#
# Requeue-safe: the run directory is fixed, and tmap.train resumes from <RUN>/ckpt/latest.pt (model, EMA, optimizer,
# step, sampler + noise RNG). 5 min before the wall limit Slurm sends USR1 to this batch shell; it is forwarded to
# python, which writes a checkpoint and exits with code 3; the script then requeues itself. Preemption (SIGTERM)
# takes the same path. On a normal finish (exit 0) it runs the final evaluation (D1 + FID-10k, EMA and raw).
#SBATCH --account=torch_pr_527_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300
#SBATCH --job-name=tmap_gpu_cifar10_ditb2
#SBATCH --output=/scratch/wg2381/transport_map/runs/slurm-%x-%j.out
# Assumption: QOS gpu48 (max 2 days, from `sacctmgr show qos` on 2026-11-06) applies to l40s_public; 24 h is inside it.
# Stage 2 estimates ~0.5-0.7 s/step for DiT-B/2 at N = 512 (200k steps ~ 1-1.5 days), i.e. 1-2 requeues.

ROOT=/scratch/wg2381/transport_map
CONFIG=${CONFIG:-$ROOT/configs/base_gpu.yaml}
RUN=${RUN:-$ROOT/runs/gpu_base_cifar10_dit-b2}
cd "$ROOT" || exit 1
source "$ROOT/env/activate.sh" || exit 1      # not under `set -u` (conda activate.d reads unset variables)
set -o pipefail
mkdir -p "$RUN"
echo "[$(date)] job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} on $(hostname); run $RUN"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# First start only: environment check on the GPU (pinned versions, DiT clone, CUDA forward/backward at N = 512).
if [ "${SLURM_RESTART_COUNT:-0}" = "0" ] && [ ! -f "$RUN/ckpt/latest.pt" ]; then
  python scripts/check_env.py --config "$CONFIG" --device cuda --reps 5 2>&1 | tee "$RUN/check_env.log" || {
    echo "check_env failed; not training"; exit 1; }
fi

python -m tmap.train --config "$CONFIG" --out "$RUN" ${TRAIN_SET:+--set $TRAIN_SET} >> "$RUN/train.log" 2>&1 &
PID=$!
forward() { echo "[$(date)] got $1, forwarding to python ($PID)"; kill -s "$1" "$PID" 2>/dev/null; }
trap 'forward USR1' USR1
trap 'forward TERM' TERM
# `wait` returns early when a trapped signal arrives; keep waiting until python has actually exited.
while true; do
  wait "$PID"; CODE=$?
  kill -0 "$PID" 2>/dev/null || break
done
echo "[$(date)] tmap.train exited with $CODE"

if [ "$CODE" = "3" ]; then
  echo "checkpoint written; requeueing $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
if [ "$CODE" != "0" ]; then
  exit "$CODE"
fi

# Final evaluation: per-class held-out D1 (1000/class) with baselines, nearest neighbours, FID-10k (clean-fid,
# legacy_tensorflow, CIFAR-10 train stats). For the paper-standard FID-50k use --fid-num-gen 50000.
python -m tmap.evaluate --ckpt "$RUN/ckpt/latest.pt" --out "$RUN/eval_final.json" --device cuda \
  --n-per-class 1000 --fid-num-gen 10000 --nn-grid "$RUN/nearest_neighbours.png" 2>&1 | tee "$RUN/eval_final.log"
python -m tmap.sample --ckpt "$RUN/ckpt/latest.pt" --out "$RUN/samples_final_ema.png" --weights ema --device cuda
python scripts/plot_run.py "$RUN"
