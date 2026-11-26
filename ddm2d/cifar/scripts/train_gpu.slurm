#!/bin/bash
# One E3 CIFAR-10 run on one L40S: training (ddmcifar.train), then the evaluation protocol of cifar/runs/PLAN.md (ddmcifar.evaluate).
# Modelled on transport_map scripts/exp_gpu.slurm (main, 4b9e8f9).
#   RUN=cifar_B sbatch --job-name=ddmx_cifar_B --time=HH:MM:SS /scratch/wg2381/tmap-ddm2d/ddm2d/cifar/scripts/train_gpu.slurm
# Requeue-safe: the run directory is fixed (cifar/runs/<RUN>); ddmcifar.train resumes from ckpt/latest.pt. 5 min before the wall
# limit Slurm sends USR1 to this shell: during training it is forwarded to python (checkpoint, exit 3, requeue); during evaluation the
# job requeues itself and finished evaluations (eval/*.json) are skipped next time.
#SBATCH --account=torch_pr_527_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@300
#SBATCH --output=/scratch/wg2381/tmap-ddm2d/ddm2d/cifar/runs/slurm-%x-%j.out
ROOT=/scratch/wg2381/tmap-ddm2d/ddm2d
: "${RUN:?set RUN}"
cd "$ROOT" || exit 1
for v in $(env | grep -oE '^(CLAUDECODE|CLAUDE_CODE_[A-Z_]*)='); do unset "${v%=}"; done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source /scratch/wg2381/transport_map/env/activate.sh || exit 1    # not under `set -u`
export HF_HUB_OFFLINE=1 PYTHONPATH="$ROOT/src:$ROOT/cifar/src:$PYTHONPATH"
set -o pipefail
OUT=cifar/runs/$RUN
mkdir -p "$OUT/eval"
echo "[$(date)] job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} on $(hostname); run $RUN; git $(git -C "$ROOT" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

PID=""
STAGE=train
on_usr1() {
  echo "[$(date)] got USR1 during $STAGE"
  if [ "$STAGE" = train ]; then
    kill -s USR1 "$PID" 2>/dev/null
  else
    kill "$PID" 2>/dev/null; echo "requeueing $SLURM_JOB_ID for the remaining evaluation"
    scontrol requeue "$SLURM_JOB_ID"; exit 0
  fi
}
trap on_usr1 USR1
trap 'kill -s TERM "$PID" 2>/dev/null' TERM
run() {
  "$@" & PID=$!
  while true; do wait "$PID"; CODE=$?; kill -0 "$PID" 2>/dev/null || break; done
  return $CODE
}

run python -m ddmcifar.train --config "cifar/configs/$RUN.json" --out "$OUT" >> "$OUT/train.log" 2>&1
CODE=$?
echo "[$(date)] ddmcifar.train exited with $CODE"
if [ "$CODE" = "3" ]; then
  echo "checkpoint written; requeueing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 0
fi
[ "$CODE" = "0" ] && [ -f "$OUT/DONE.json" ] || { echo "training failed"; exit 1; }

STAGE=eval
SPREAD=--spread; [ "$RUN" = cifar_A ] && SPREAD=
# checkpoint selection (both NFE 10 and 100 for every arm), then the full NFE grid on the final (30k) EMA weights
for s in snap_0010000 snap_0020000; do
  run python -m ddmcifar.evaluate --run "$OUT" --ckpt $s --nfe 10,100 >> "$OUT/eval/eval.log" 2>&1 || echo "eval $s failed"
done
run python -m ddmcifar.evaluate --run "$OUT" --ckpt snap_0030000 --nfe 2,4,8,10,15,30,100 --grid-nfe 10,100 $SPREAD \
  >> "$OUT/eval/eval.log" 2>&1 || echo "eval snap_0030000 failed"
tail -n 12 "$OUT/eval/eval.log"
echo "[$(date)] done"
