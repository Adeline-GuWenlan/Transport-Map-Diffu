#!/bin/bash
# Experiments A / B (docs/experiments_AB.md): one 30k-step DiT-B/2 CIFAR-10 run on one L40S, then evaluation of
# every weight snapshot (ckpt/step*.pt): FID-10k (clean-fid legacy_tensorflow vs CIFAR-10 train), held-out per-class
# pixel D1 (1000/class), Inception-pool3 and DINOv2 feature D1, class checks, blockiness, spectra, sample grids.
#
#   CONFIG=configs/expA_cifar10_dit-b2_pixel.yaml RUN=runs/expA_s0 TRAIN_SET="seed=0" \
#     sbatch --job-name=tmap_A_s0 --time=HH:MM:SS scripts/exp_gpu.slurm
#
# Requeue-safe: the run directory is fixed; tmap.train resumes from <RUN>/ckpt/latest.pt. 5 min before the wall
# limit Slurm sends USR1 to this shell: during training it is forwarded to python (checkpoint, exit 3, requeue);
# during evaluation the job requeues itself and the finished snapshots (eval/<step>.json) are skipped next time.
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
#SBATCH --output=/scratch/wg2381/transport_map/runs/slurm-%x-%j.out

ROOT=/scratch/wg2381/transport_map
: "${CONFIG:?set CONFIG}" "${RUN:?set RUN}"
case "$CONFIG" in /*) ;; *) CONFIG=$ROOT/$CONFIG ;; esac
case "$RUN" in /*) ;; *) RUN=$ROOT/$RUN ;; esac
cd "$ROOT" || exit 1
for v in $(env | grep -oE '^(CLAUDECODE|CLAUDE_CODE_[A-Z_]*)='); do unset "${v%=}"; done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$ROOT/env/activate.sh" || exit 1      # not under `set -u` (conda activate.d reads unset variables)
export HF_HUB_OFFLINE=1                        # DINOv2 weights are already in $HF_HOME; no hub calls from jobs
set -o pipefail
mkdir -p "$RUN/eval"
echo "[$(date)] job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} on $(hostname); config $CONFIG; run $RUN; set: ${TRAIN_SET:-}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

PID=""
STAGE=train
on_usr1() {
  echo "[$(date)] got USR1 during $STAGE"
  if [ "$STAGE" = train ]; then
    kill -s USR1 "$PID" 2>/dev/null           # python checkpoints and exits 3; the train block requeues
  else
    kill "$PID" 2>/dev/null; echo "requeueing $SLURM_JOB_ID for the remaining evaluation"
    scontrol requeue "$SLURM_JOB_ID"; exit 0
  fi
}
trap on_usr1 USR1
trap 'kill -s TERM "$PID" 2>/dev/null' TERM
run() {  # run a command in the background and wait for it, so that traps fire while it runs
  "$@" & PID=$!
  while true; do wait "$PID"; CODE=$?; kill -0 "$PID" 2>/dev/null || break; done
  return $CODE
}

run python -m tmap.train --config "$CONFIG" --out "$RUN" ${TRAIN_SET:+--set $TRAIN_SET} >> "$RUN/train.log" 2>&1
CODE=$?
echo "[$(date)] tmap.train exited with $CODE"
if [ "$CODE" = "3" ]; then
  echo "checkpoint written; requeueing $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; exit 0
fi
[ "$CODE" = "0" ] || exit "$CODE"

STAGE=eval
for f in "$RUN"/ckpt/step*.pt; do
  s=$(basename "$f" .pt)
  [ -f "$RUN/eval/$s.json" ] && continue
  echo "[$(date)] evaluating $s"
  run python -m tmap.evaluate --ckpt "$f" --out "$RUN/eval/$s.json.tmp" --device cuda --n-per-class 1000 \
    --fid-num-gen 10000 --feature-ed inception,dinov2 --nn-grid "$RUN/eval/${s}_nn.png" > "$RUN/eval/$s.log" 2>&1 \
    || { echo "evaluate failed on $s (see $RUN/eval/$s.log)"; continue; }
  for w in ema raw; do
    run python -m tmap.sample --ckpt "$f" --out "$RUN/eval/${s}_samples_$w.png" --weights $w --device cuda \
      --n-per-class 10 >> "$RUN/eval/$s.log" 2>&1
  done
  mv "$RUN/eval/$s.json.tmp" "$RUN/eval/$s.json"
  tail -n 20 "$RUN/eval/$s.log"
done
run python scripts/plot_run.py "$RUN" > /dev/null 2>&1 || echo "plot_run.py failed"
echo "[$(date)] done"
