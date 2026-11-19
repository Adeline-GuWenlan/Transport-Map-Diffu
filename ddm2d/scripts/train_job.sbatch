#!/bin/bash
# Paper-scale training job for ddm2d: K packed training processes x 1 thread, resumable and self-requeueing.
#
# Usage (from ddm2d/):  sbatch --job-name=ddm2d_train_s0 --export=ALL,RUNLIST=runs/paper/runlists/seed0.txt scripts/train_job.sbatch
#   RUNLIST  file, one run per line: "<config.json> <out_dir> <seed>" (paths relative to ddm2d/; '#' lines ignored).
#            Passed as a file because `sbatch --export=VAR=a,b` splits at commas.
#   SEG      segment number (default 1); MAXSEG (default 5) caps the continuation chain.
#   JOBIDS   file the continuation ids are appended to (default: the chain's jobids/S2.txt).
#   Tests only: DEADLINE_OVERRIDE (unix time), SBATCH_CMD (replaces sbatch), JOBLOG (default runs/paper/joblog.tsv).
# Each run: python -m ddm2d.train ... --threads 1 --deadline <job end - 12 min>. The trainer resumes from ckpt/latest.pt, checkpoints and
# exits 3 at the deadline, exits 0 with DONE.json when finished. If any run exited 3, this script submits a continuation of itself with
# --dependency=afterany:$SLURM_JOB_ID before it ends. Runs that already have DONE.json are skipped.
#SBATCH --account=torch_pr_527_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/wg2381/claude_jobs/logs/ddm2d_train_%x-%j.out
set -uo pipefail

DDM=/scratch/wg2381/tmap-ddm2d/ddm2d
PY=/scratch/wg2381/.conda/envs/tmap/bin/python
RUNLIST=${RUNLIST:?RUNLIST not set}
SEG=${SEG:-1}
MAXSEG=${MAXSEG:-5}
JOBIDS=${JOBIDS:-/scratch/wg2381/claude_jobs/ddm2d/jobids/S2.txt}
cd "$DDM" || exit 1
JOBLOG=${JOBLOG:-runs/paper/joblog.tsv}; mkdir -p "$(dirname "$JOBLOG")"
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

NCPU=$("$PY" -c 'import os; print(len(os.sched_getaffinity(0)))')
END_STR=$(squeue -h -j "$SLURM_JOB_ID" -o %e)
END_TS=$(date -d "$END_STR" +%s 2>/dev/null || echo $(( $(date +%s) + 5*3600 )))
DEADLINE=${DEADLINE_OVERRIDE:-$(( END_TS - 12*60 ))}   # DEADLINE_OVERRIDE: tests only
echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) seg $SEG/$MAXSEG on $(hostname) cpus=$NCPU restart=${SLURM_RESTART_COUNT:-0} start=$(date -Is) end=$END_STR deadline=$(date -d @$DEADLINE -Is) runlist=$RUNLIST"
echo -e "$SLURM_JOB_ID\t$SLURM_JOB_NAME\tseg$SEG\trestart${SLURM_RESTART_COUNT:-0}\t$(hostname)\t$(date -u +%FT%TZ)\tstart\t$RUNLIST" >> "$JOBLOG"

names=(); pids=()
while read -r cfg out seed; do
  [[ -z "${cfg:-}" || "$cfg" == \#* ]] && continue
  if [[ -f "$out/DONE.json" ]]; then echo "skip $out (DONE)"; continue; fi
  mkdir -p "$out"
  "$PY" -m ddm2d.train --config "$cfg" --out "$out" --seed "$seed" --threads 1 --deadline "$DEADLINE" >> "$out.out" 2>&1 &
  pids+=($!); names+=("$out")
  echo "started $out (pid $!)"
done < "$RUNLIST"
if (( ${#pids[@]} > NCPU )); then echo "WARNING: ${#pids[@]} processes on $NCPU CPUs"; fi

need_cont=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"; rc=$?
  echo "$(date -Is) ${names[$i]} exit $rc"
  echo -e "$SLURM_JOB_ID\t$SLURM_JOB_NAME\tseg$SEG\t${names[$i]}\texit$rc\t$(date -u +%FT%TZ)" >> "$JOBLOG"
  if (( rc == 3 )); then need_cont=1
  elif (( rc != 0 )); then echo "FAILED exit $rc in job $SLURM_JOB_ID seg $SEG at $(date -Is)" >> "${names[$i]}/FAILED.txt"; fi
done

if (( need_cont )); then
  if (( SEG < MAXSEG )); then
    cid=$(${SBATCH_CMD:-sbatch} --parsable --job-name="$SLURM_JOB_NAME" --dependency=afterany:"$SLURM_JOB_ID" \
          --export=ALL,RUNLIST="$RUNLIST",SEG=$((SEG + 1)),MAXSEG="$MAXSEG",JOBIDS="$JOBIDS" "$DDM/scripts/train_job.sbatch")
    if [[ -n "$cid" ]]; then
      echo -e "$cid $SLURM_JOB_NAME $(date -u +%FT%TZ) continuation seg$((SEG + 1)) of $SLURM_JOB_ID ($RUNLIST)" >> "$JOBIDS"
      echo "submitted continuation $cid"
    else
      echo "ERROR: continuation sbatch failed"
    fi
  else
    echo "MAXSEG $MAXSEG reached; not resubmitting"
  fi
fi
echo "job $SLURM_JOB_ID end $(date -Is)"
