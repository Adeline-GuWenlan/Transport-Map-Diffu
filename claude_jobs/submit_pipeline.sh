#!/bin/bash
# Submit the 4-stage transport_map agent pipeline.
#
#   bash claude_jobs/submit_pipeline.sh [START] [INTERVAL_HOURS]
#
# START is anything `date -d` understands (default "now"); a time already past starts at once.
# Stage k begins at START + (k-1) * INTERVAL (default 5 h), and never before stage k-1 has ended
# (afterany), so a late stage can't overlap the next one. Job ids are appended to claude_jobs/jobs.txt.
#
# Watch:     squeue -u $USER -n tmap_stage1_loss,tmap_stage2_investigate,tmap_stage3_env,tmap_stage4_run
#            tail -f claude_jobs/logs/stage1-<jobid>.agent.log ;  cat claude_jobs/standups/stage1.md
# Move:      scontrol update JobId=<id> StartTime=2026-11-06T07:00:00
# Cancel:    scancel <ids from claude_jobs/jobs.txt>
# Test only: srun -A torch_pr_527_general -p cpu_short -c 2 --mem=4G --time=00:10:00 \
#              env STAGE=1 PROMPT=claude_jobs/prompts/stage1_loss.md WALL_MIN=15 DRY_RUN=1 bash claude_jobs/run_agent.sh
set -euo pipefail
H=/scratch/wg2381/transport_map/claude_jobs
START="${1:-now}"; INTERVAL_H="${2:-5}"

live=$(squeue -h -u "$USER" -o '%j' | grep -c '^tmap_stage' || true)
if [ "$live" -gt 0 ] && [ "${FORCE:-0}" != 1 ]; then
  echo "$live tmap_stage jobs are already queued or running; set FORCE=1 to submit another set" >&2
  squeue -u "$USER" -o '%.10i %.26j %.8T %.20S' | grep -E 'JOBID|tmap_stage' >&2
  exit 1
fi

t0=$(date -d "$START" +%s)
prev=""
echo "# submitted $(date -Is), START=$(date -d "@$t0" -Is), interval ${INTERVAL_H} h" >> "$H/jobs.txt"
for k in 1 2 3 4; do
  begin=$(date -d "@$(( t0 + (k - 1) * INTERVAL_H * 3600 ))" +%Y-%m-%dT%H:%M:%S)
  dep=(); [ -n "$prev" ] && dep=(--dependency="afterany:$prev")
  id=$(sbatch --parsable --begin="$begin" "${dep[@]}" "$H/stage$k.slurm")
  echo "$id stage$k begin=$begin ${dep[*]:-}" | tee -a "$H/jobs.txt"
  prev=$id
done
