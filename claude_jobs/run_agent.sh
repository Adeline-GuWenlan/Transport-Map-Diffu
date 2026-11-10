#!/bin/bash
# Shared runner for the transport_map Claude stages; claude_jobs/stage<N>.slurm execs this.
# A slimmed copy of the mechanics in /scratch/wg2381/claude_jobs/*/run_agent.sh.
#
# Env in:  STAGE (1-4)  PROMPT (prompt file)  WALL_MIN (the job's --time in minutes)
#          [EFFORT=xhigh] [MODEL=claude-opus-5-5] [DRY_RUN=1: tiny prompt, throwaway session, no standup]
#
# One stage = one Claude conversation. Its session id is pinned in state/, so a retry in the same job
# resumes the conversation instead of starting from nothing. Every exit path leaves a standup: the
# agent writes one; if it doesn't, this script writes an INCOMPLETE stub so the next stage knows.
set -uo pipefail

ROOT=/scratch/wg2381/transport_map
J=$ROOT/claude_jobs
LOGS=$J/logs; STAND=$J/standups; STATE=$J/state
mkdir -p "$LOGS" "$STAND" "$STATE"

# sbatch copies the submitting shell's environment. When that shell was itself a Claude Code
# session, its session/socket variables would leak in and make this agent think it is nested.
for v in $(compgen -e | grep -E '^(CLAUDECODE$|CLAUDE_CODE_|CLAUDE_PID$|CLAUDE_EFFORT$)'); do unset "$v"; done
# Credentials and transcripts live on /scratch (the home disk filled up on 10-17).
export CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-/scratch/wg2381/claude-config}"
export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

T0=$(date +%s)
END=$(( T0 + ${WALL_MIN:-290} * 60 ))   # Slurm kills the job here
STOP=$(( END - 300 ))                     # claude is stopped here, so this script can still clean up
DUE=$(( END - 1800 ))                     # the agent is told to have its standup written by here
[ "$DUE" -gt $(( T0 + 600 )) ] || DUE=$(( STOP - 120 ))   # short test jobs: keep the due time after the start
STANDUP=$STAND/stage${STAGE}.md
fmt() { date -d "@$1" '+%Y-%m-%d %H:%M %Z'; }

if [ "${DRY_RUN:-0}" = 1 ]; then
  AGENT_LOG=$LOGS/dryrun-stage${STAGE}-${SLURM_JOB_ID:-local}.agent.log
  SID=$(cat /proc/sys/kernel/random/uuid)
else
  AGENT_LOG=$LOGS/stage${STAGE}-${SLURM_JOB_ID:-local}.agent.log
  SID_FILE=$STATE/stage${STAGE}.session
  if [ -s "$SID_FILE" ]; then SID=$(tr -d '[:space:]' < "$SID_FILE")
  else SID=$(cat /proc/sys/kernel/random/uuid); printf '%s\n' "$SID" > "$SID_FILE"; fi
fi
# Claude stores transcripts under projects/<cwd with every non-alphanumeric char turned into '-'>.
TRANSCRIPT="$CLAUDE_CONFIG_DIR/projects/$(printf '%s' "$ROOT" | sed 's/[^A-Za-z0-9]/-/g')/$SID.jsonl"

echo "=== stage ${STAGE} start $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-none} ==="
echo "session ${SID}  (inspect later: cd ${ROOT} && claude --resume ${SID})"
echo "agent log ${AGENT_LOG}"
echo "stop $(fmt $STOP), standup due $(fmt $DUE)"

HEADER="## Pipeline context (written by claude_jobs/run_agent.sh)
- You are stage ${STAGE} of 4 of an unattended pipeline. No human is watching; don't ask questions, decide and record why.
- Slurm job ${SLURM_JOB_ID:-none} on $(hostname): ${SLURM_CPUS_PER_TASK:-?} CPUs, $(( ${SLURM_MEM_PER_NODE:-0} / 1024 )) GB RAM, no GPU.
- Started $(fmt $T0). You are stopped at $(fmt $STOP). Write your standup (claude_jobs/standups/stage${STAGE}.md) by $(fmt $DUE). Check \`date\` as you go.
- Project root ${ROOT}; its CLAUDE.md is binding.
"

# Prints the epoch second to resume at (reset + 3 min) from a "resets 3:40am" style usage-limit message.
reset_epoch() {
  local s t now
  s=$(tail -n 60 "$AGENT_LOG" 2>/dev/null | grep -oiE 'resets [0-9]{1,2}(:[0-9]{2})?(am|pm)' | tail -1 | awk '{print $2}')
  [ -n "$s" ] || return 1
  t=$(TZ=America/New_York date -d "$s" +%s 2>/dev/null) || return 1
  now=$(date +%s)
  if [ "$t" -le "$now" ]; then
    if [ $(( now - t )) -lt 7200 ]; then t=$now; else t=$(( t + 86400 )); fi
  fi
  echo $(( t + 180 ))
}

ATTEMPT=0; OTHER_FAILS=0; RC=1; REASON=""
while :; do
  ATTEMPT=$((ATTEMPT + 1))
  LEFT=$(( STOP - $(date +%s) ))
  [ "$LEFT" -gt 300 ] || { REASON="out of time before attempt ${ATTEMPT}"; break; }
  if [ "${DRY_RUN:-0}" = 1 ]; then
    ARGS=(--session-id "$SID"); EFF=low
    TEXT="${HEADER}
DRY RUN of the pipeline machinery. Create or edit nothing. Run \`date\`, \`nproc\`, \`pwd\` and \`ls claude_jobs/standups\`, then reply in three lines: the standup due time from the header above, the CPU count, and the first line of CLAUDE.md."
  elif [ -s "$TRANSCRIPT" ]; then
    ARGS=(--resume "$SID"); EFF="${EFFORT:-xhigh}"
    TEXT="${HEADER}
You were interrupted (usage limit, API error, or an earlier job of this stage). Continue stage ${STAGE} where you stopped: re-check on disk what is already done, then finish the remaining work and the standup."
  else
    ARGS=(--session-id "$SID"); EFF="${EFFORT:-xhigh}"
    TEXT="${HEADER}
$(cat "$PROMPT")"
  fi
  echo "--- attempt ${ATTEMPT} $(date -Is) ${ARGS[0]} ${SID} effort=${EFF} timeout=${LEFT}s ---" | tee -a "$AGENT_LOG"
  cd "$ROOT" || exit 2
  timeout --signal=INT --kill-after=120 "$LEFT" \
    claude -p "$TEXT" "${ARGS[@]}" --name "transport_map stage ${STAGE}" \
      --model "${MODEL:-claude-opus-5-5}" --effort "$EFF" --dangerously-skip-permissions \
      --add-dir "$ROOT" --add-dir /scratch/wg2381/.conda --add-dir /scratch/wg2381/.cache-home \
      2>&1 | tee -a "$AGENT_LOG"
  RC=${PIPESTATUS[0]}
  [ "$RC" -eq 0 ] && break
  [ "$RC" -eq 124 ] && { REASON="hit the stop time (timeout)"; break; }
  [ "${DRY_RUN:-0}" = 1 ] && { REASON="dry run failed rc=${RC}"; break; }
  if tail -n 200 "$AGENT_LOG" | grep -qiE 'session limit|usage limit|hit your .*limit'; then
    R=$(reset_epoch) || R=$(( $(date +%s) + 1200 ))
    if [ "$R" -lt $(( DUE - 1200 )) ]; then
      echo "usage limit; resets $(fmt $R); sleeping until then and resuming" | tee -a "$AGENT_LOG"
      sleep $(( R > $(date +%s) ? R - $(date +%s) : 0 )); continue
    fi
    REASON="usage limit until $(fmt $R), past this stage's deadline"; break
  fi
  OTHER_FAILS=$((OTHER_FAILS + 1))
  if [ "$OTHER_FAILS" -le 2 ]; then
    echo "rc=${RC} (not a usage limit); retrying in 120 s" | tee -a "$AGENT_LOG"; sleep 120; continue
  fi
  REASON="rc=${RC} three times"; break
done

if [ "${DRY_RUN:-0}" != 1 ] && [ ! -s "$STANDUP" ]; then
  cat > "$STANDUP" <<EOF
Status: INCOMPLETE

# Stage ${STAGE} standup (stub written by run_agent.sh; the agent wrote none)
- Slurm job ${SLURM_JOB_ID:-none}, ended $(date -Is), claude rc=${RC}. Reason: ${REASON:-agent exited without a standup}.
- Agent log: ${AGENT_LOG}
- Conversation: cd ${ROOT} && claude --resume ${SID}
- Next stage: check on disk what this stage produced, and finish what you depend on.
EOF
  echo "wrote stub standup ${STANDUP}"
fi
echo "=== stage ${STAGE} exit rc=${RC} ${REASON} $(date -Is) ==="
head -n 1 "$STANDUP" 2>/dev/null
exit 0
