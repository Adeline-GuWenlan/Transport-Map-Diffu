#!/bin/bash
#SBATCH --account=torch_pr_527_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:50:00
#SBATCH --job-name=tmap_stage2_investigate
#SBATCH --output=/scratch/wg2381/transport_map/claude_jobs/logs/%x-%j.out
#SBATCH --error=/scratch/wg2381/transport_map/claude_jobs/logs/%x-%j.err
# Stage 2 of 4. Wall 4h50m so a stage that starts a few minutes late still ends before the next
# 5-hour slot. The prompt and run_agent.sh are read when the job starts, so edits made before then apply.
H=/scratch/wg2381/transport_map/claude_jobs
export STAGE=2 PROMPT=$H/prompts/stage2_investigate.md WALL_MIN=290
exec bash $H/run_agent.sh
