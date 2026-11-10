# Activate the transport_map env. Usage (any cwd, bash):  source /scratch/wg2381/transport_map/env/activate.sh
# Do not run under `set -u`: conda's activate.d scripts on this cluster read unset variables.
# Rebuild the env with env/setup_env.sh.

TMAP_ENV=/scratch/wg2381/.conda/envs/tmap
export TMAP_ROOT=/scratch/wg2381/transport_map

if [ ! -x "$TMAP_ENV/bin/python" ]; then
  echo "env/activate.sh: $TMAP_ENV missing; run bash $TMAP_ROOT/env/setup_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi
eval "$(conda shell.bash hook)"
conda activate "$TMAP_ENV" || { return 1 2>/dev/null || exit 1; }

# Caches on /scratch, never HOME (30k-file quota). HF_HOME keeps the user's existing location.
export XDG_CACHE_HOME=/scratch/wg2381/.cache-home
export HF_HOME=/scratch/wg2381/.huggingface
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME=/scratch/wg2381/.cache-home/torch
export PIP_CACHE_DIR=/scratch/wg2381/.cache-home/pip
export MPLCONFIGDIR=/scratch/wg2381/.cache-home/matplotlib
export MPLBACKEND=Agg
export PIP_USER=0 PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# Importing third_party/DiT/models.py would otherwise write third_party/DiT/__pycache__ (keep the clone clean).
export PYTHONDONTWRITEBYTECODE=1
# Project code (stage 4: src/tmap). DiT is NOT put on PYTHONPATH (its module is called `models`);
# code inserts $TMAP_DIT_DIR into sys.path explicitly.
export PYTHONPATH="$TMAP_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TMAP_DIT_DIR="$TMAP_ROOT/third_party/DiT"
# Threads: one per allocated CPU unless the caller already chose.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
# `pip` on PATH can resolve to ~/.local/bin/pip in agent jobs; always use `python -m pip`.
alias pip='python -m pip'
