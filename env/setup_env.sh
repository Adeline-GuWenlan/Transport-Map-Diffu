#!/bin/bash
# Rebuild the project env /scratch/wg2381/.conda/envs/tmap from scratch (stage 3, 2026-11-06).
#
#   bash env/setup_env.sh          # top-level pins below; pip resolves the rest
#   bash env/setup_env.sh --lock   # exact versions from env/requirements.txt (pip freeze of the tested env)
#   TMAP_ENV=/path/to/other/prefix bash env/setup_env.sh --lock   # build a copy elsewhere (used to test this script)
#
# conda-forge only: python + pip. PyTorch from pip, CUDA 12.6 build (runs on the CPU nodes too; the
# same env serves the later GPU job). Choices: docs/stage2_choices.md and claude_jobs/standups/stage2.md.
# No `set -u`: conda's activate.d scripts read unset variables on this cluster.
set -eo pipefail

ENV=${TMAP_ENV:-/scratch/wg2381/.conda/envs/tmap}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TORCH_INDEX=https://download.pytorch.org/whl/cu126   # torch 2.14.1 has cu126 and cu130 wheels only; cu126 needs the older driver

PY_VER=3.11
TORCH_PINS="torch==2.14.1 torchvision==0.29.1"
PIP_PINS="timm==1.0.31 numpy scipy pyyaml pillow matplotlib tqdm pyarrow pytest requests"
NODEPS_PINS="clean-fid==0.1.35"   # --no-deps so it cannot replace torch

# Keep every cache and install off HOME (30k-file quota).
export PIP_CACHE_DIR=/scratch/wg2381/.cache-home/pip
export PYTHONNOUSERSITE=1 PIP_USER=0 PIP_DISABLE_PIP_VERSION_CHECK=1
export CONDA_PKGS_DIRS=/scratch/wg2381/.conda/pkgs

eval "$(conda shell.bash hook)"
if [ -d "$ENV" ]; then
  echo "removing existing $ENV"
  conda env remove -y -p "$ENV" || rm -rf "$ENV"
fi
conda create -y -p "$ENV" --override-channels -c conda-forge "python=$PY_VER" pip
PY="$ENV/bin/python"     # never bare `pip`: ~/.local/bin/pip is first on PATH in the agent jobs

if [ "${1:-}" = "--lock" ]; then
  # requirements.txt has torch==2.14.1+cu126 and the nvidia-* wheels; they live on the PyTorch index.
  "$PY" -m pip install --extra-index-url "$TORCH_INDEX" -r "$HERE/requirements.txt"
else
  "$PY" -m pip install --index-url "$TORCH_INDEX" $TORCH_PINS
  "$PY" -m pip install $PIP_PINS
  "$PY" -m pip install --no-deps $NODEPS_PINS
fi
"$PY" -m pip check

# clean-fid looks for reference stats in <site-packages>/cleanfid/stats/ and downloads them if absent.
# Seed that cache from data/fid/stats/ (downloaded by stage 3; see data/README.md) when present.
STATS_SRC="$HERE/../data/fid/stats"
if ls "$STATS_SRC"/*.npz >/dev/null 2>&1; then
  STATS_DST=$("$PY" -c "import os, cleanfid; print(os.path.join(os.path.dirname(cleanfid.__file__), 'stats'))")
  mkdir -p "$STATS_DST" && cp -v "$STATS_SRC"/*.npz "$STATS_DST"/
fi
"$PY" -c "import torch, torchvision, timm; print('torch', torch.__version__, 'cuda build', torch.version.cuda, '| torchvision', torchvision.__version__, '| timm', timm.__version__)"
echo "done: source env/activate.sh, then python scripts/check_env.py"
