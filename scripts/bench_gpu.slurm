#!/bin/bash
# Short GPU check before Experiments A / B (docs/experiments_AB.md): environment on the GPU, then 60 training steps
# of A (grad_chunk 128 and 256) and of B (128), recording s/step and peak memory, then the snapshot evaluation path
# (small sizes). The numbers set the --time and chunk size of the real runs.   sbatch scripts/bench_gpu.slurm
#SBATCH --account=torch_pr_527_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --job-name=tmap_bench
#SBATCH --output=/scratch/wg2381/transport_map/runs/slurm-%x-%j.out

ROOT=/scratch/wg2381/transport_map
cd "$ROOT" || exit 1
for v in $(env | grep -oE '^(CLAUDECODE|CLAUDE_CODE_[A-Z_]*)='); do unset "${v%=}"; done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$ROOT/env/activate.sh" || exit 1
export HF_HUB_OFFLINE=1
B=$ROOT/runs/bench_gpu_$SLURM_JOB_ID
mkdir -p "$B"
echo "[$(date)] job $SLURM_JOB_ID on $(hostname); out $B"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

timeout 600 python scripts/check_env.py --config configs/expA_cifar10_dit-b2_pixel.yaml --device cuda --reps 3 \
  --n-per-class 16 > "$B/check_env.log" 2>&1; echo "check_env exit $?"; tail -n 3 "$B/check_env.log"

SET="train.max_steps=60 train.log_every=10 train.snapshot_steps=[60] train.ckpt_every=1000000 train.sample_every=60
     eval.every=60 eval.energy_distance.n_per_class=100"
for spec in "A_c128 expA_cifar10_dit-b2_pixel 128" "A_c256 expA_cifar10_dit-b2_pixel 256" \
            "B_c128 expB_cifar10_dit-b2_pixel-dinov2 128"; do
  set -- $spec
  t0=$(date +%s)
  timeout 900 python -m tmap.train --config configs/$2.yaml --out "$B/$1" --no-resume --set $SET train.grad_chunk=$3 \
    > "$B/$1.log" 2>&1
  echo "== $1: exit $? in $(( $(date +%s) - t0 )) s"
  python - "$B/$1/metrics.jsonl" <<'EOF'
import json, sys
try:
    r = [json.loads(l) for l in open(sys.argv[1])]
except FileNotFoundError:
    sys.exit(print("   no metrics"))
late = r[2:] or r
sps = sorted(x["steps_per_s"] for x in late)[len(late) // 2]
print(f"   median {1 / sps:.3f} s/step over steps {late[0]['step']}-{late[-1]['step']}; peak mem "
      f"{max(x.get('max_mem_gib', 0) for x in r):.1f} GiB; loss {r[0]['loss']:.3f} -> {r[-1]['loss']:.3f}")
EOF
  grep -E 'feature term|Error|error' "$B/$1.log" | head -5
done

t0=$(date +%s)
timeout 900 python -m tmap.evaluate --ckpt "$B/B_c128/ckpt/step0000060.pt" --out "$B/eval_B60.json" --device cuda \
  --n-per-class 100 --fid-num-gen 2000 --feature-ed inception,dinov2 --nn-grid "$B/eval_B60_nn.png" > "$B/eval.log" 2>&1
echo "== evaluate (100/class, FID-2k, both weights): exit $? in $(( $(date +%s) - t0 )) s"; tail -n 14 "$B/eval.log"
python -m tmap.sample --ckpt "$B/B_c128/ckpt/step0000060.pt" --out "$B/samples_B60.png" --device cuda > /dev/null 2>&1
echo "sample exit $?"

# Does DINOv2 see the structure pixels miss? Same sets as stage 4's Inception check (block-shuffled right-class images
# vs wrong-class real images), 200/class, plus the stage-4 CIFAR CPU model. Informs the reading of Experiment B.
timeout 600 python scripts/feature_space_check.py --n 200 --spaces inception,dinov2 --device cuda \
  --ckpt runs/2026-11-06_smoke_cifar10_dit-t4/ckpt/latest.pt --out "$B/feature_space_check.json" 2>&1 | grep -v -i warn
echo "[$(date)] done"
