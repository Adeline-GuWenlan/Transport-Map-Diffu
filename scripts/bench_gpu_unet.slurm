#!/bin/bash
# unet chain S1: short GPU check of the UNet backbone before the 4 runs (docs/experiments_unet.md "GPU bench"), modelled
# on scripts/bench_gpu.slurm. 60 training steps each of
#   UNet A at grad_chunk 512 (chosen; = N, one pass) and 256 (fallback),
#   UNet B at grad_chunk 256 (chosen) and 512 (one size up; may run out of memory, which is then the answer),
# recording s/step and peak memory, then one pass of the snapshot evaluation path (tmap.evaluate + tmap.sample) at
# small sizes. Output: runs/bench_gpu_unet_<jobid>/.      sbatch scripts/bench_gpu_unet.slurm
#SBATCH --account=torch_pr_527_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --job-name=tmap_bench_unet
#SBATCH --output=/scratch/wg2381/transport_map/runs/slurm-%x-%j.out

ROOT=/scratch/wg2381/transport_map
cd "$ROOT" || exit 1
for v in $(env | grep -oE '^(CLAUDECODE|CLAUDE_CODE_[A-Z_]*)='); do unset "${v%=}"; done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source "$ROOT/env/activate.sh" || exit 1
export HF_HUB_OFFLINE=1
B=$ROOT/runs/bench_gpu_unet_$SLURM_JOB_ID
mkdir -p "$B"
echo "[$(date)] job $SLURM_JOB_ID on $(hostname); out $B; commit $(git -C "$ROOT" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

SET="train.max_steps=60 train.log_every=10 train.snapshot_steps=[60] train.ckpt_every=1000000 train.sample_every=60
     eval.every=60 eval.energy_distance.n_per_class=100"
for spec in "A_c512 expA_cifar10_unet_pixel 512" "A_c256 expA_cifar10_unet_pixel 256" \
            "B_c256 expB_cifar10_unet_pixel-dinov2 256" "B_c512 expB_cifar10_unet_pixel-dinov2 512"; do
  set -- $spec
  t0=$(date +%s)
  timeout 600 python -m tmap.train --config configs/$2.yaml --out "$B/$1" --no-resume --set $SET train.grad_chunk=$3 \
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
  grep -E 'model .* params|feature term|Error|error' "$B/$1.log" | head -5
done

t0=$(date +%s)
timeout 900 python -m tmap.evaluate --ckpt "$B/B_c256/ckpt/step0000060.pt" --out "$B/eval_B60.json" --device cuda \
  --n-per-class 100 --fid-num-gen 2000 --feature-ed inception,dinov2 --nn-grid "$B/eval_B60_nn.png" > "$B/eval.log" 2>&1
echo "== evaluate (100/class, FID-2k, both weights): exit $? in $(( $(date +%s) - t0 )) s"; tail -n 14 "$B/eval.log"
for w in ema raw; do
  python -m tmap.sample --ckpt "$B/B_c256/ckpt/step0000060.pt" --out "$B/samples_B60_$w.png" --weights $w \
    --device cuda --n-per-class 10 >> "$B/eval.log" 2>&1
  echo "sample $w exit $?"
done
echo "[$(date)] done"
