#!/bin/bash
# The CPU runs of stage 4 (2026-11-06, cpu_short node, 8 CPUs), as run. Each step is a function so it can be
# re-run alone:   bash scripts/stage4_cpu_runs.sh <step>   (steps: smoke overfit eval_smoke eval_overfit cifar eval_cifar blindspot plots)
# Threads: never more than 8 in total in an 8-CPU allocation (oversubscription cut throughput ~6x); a long run
# uses 7 and leaves 1 for evaluation in parallel.
cd /scratch/wg2381/transport_map || exit 1
source env/activate.sh || exit 1
SMOKE=runs/2026-11-06_smoke_fmnist_dit-t4
OVERFIT=runs/2026-11-06_overfit_fmnist_sneaker256
CIFAR=runs/2026-11-06_smoke_cifar10_dit-t4

smoke() {   # 68 min of training, Fashion-MNIST, DiT-T/4
  OMP_NUM_THREADS=7 MKL_NUM_THREADS=7 python -m tmap.train --config configs/smoke_cpu.yaml --out $SMOKE \
    --set train.time_limit_min=68 num_threads=7 > $SMOKE/train.log 2>&1
}
overfit() { # one class (Sneaker), 256 fixed images, 12 min
  mkdir -p $OVERFIT
  OMP_NUM_THREADS=7 MKL_NUM_THREADS=7 python -m tmap.train --config configs/overfit_cpu.yaml --out $OVERFIT \
    --set num_threads=7 > $OVERFIT/train.log 2>&1
}
eval_smoke() {
  OMP_NUM_THREADS=${T:-1} MKL_NUM_THREADS=${T:-1} python -m tmap.evaluate --ckpt $SMOKE/ckpt/latest.pt --out $SMOKE/eval_final.json \
    --n-per-class 1000 --nn-grid $SMOKE/nearest_neighbours.png 2>&1 | tee $SMOKE/eval_final.log
  OMP_NUM_THREADS=${T:-1} MKL_NUM_THREADS=${T:-1} python -m tmap.sample --ckpt $SMOKE/ckpt/latest.pt --out $SMOKE/samples_final_ema_10x10.png \
    --weights ema --n-per-class 10
}
eval_overfit() {
  OMP_NUM_THREADS=${T:-1} MKL_NUM_THREADS=${T:-1} python -m tmap.evaluate --ckpt $OVERFIT/ckpt/latest.pt --out $OVERFIT/eval_final.json \
    --nn-grid $OVERFIT/nearest_neighbours.png 2>&1 | tee $OVERFIT/eval_final.log
}
cifar() {   # CIFAR-10 on the same small model, 30 min: an early look at the real dataset, not the real run
  mkdir -p $CIFAR
  OMP_NUM_THREADS=7 MKL_NUM_THREADS=7 python -m tmap.train --config configs/smoke_cpu_cifar10.yaml --out $CIFAR \
    --set train.time_limit_min=30 num_threads=7 > $CIFAR/train.log 2>&1
}
eval_cifar() {  # FID on CPU from 2000 samples: indicative only (biased upward vs FID-50k)
  OMP_NUM_THREADS=${T:-8} MKL_NUM_THREADS=${T:-8} python -m tmap.evaluate --ckpt $CIFAR/ckpt/latest.pt --out $CIFAR/eval_final.json \
    --n-per-class 1000 --fid-num-gen 2000 --nn-grid $CIFAR/nearest_neighbours.png 2>&1 | tee $CIFAR/eval_final.log
}
blindspot() {  # D1 of real images with 1/2/4/8/16-px blocks drawn from independent same-class images (no model)
  for D in cifar10 fashion_mnist; do
    OMP_NUM_THREADS=${T:-1} MKL_NUM_THREADS=${T:-1} python scripts/patch_shuffle_check.py --dataset $D \
      --out runs/2026-11-06_loss_blindspot/$D.json
  done
  OMP_NUM_THREADS=${T:-8} MKL_NUM_THREADS=${T:-8} python scripts/feature_space_check.py --n 100 \
    --ckpt $CIFAR/ckpt/latest.pt --out runs/2026-11-06_loss_blindspot/cifar10_feature_space.json
}
plots() {
  for R in $SMOKE $OVERFIT $CIFAR; do [ -d $R ] && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/plot_run.py $R; done
}

for s in "$@"; do echo "== $s $(date)"; $s; done
