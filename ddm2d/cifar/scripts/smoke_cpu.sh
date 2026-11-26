#!/bin/bash
# CPU smoke (proves the path, not a result): every run config with a tiny U-Net, 20 steps on 512 train images, then
# NFE-2 samples of 300 images, a 300-image clean-fid FID call, a grid and the xi-spread. Outputs: cifar/runs/smoke/<run>/.
# Run from ddm2d/:  bash cifar/scripts/smoke_cpu.sh
export PYTHONPATH=src:cifar/src OMP_NUM_THREADS=${OMP_NUM_THREADS:-7} HF_HOME=/scratch/wg2381/.huggingface HF_HUB_OFFLINE=1 \
       XDG_CACHE_HOME=/scratch/wg2381/.cache-home TORCH_HOME=/scratch/wg2381/.cache-home/torch
PY=/scratch/wg2381/.conda/envs/tmap/bin/python
TINY='{"base_channels":16,"channel_mult":[1,2],"num_res_blocks":1,"attention_resolutions":[16],"num_heads":2,"emb_dim":32}'
for r in cifar_A cifar_B cifar_B_b1 cifar_B_pH cifar_C cifar_C_dino; do
  out=cifar/runs/smoke/$r
  B=8; [ $r = cifar_C_dino ] && B=4
  $PY -m ddmcifar.train --config cifar/configs/$r.json --out $out --set "model=$TINY" device='"cpu"' steps=20 data_n=512 \
      batch=$B log_every=5 ckpt_every=10 'snapshots=[10,20]' grad_chunk=12 > $out.train.log 2>&1 || { echo "$r train FAILED"; tail $out.train.log; continue; }
  $PY -m ddmcifar.evaluate --run $out --ckpt final --nfe 2 --grid-nfe 2 --n 300 --device cpu --batch 100 --spread \
      > $out.eval.log 2>&1 || { echo "$r eval FAILED"; tail $out.eval.log; continue; }
  echo "$r ok: $(tail -n 1 $out/log.jsonl | cut -c1-160)"; tail -n 1 $out.eval.log
done
