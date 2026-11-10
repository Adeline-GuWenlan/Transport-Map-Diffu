#!/bin/bash
# Runs all stage-1 numerical checks (CPU only, ~20 min on 8 cores). Usage: bash run_all.sh
cd "$(dirname "$0")"
eval "$(conda shell.bash hook)" && conda activate /scratch/wg2381/.conda/envs/tmap-tex
mkdir -p logs results ../figures
for s in check_e_grad_at_zero check_b_beta2 check_a_estimator check_c_toy2d check_c2_long check_d_conditional \
         check_d2_conditional_2d check_f_zero_init; do
  echo "== $s $(date)"
  ( time timeout 1500 python "$s.py" ) > "logs/$s.log" 2>&1
  echo "   exit $? $(date)"
done
python make_tex_results.py > logs/make_tex_results.log 2>&1
