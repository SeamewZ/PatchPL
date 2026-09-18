#!/bin/bash
# Launch 3-GPU parallel v3 eval pilot
cd /home/wcx/swe
export PATH=/home/wcx/miniconda3/bin:/usr/bin:/bin
for R in 0 1 2; do
  nohup env CUDA_VISIBLE_DEVICES=$R EVAL_RANK=$R EVAL_WORLD=3 N_EVAL=20 \
    /home/wcx/miniconda3/bin/python -u eval_agent_sft.py \
    > logs/eval_v3_r$R.log 2>&1 &
done
echo LAUNCHED-3-WORKERS
