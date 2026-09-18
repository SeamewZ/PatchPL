#!/bin/bash
# Launch 3-GPU parallel full 250-instance v4 eval
cd /home/wcx/swe
export PATH=/home/wcx/miniconda3/bin:/usr/bin:/bin
pkill -f 'eval_agent_sf[t]' 2>/dev/null
sleep 2
for R in 0 1 2; do
  nohup env CUDA_VISIBLE_DEVICES=$R EVAL_RANK=$R EVAL_WORLD=3 N_EVAL=250 \
    /home/wcx/miniconda3/bin/python -u eval_agent_sft.py \
    > logs/eval_full_r$R.log 2>&1 &
done
echo FULL-250-STARTED
