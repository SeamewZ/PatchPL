#!/bin/bash
cd /home/wcx/swe
mkdir -p data/swe_verify/rollouts
nohup env N_SAMPLES=3 MAX_TURNS=25 /home/wcx/miniconda3/bin/python -u gen_ds_rollout_v2.py 0 5 \
  > logs/rollout_0_5.log 2>&1 &
nohup env N_SAMPLES=3 MAX_TURNS=25 /home/wcx/miniconda3/bin/python -u gen_ds_rollout_v2.py 5 10 \
  > logs/rollout_5_10.log 2>&1 &
echo ROLLOUTS-STARTED
