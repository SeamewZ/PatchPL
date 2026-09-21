#!/bin/bash
cd /home/wcx/swe
nohup /home/wcx/miniconda3/bin/python -u label_rollouts.py > logs/label_all.log 2>&1 &
echo LABEL-ALL-STARTED
