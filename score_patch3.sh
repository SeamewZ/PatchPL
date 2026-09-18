#!/bin/bash
# Score non-empty model patches with swesmith harness
cd /home/wcx/swe
export PATH=/home/wcx/miniconda3/bin:/usr/bin:/bin
GTOK=$(/home/wcx/bin/gh auth token 2>/dev/null)
nohup env GITHUB_TOKEN="$GTOK" \
  /home/wcx/miniconda3/bin/python -u -m swesmith.harness.eval \
  --dataset_path data/swe_smith_test250.jsonl \
  --predictions_path predictions/patch3_nonempty.json \
  --run_id patch3-score --workers 4 \
  > logs/score_patch3.log 2>&1 &
echo SCORE-STARTED
