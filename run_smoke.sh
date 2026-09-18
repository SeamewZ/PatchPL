#!/bin/bash
# Run swesmith gold smoke with GitHub token from gh CLI
cd /home/wcx/swe
export PATH=/home/wcx/miniconda3/bin:/usr/bin:/bin
GTOK=$(/home/wcx/bin/gh auth token 2>/dev/null)
nohup env GITHUB_TOKEN="$GTOK" \
  /home/wcx/miniconda3/bin/python -u -m swesmith.harness.eval \
  --dataset_path data/swe_smith_test250.jsonl \
  --predictions_path predictions/gold_smoke.json \
  --run_id gold-smith-smoke2 --workers 4 \
  > logs/smith_smoke2.log 2>&1 &
echo SMOKE2-STARTED
