#!/bin/bash
# Hourly guard for 3-GPU DDP patch SFT (train_patch_ddp.py)
cd /home/wcx/swe
export PATH=/home/wcx/miniconda3/bin:/usr/bin:/bin

GUARD_LOG=logs/base_patch_guard.log
STATE=.patch_guard_state
CHECK_SECONDS=3600
MAX_RESTARTS=10

ts() { date '+%F %T'; }
log() { echo "$(ts) $*" >> "$GUARD_LOG"; }

restarts=0; last_steps=""; stall_ts=""
[ -f "$STATE" ] && . "$STATE" 2>/dev/null

alive() { pgrep -f 'train_patch_dd[p]' > /dev/null 2>&1; }

cur_steps() {
  grep -a -o -E 'step [0-9]+/189' logs/base_patch_ddp.log 2>/dev/null | tail -1 | grep -o -E '[0-9]+' | head -1
}

relaunch() {
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    log "RESTART-LIMIT-REACHED manual fix needed"
    exit 1
  fi
  restarts=$((restarts+1))
  log "relaunching DDP training (count=$restarts)"
  nohup env PATH=/home/wcx/miniconda3/bin:/usr/bin:/bin \
    CUDA_VISIBLE_DEVICES=0,1,2 \
    /home/wcx/miniconda3/bin/python -u -m torch.distributed.run \
    --nproc_per_node=3 --master_port=29511 train_patch_ddp.py \
    >> logs/base_patch_ddp.log 2>&1 &
  last_steps=""; stall_ts=""
  save_state
}

save_state() {
  printf 'restarts=%s\nlast_steps=%s\nstall_ts=%s\n' \
    "$restarts" "$last_steps" "$stall_ts" > "$STATE"
}

while true; do
  if grep -aq 'QWEN35-4B-LORA-PATCH-DONE' logs/base_patch_ddp.log 2>/dev/null; then
    log "TRAINING-DONE guard exits"
    break
  fi
  s=$(cur_steps)
  if ! alive; then
    log "trainer dead (last step=$s) -> relaunch"
    relaunch
  else
    if [ -n "$s" ]; then
      if [ "$s" != "$last_steps" ]; then
        last_steps="$s"; stall_ts=""; save_state
        log "progress ok step=$s"
      elif [ -n "$stall_ts" ]; then
        stalled=$(( $(date +%s) - $(date -d "$stall_ts" +%s) ))
        if [ "$stalled" -gt $((CHECK_SECONDS * 2)) ]; then
          log "STALLED 2h at step=$s -> restart"
          pkill -f 'train_patch_dd[p]'
          sleep 10
          relaunch
        fi
      else
        stall_ts=$(date '+%F %T'); save_state
        log "no progress since step=$s (watching for stall)"
      fi
    fi
  fi
  sleep $CHECK_SECONDS
done
