#!/bin/bash
# Pull SWE-smith images for pilot instances, with proxy fallbacks.
cd /home/wcx/swe
log=logs/pull_pilot.log
: > "$log"

cut -f3 /tmp/pilot_list.txt | sort -u | while read -r img; do
  if docker image inspect "$img" >/dev/null 2>&1; then
    echo "have $img" >> "$log"
    continue
  fi
  echo "pull $img" >> "$log"
  if docker pull "$img" >> "$log" 2>&1; then
    echo "OK $img" >> "$log"
    continue
  fi
  ok=0
  for proxy in dockerproxy.net hub.rat.dev docker.1panel.live; do
    if docker pull "$proxy/$img" >> "$log" 2>&1; then
      docker tag "$proxy/$img" "$img"
      echo "OK-via-$proxy $img" >> "$log"
      ok=1
      break
    fi
  done
  [ "$ok" = 0 ] && echo "FAIL $img" >> "$log"
done
echo "ALL-PULL-DONE" >> "$log"
