#!/bin/bash
cd /home/wcx/swe
mkdir -p repos
declare -A R=(
  [astropy]="https://github.com/astropy/astropy.git"
  [django]="https://github.com/django/django.git"
  [matplotlib]="https://github.com/matplotlib/matplotlib.git"
  [seaborn]="https://github.com/mwaskom/seaborn.git"
  [flask]="https://github.com/pallets/flask.git"
  [requests]="https://github.com/psf/requests.git"
)
for name in astropy django matplotlib seaborn flask requests; do
  if [ ! -d "repos/$name/.git" ]; then
    echo "=== cloning $name ==="
    git clone --filter=blob:none --quiet "${R[$name]}" "repos/$name" 2>&1 | tail -2
    echo "$name done: $(du -sh repos/$name | cut -f1)"
  else
    echo "$name exists"
  fi
done
echo ALL-CLONES-DONE
