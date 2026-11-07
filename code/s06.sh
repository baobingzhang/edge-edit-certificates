#!/bin/bash
# E7 (PLAN_v6, exploratory): exact adversarial training of two-layer SAGE on the six non-trivial graphs with <= 8000 nodes.
cd "$(dirname "$0")"; PY=${PY:-python}
for d in Texas Wisconsin Cornell Cora CiteSeer Actor; do
  timeout 43200 $PY -W ignore exp7.py $d 2>&1 | grep --line-buffered -v Warn | cut -c1-200
done
echo S06_DONE
