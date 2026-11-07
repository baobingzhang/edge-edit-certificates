#!/bin/bash
# E6 (PLAN_v5): two-layer SAGE exact b=1 certificates. One GPU job at a time; finished datasets are skipped.
cd "$(dirname "$0")"; PY=${PY:-python}
for d in Texas Wisconsin Cornell Cora CiteSeer Actor minesweeper tolokers PubMed roman-empire amazon-ratings; do
  timeout 86400 $PY -W ignore exp6.py $d 2>&1 | grep --line-buffered -v Warn | cut -c1-200
done
echo S05_DONE
