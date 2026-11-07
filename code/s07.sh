#!/bin/bash
# E8 then E9 (PLAN_v7).
cd "$(dirname "$0")"; PY=${PY:-python}
for d in Texas Wisconsin Cornell Cora CiteSeer Actor PubMed minesweeper tolokers amazon-ratings roman-empire; do
  timeout 21600 $PY -W ignore exp8.py $d 2>&1 | grep --line-buffered -v Warn | cut -c1-220
done
echo S07_DONE
for d in CiteSeer Actor Cora; do
  timeout 43200 $PY -W ignore exp9.py $d 2>&1 | grep --line-buffered -v Warn | cut -c1-200
done
echo S08_DONE
