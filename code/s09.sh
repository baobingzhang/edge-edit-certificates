#!/bin/bash
# E10e (PLAN_v8b, exact b=1, n<=8000) then E10 (PLAN_v8, local-budget certificate + attack upper bound).
cd "$(dirname "$0")"; PY=${PY:-python}
for d in Texas Cornell Wisconsin Cora CiteSeer Actor; do
  $PY -W ignore exp10e.py $d 2>&1 | grep --line-buffered -v Warn
done
echo S09E_DONE
for d in Texas Cornell Wisconsin Cora CiteSeer Actor PubMed amazon-ratings roman-empire minesweeper tolokers; do
  $PY -W ignore exp10.py $d 2>&1 | grep --line-buffered -v Warn
done
echo S09_DONE
