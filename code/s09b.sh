#!/bin/bash
# PLAN_v8c: rerun E10 on the two datasets that failed, after the ppr.py fixes (G10, G10b, G10c rerun first).
cd "$(dirname "$0")"; PY=${PY:-python}
for d in roman-empire amazon-ratings; do
  $PY -W ignore exp10.py $d 2>&1 | grep --line-buffered -v Warn
done
echo S09B_DONE
