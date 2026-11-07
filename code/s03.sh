#!/bin/bash
# S03: frozen configs on splits 3-9 (see PLAN_v2 and addendum). Run only after results/s02/FROZEN.sha256 exists.
cd "$(dirname "$0")"; PY=${PY:-python}; [ -f ../results/s02/FROZEN.sha256 ] || { echo "S02 not frozen"; exit 1; }
DS="Cora CiteSeer PubMed Cora-public CiteSeer-public PubMed-public Texas Wisconsin Cornell Actor roman-empire amazon-ratings minesweeper tolokers"
mkdir -p ../results/s03
run(){ timeout $1 $PY -W ignore exp.py s03 $2 $3 2>&1 | grep --line-buffered -E " s03 |skip" | cut -c1-220; rc=${PIPESTATUS[0]}; [ $rc = 124 ] && echo "{\"ds\":\"$2\",\"model\":\"$3\",\"status\":\"TIMEOUT\"}" > ../results/s03/$2__$3.json && echo "$2 $3 TIMEOUT"; [ $rc != 0 ] && [ $rc != 124 ] && echo "$2 $3 CRASH rc=$rc"; }
for d in $DS; do for m in logreg egogam certgam gcn mlp sage gat h2gcn; do run 3600 $d $m; done; done
for d in $DS; do run 7200 $d certrob; done
$PY -W ignore synth.py 2>&1 | grep -vE 'Warning|warn\(' | tail -1 | cut -c1-200
for d in $DS; do run 7200 $d smoothgcn; done
echo S03_CORE_DONE
for d in $DS; do run 28800 $d gnamrff; done
echo S03_DONE
