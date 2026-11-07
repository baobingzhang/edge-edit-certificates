#!/bin/bash
# S02: config selection on validation only. G-NAMRFF last because it is by far the slowest.
cd "$(dirname "$0")"; PY=${PY:-python}
DS="Cora CiteSeer PubMed Cora-public CiteSeer-public PubMed-public Texas Wisconsin Cornell Actor roman-empire amazon-ratings minesweeper tolokers"
run(){ timeout $1 $PY -W ignore exp.py s02 $2 $3 2>&1 | grep --line-buffered -E " s02 |skip" | cut -c1-260; [ ${PIPESTATUS[0]} = 124 ] && echo "{\"ds\":\"$2\",\"model\":\"$3\",\"status\":\"TIMEOUT\"}" > ../results/s02/$2__$3.json && echo "$2 $3 TIMEOUT"; }
mkdir -p ../results/s02
for d in $DS; do for m in logreg egogam certgam gcn mlp sage gat h2gcn; do run 3600 $d $m; done; done
for d in $DS; do run 3600 $d certrob; done
for d in $DS; do run 5400 $d smoothgcn; done
echo S02_CORE_DONE
for d in $DS; do run 14400 $d gnamrff; done
echo S02_DONE
