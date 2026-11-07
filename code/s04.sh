#!/bin/bash
# Supplementary post-review runs. Ablation and the public-split G-NAMRFF queue may still be running: at most one S04 job at a time.
cd "$(dirname "$0")"; PY=${PY:-python}; ALL="Texas Wisconsin Cornell Cora CiteSeer Actor minesweeper tolokers amazon-ratings roman-empire PubMed"
r(){ timeout $1 $PY -W ignore exp4.py $2 $3 2>&1 | grep --line-buffered -E "ok|OOM|error|skip|Traceback" | cut -c1-200; rc=${PIPESTATUS[0]}; [ $rc != 0 ] && echo "$3 $2 rc=$rc"; }
for d in Cora CiteSeer PubMed Cornell; do r 18000 gnam_t3 $d; done
for d in $ALL; do r 5400 certmlp $d; r 7200 certmlprob $d; done
for d in $ALL; do r 14400 smoothsage $d; done
# The paper also reran the G-NAMRFF authors' own script (train_node.py, lr 0.018, seeds 0-4); it is not part of this
# repository. Its five "Best Test Accuracy" lines are kept in logs/s04.log, which make_paper.py reads.
echo S04_DONE
