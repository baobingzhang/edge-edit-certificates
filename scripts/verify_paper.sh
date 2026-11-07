#!/bin/bash
# Level 1 (CPU, minutes): recompute every verdict and every number, table and figure of the paper from the
# shipped per-split results, and compare the generated LaTeX with expected/gen/ (the files the paper uses).
# The first run downloads the datasets (about 270 MB) through PyTorch Geometric; make_paper.py needs them
# for the dataset table.
source "$(dirname "$0")/_work.sh"
cd "$WORK/code"
echo "== 1/3 verdicts from per-split results (judge*.py, analysis.py, sensitivity.py)"
for s in judge.py analysis.py sensitivity.py judge5.py judge7.py judge8.py; do
  $PY -W ignore $s > /dev/null || { echo "FAIL: $s"; exit 1; }
done
$PY "$REPO/scripts/compare_json.py" "$REPO/results" "$WORK/results" \
  verdict.json analysis.json sensitivity.json verdict5.json verdict7.json verdict8.json
echo "== 2/3 tables, figures and numbers (make_paper.py, make_figs.py; make_figs.py asserts its key numbers)"
$PY -W ignore make_paper.py > "$WORK/make_paper.log" 2>&1 || { tail -20 "$WORK/make_paper.log"; exit 1; }
$PY -W ignore make_figs.py > "$WORK/make_figs.log" 2>&1 || { tail -20 "$WORK/make_figs.log"; exit 1; }
echo "== 3/3 generated LaTeX against expected/gen/"
fail=0
for f in "$REPO"/expected/gen/*.tex; do
  n=$(basename "$f")
  if ! diff -q "$f" "$WORK/paper/gen/$n" > /dev/null 2>&1; then echo "DIFFERS: $n"; fail=1; fi
done
[ $fail = 0 ] && echo "PASS: $(ls "$REPO"/expected/gen/*.tex | wc -l) generated files identical; figures in $WORK/paper/figs/" || { echo "FAIL (see diff -r expected/gen $WORK/paper/gen)"; exit 1; }
