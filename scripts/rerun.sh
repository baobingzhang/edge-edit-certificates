#!/bin/bash
# Level 3 (GPU): rerun one experiment with its frozen configuration and compare with the shipped result.
# usage: scripts/rerun.sh RESULT_FILE COMMAND...
#   scripts/rerun.sh s03/Cora__certgam.json exp.py s03 Cora certgam     # CertGAM, splits 3-9
#   scripts/rerun.sh s03/Cora__certrob.json exp.py s03 Cora certrob     # CertGAM-R
#   scripts/rerun.sh s07/Texas__shrink.json exp8.py Texas               # ShrinkGAM-R (kappa selection + test)
#   scripts/rerun.sh s05/Texas__sage2.json  exp6.py Texas               # two-layer SAGE, exact b=1
#   scripts/rerun.sh s09/Texas__pippnp_exact.json exp10e.py Texas       # pi-PPNP, exact single toggle
[ $# -ge 2 ] || { sed -n 2,9p "$0"; exit 2; }
RES="$1"; shift
source "$(dirname "$0")/_work.sh"
[ -f "$REPO/results/$RES" ] || { echo "no shipped result results/$RES"; exit 2; }
rm -f "$WORK/results/$RES"
cd "$WORK/code"
$PY -W ignore "$@" 2>&1 | grep -v Warn | cut -c1-200
[ -f "$WORK/results/$RES" ] || { echo "the command did not write results/$RES"; exit 1; }
$PY "$REPO/scripts/compare_json.py" --metrics "$REPO/results" "$WORK/results" "$RES"
