#!/bin/bash
# Level 2: correctness gates. Each script exits non-zero if a certificate is not exact (a real edit flips a
# "certified" node, or brute force finds a tighter bound) or a theorem is violated on a sampled instance.
# CPU checks always run; GPU checks run when CUDA is available.
source "$(dirname "$0")/_work.sh"
cd "$WORK/code"
CPU="verify_theory.py verify_shrink.py test_cert.py test_smooth.py test_fallback.py"
GPU="test_cert2.py test_shrink.py test_ppr_exact.py test_ppr_chunk.py test_ppr_fix.py"
cuda=$($PY -c "import torch; print(int(torch.cuda.is_available()))")
pass=0; fail=0; skip=0
for t in $CPU $GPU; do
  if [ $cuda = 0 ] && [[ " $GPU " == *" $t "* ]]; then echo "SKIP (no CUDA) $t"; skip=$((skip+1)); continue; fi
  t0=$(date +%s)
  if $PY -W ignore "$t" > "$WORK/$t.log" 2>&1; then echo "PASS $t ($(( $(date +%s)-t0 ))s)"; pass=$((pass+1));
  else echo "FAIL $t (log: $WORK/$t.log)"; tail -5 "$WORK/$t.log"; fail=$((fail+1)); fi
done
echo "== $pass passed, $fail failed, $skip skipped"
[ $fail = 0 ]
