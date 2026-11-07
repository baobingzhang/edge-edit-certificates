"""Compare result files of two result trees.

compare_json.py SHIPPED_DIR NEW_DIR FILE...            same content (numbers equal to 1e-9; key order ignored)
compare_json.py --metrics SHIPPED_DIR NEW_DIR FILE     per-model means over splits of accuracy and certified
                                                       accuracy at b=1; tolerance --tol (default 0.02, i.e. two
                                                       points, for GPU non-determinism and timing-dependent paths)
Exits 1 on a difference.
"""

import json
import math
import os
import sys


def same(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return f"{path}: keys differ {sorted(set(a) ^ set(b))[:5]}"
        for k in a:
            r = same(a[k], b[k], f"{path}.{k}")
            if r:
                return r
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            r = same(x, y, f"{path}[{i}]")
            if r:
                return r
        return None
    if (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and not isinstance(a, bool)
    ):
        if (isinstance(a, float) and math.isnan(a)) and (
            isinstance(b, float) and math.isnan(b)
        ):
            return None
        return None if abs(a - b) <= 1e-9 * max(1.0, abs(a)) else f"{path}: {a} vs {b}"
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def c1(r):
    if "cert1" in r:
        return r["cert1"]
    c = r.get("cert")
    return c.get("1") if isinstance(c, dict) else None


def groups(d):
    rows = d.get("rows")
    if isinstance(rows, list):
        return {"rows": rows}
    if isinstance(rows, dict):
        return {k: v for k, v in rows.items() if isinstance(v, list)}
    return {}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


args = sys.argv[1:]
metrics = args[0] == "--metrics"
if metrics:
    args = args[1:]
tol = 0.02
if "--tol" in args:
    i = args.index("--tol")
    tol = float(args[i + 1])
    del args[i : i + 2]
old_dir, new_dir, files = args[0], args[1], args[2:]
bad = 0
for f in files:
    a = json.load(open(os.path.join(old_dir, f)))
    b = json.load(open(os.path.join(new_dir, f)))
    if not metrics:
        r = same(a, b)
        print(
            ("SAME    " if r is None else "DIFFERS ")
            + f
            + ("" if r is None else f"  ({r})")
        )
        bad += r is not None
        continue
    if b.get("status") != "ok":
        print(f"FAIL {f}: status {b.get('status')}  {str(b.get('trace', ''))[-300:]}")
        bad += 1
        continue
    if "cfg" in a and a.get("cfg") != b.get("cfg"):
        print(f"note: selected configuration differs: {a.get('cfg')} vs {b.get('cfg')}")
    ga, gb = groups(a), groups(b)
    print(f"{f}: mean over splits (shipped -> rerun)")
    for k in ga:
        if k not in gb:
            print(f"  {k}: missing in rerun")
            bad += 1
            continue
        for name, fn in (
            ("accuracy", lambda r: r.get("acc")),
            ("certified acc. b=1", c1),
        ):
            x, y = mean(map(fn, ga[k])), mean(map(fn, gb[k]))
            if x is None or y is None:
                continue
            ok = abs(x - y) <= tol
            bad += not ok
            print(
                f"  {k:10s} {name:20s} {100*x:6.2f} -> {100*y:6.2f}  {'ok' if ok else f'DIFF > {100*tol:.0f} points'}"
            )
print("PASS" if bad == 0 else f"FAIL ({bad})")
sys.exit(1 if bad else 0)
