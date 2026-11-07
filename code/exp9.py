"""E9 (PLAN_v7, exploratory): E7 with a 4x training budget (1600 steps, 32 adversarial nodes per step).
usage: exp9.py DATASET -> results/s08/DATASET__sageadv4.json"""

import json
import os
import sys
import time
import traceback

import exp
import exp7
import gam
from exp6 import evaluate

OUT = os.path.join(gam.ROOT, "results", "s08")


def main():
    ds = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{ds}__sageadv4.json")
    if os.path.exists(path) and json.load(open(path)).get("status") == "ok":
        print(ds, "skip")
        return
    out = dict(ds=ds, status="running", rows=[], steps=1600, batch=32)
    try:
        g, ms = exp.splits(ds)
        g = g.to("cuda")
        c = int(g.y.max()) + 1
        out["cfg"] = cfg = json.load(open(os.path.join(gam.ROOT, "results", "s02", f"{ds}__sage.json")))["cfg"]
        for s in range(3, ms[2].size(1)):
            t0 = time.time()
            m, vbest = exp7.train_adv(ds, g, c, ms, s, cfg, steps=1600, batch=32)
            r = evaluate(m, g, ms[2][:, s], 1000, s)
            r.update(split=s, val_score=vbest, seconds=time.time() - t0)
            out["rows"].append(r)
            print(ds, "sageadv4", s, f"acc={r['acc']:.3f} cert1={r['cert1']:.3f} {r['seconds']:.0f}s", flush=True)
        out["status"] = "ok"
    except Exception:
        out.update(status="error", trace=traceback.format_exc())
        print(out["trace"])
    json.dump(out, open(path, "w"))
    print(ds, out["status"])


if __name__ == "__main__":
    main()
