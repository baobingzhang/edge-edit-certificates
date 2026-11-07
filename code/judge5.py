"""Verdicts for H11-H12 (PLAN_v5) from results/s05. Thresholds are copied from the frozen plan; writes results/verdict5.json."""

import json
import os

import numpy as np

import exp
import gam

R = os.path.join(gam.ROOT, "results")
ALL = "Cora CiteSeer PubMed Texas Wisconsin Cornell Actor roman-empire amazon-ratings minesweeper tolokers".split()
AN = json.load(open(os.path.join(R, "analysis.json")))


def majority(
    ds,
):  # same rule as make_paper.py: majority-class test accuracy on splits 3-9
    g, ms = exp.splits(ds)  # Planetoid: the 10 random splits, not the single public one
    te = ms[2]
    return float(
        np.mean(
            [
                np.bincount(g.y[te[:, s]].numpy()).max() / int(te[:, s].sum())
                for s in range(3, te.size(1))
            ]
        )
    )


def load(ds):
    p = os.path.join(R, "s05", f"{ds}__sage2.json")
    return json.load(open(p)) if os.path.exists(p) else None


out, h11, h12 = {}, [], []
for d in ALL:
    x = load(d)
    if not x or x.get("status") != "ok":
        out[d] = dict(status=x.get("status") if x else "absent")
        continue
    rows = x["rows"][x["chosen"]]
    cert = float(np.mean([r["cert1"] for r in rows]))
    acc = float(np.mean([r["acc"] for r in rows]))
    ref = AN[d]["by_ref"][AN[d]["selected"]]["acc"]
    trivial = AN[d]["best_test_ref"] <= majority(d) + 0.005
    deg = np.concatenate([r["deg"] for r in rows])
    ca = np.concatenate([np.array(r["correct"]) & np.array(r["robust"]) for r in rows])
    lo, mid = deg <= 1, (deg >= 4) & (deg <= 7)
    strata = dict(
        n_low=int(lo.sum()),
        n_mid=int(mid.sum()),
        c1_low=float(ca[lo].mean()) if lo.any() else None,
        c1_mid=float(ca[mid].mean()) if mid.any() else None,
    )
    out[d] = dict(
        status="ok",
        chosen=x["chosen"],
        acc=acc,
        cert1=cert,
        ref_sel=ref,
        ref_best=AN[d]["best_test_ref"],
        certrob=AN[d]["rob_cert"],
        trivial=trivial,
        below_ref=cert < ref,
        **strata,
    )
    if not trivial:
        h11.append(cert < ref)
    if strata["n_low"] >= 30 and strata["n_mid"] >= 30:
        h12.append(strata["c1_low"] < strata["c1_mid"])
nt_total = sum(
    1
    for d in ALL
    if d in out and out[d].get("status") == "ok" and not out[d]["trivial"]
)
V = dict(
    H11=dict(
        n=len(h11),
        count=int(sum(h11)),
        passed=len(h11) >= 9 and sum(h11) >= 7,
        note="threshold >=7 of the 9 non-trivial datasets; fewer than 9 completed means not judged",
    ),
    H12=dict(
        n=len(h12), count=int(sum(h12)), passed=bool(h12) and sum(h12) >= 0.8 * len(h12)
    ),
)
json.dump(
    dict(verdict=V, table=out), open(os.path.join(R, "verdict5.json"), "w"), indent=1
)
print(json.dumps(V, indent=1))
for d, v in out.items():
    print(d, {k: (round(t, 3) if isinstance(t, float) else t) for k, t in v.items()})
