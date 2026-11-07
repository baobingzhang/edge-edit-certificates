"""Verdicts for H18-H20 (PLAN_v8) and H18e, H20e (PLAN_v8b) from results/s09. Thresholds copied from the frozen
plans; writes results/verdict8.json."""

import json
import os

import numpy as np

import exp
import gam

ALL = "Cora CiteSeer PubMed Texas Wisconsin Cornell Actor roman-empire amazon-ratings minesweeper tolokers".split()
AN = json.load(open(os.path.join(gam.ROOT, "results", "analysis.json")))


def majority(
    ds,
):  # same rule as judge5.py / judge7.py (importing them would rerun them)
    g, ms = exp.splits(ds)
    te = ms[2]
    return float(
        np.mean(
            [
                np.bincount(g.y[te[:, s]].numpy()).max() / int(te[:, s].sum())
                for s in range(3, te.size(1))
            ]
        )
    )


R = os.path.join(gam.ROOT, "results")
SMALL = ["Texas", "Cornell", "Wisconsin", "Cora", "CiteSeer", "Actor"]


def load(d, tag):
    p = os.path.join(R, "s09", f"{d}__{tag}.json")
    x = json.load(open(p)) if os.path.exists(p) else None
    return x if x and x.get("status") == "ok" else None


def strata(rows, robust):
    deg = np.concatenate([r["deg"] for r in rows])
    ca = np.concatenate([np.array(r["correct"]) & robust(r) for r in rows])
    lo, mid = deg <= 1, (deg >= 4) & (deg <= 7)
    return (
        int(lo.sum()),
        int(mid.sum()),
        (float(ca[lo].mean()) if lo.any() else None),
        (float(ca[mid].mean()) if mid.any() else None),
    )


ref = {d: AN[d]["by_ref"][AN[d]["selected"]]["acc"] for d in ALL}
triv = {d: AN[d]["best_test_ref"] <= majority(d) + 0.005 for d in ALL}
tab, h18, h19, h20, h18e, h20e = {}, [], [], [], [], []
for d in ALL:
    row = dict(ref_sel=ref[d], trivial=triv[d])
    x = load(d, "pippnp")
    if x:
        rs = x["rows"]
        row.update(
            cfg=x["cfg"],
            robust_trained=x["robust"],
            acc=float(np.mean([r["acc"] for r in rs])),
            cert_local={
                b: float(np.mean([r["cert"][b] for r in rs])) for b in rs[0]["cert"]
            },
            ub1=float(np.mean([r["ub1"] for r in rs])),
            unsound=int(sum(r["unsound"] for r in rs)),
            attack_tried=int(sum(r["attack_tried"] for r in rs)),
        )
        nl, nm, lo, mi = strata(rs, lambda r: np.array(r["rad"]) >= 1)
        row.update(n_low=nl, n_mid=nm, low=lo, mid=mi)
        if not triv[d]:
            h18.append(row["cert_local"]["1"] < ref[d])
            h19.append(row["ub1"] < ref[d])
        if nl >= 30 and nm >= 30:
            h20.append(lo < mi)
    xe = load(d, "pippnp_exact")
    if xe:
        rs = xe["rows"]
        row.update(
            e_cfg=xe["cfg"],
            e_robust_trained=xe["robust"],
            e_acc=float(np.mean([r["acc"] for r in rs])),
            e_cert1=float(np.mean([r["cert1"] for r in rs])),
            e_cert1_sd=float(np.std([r["cert1"] for r in rs])),
            e_acc_sd=float(np.std([r["acc"] for r in rs])),
        )
        nl, nm, lo, mi = strata(rs, lambda r: np.array(r["robust"]))
        row.update(e_n_low=nl, e_n_mid=nm, e_low=lo, e_mid=mi)
        if d in SMALL and not triv[d]:
            h18e.append(row["e_cert1"] < ref[d])
        if nl >= 30 and nm >= 30:
            h20e.append(lo < mi)
    tab[d] = row

V = dict(
    H18=dict(n=len(h18), count=int(sum(h18)), passed=len(h18) == 9 and sum(h18) >= 7),
    H19=dict(n=len(h19), count=int(sum(h19)), passed=len(h19) == 9 and sum(h19) >= 5),
    H20=dict(
        n=len(h20), count=int(sum(h20)), passed=bool(h20) and sum(h20) >= 0.7 * len(h20)
    ),
    H18e=dict(
        n=len(h18e), count=int(sum(h18e)), passed=len(h18e) == 6 and sum(h18e) >= 5
    ),
    H20e=dict(
        n=len(h20e),
        count=int(sum(h20e)),
        passed=bool(h20e) and sum(h20e) >= 0.7 * len(h20e),
    ),
)
json.dump(
    dict(verdict=V, table=tab), open(os.path.join(R, "verdict8.json"), "w"), indent=1
)
print(json.dumps(V, indent=1))
for d, v in tab.items():
    print(
        d,
        {
            k: (round(t, 3) if isinstance(t, float) else t)
            for k, t in v.items()
            if k != "cert_local"
        },
    )
