"""Verdicts for H14-H17 (PLAN_v7) from results/s07. Thresholds are copied from the frozen plan; writes results/verdict7.json."""

import json
import os

import numpy as np
from scipy.stats import wilcoxon

import gam
import exp

R = os.path.join(gam.ROOT, "results")
ALL = "Cora CiteSeer PubMed Texas Wisconsin Cornell Actor roman-empire amazon-ratings minesweeper tolokers".split()
AN = json.load(open(os.path.join(R, "analysis.json")))


def majority(ds):  # same rule as judge5.py (importing judge5 would rerun it)
    g, ms = exp.splits(ds)
    te = ms[2]
    return float(np.mean([np.bincount(g.y[te[:, s]].numpy()).max() / int(te[:, s].sum()) for s in range(3, te.size(1))]))


def mean(rows, f):
    return float(np.mean([f(r) for r in rows]))


def strata(rows):
    deg = np.concatenate([r["deg"] for r in rows])
    ca = np.concatenate(
        [np.array(r["correct"]) & (np.array(r["rad"]) >= 1) for r in rows]
    )
    lo, mid = deg <= 1, (deg >= 4) & (deg <= 7)
    return (
        int(lo.sum()),
        int(mid.sum()),
        (float(ca[lo].mean()) if lo.any() else None),
        (float(ca[mid].mean()) if mid.any() else None),
    )


out, d14, h15, h16 = {}, [], [], []
for d in ALL:
    p = os.path.join(R, "s07", f"{d}__shrink.json")
    x = json.load(open(p)) if os.path.exists(p) else None
    if not x or x.get("status") != "ok":
        out[d] = dict(status=x.get("status") if x else "absent")
        continue
    ch, k0 = x["rows"]["chosen"], x["rows"]["kappa0"]
    c1, c0 = mean(ch, lambda r: r["cert"]["1"]), mean(k0, lambda r: r["cert"]["1"])
    ref = AN[d]["by_ref"][AN[d]["selected"]]["acc"]
    trivial = AN[d]["best_test_ref"] <= majority(d) + 0.005
    nl, nm, l1, m1 = strata(ch)
    _, _, l0, m0 = strata(k0)
    out[d] = dict(
        status="ok",
        kappa=x["kappa"],
        cert1=c1,
        cert1_k0=c0,
        acc=mean(ch, lambda r: r["acc"]),
        acc_k0=mean(k0, lambda r: r["acc"]),
        cert1_s03=AN[d]["rob_cert"],
        ref_sel=ref,
        trivial=trivial,
        n_low=nl,
        n_mid=nm,
        gain_low=(l1 - l0) if nl else None,
        gain_mid=(m1 - m0) if nm else None,
        fuzz_broken=sum(r["fuzz_broken"] for r in ch + k0),
        cert={b: mean(ch, lambda r: r["cert"][b]) for b in ch[0]["cert"]},
        cert_k0={b: mean(k0, lambda r: r["cert"][b]) for b in k0[0]["cert"]},
    )
    d14.append(c1 - c0)
    if not trivial:
        h15.append(c1 > ref)
    if nl >= 30 and nm >= 30:
        h16.append(out[d]["gain_low"] > out[d]["gain_mid"])

n_ok = len(d14)
nz = [v for v in d14 if v != 0]
p14 = float(wilcoxon(nz, alternative="greater").pvalue) if len(nz) >= 1 else 1.0
V = dict(
    H14=dict(
        n=n_ok,
        count=int(sum(v > 0 for v in d14)),
        p=p14,
        passed=n_ok == 11 and sum(v > 0 for v in d14) >= 7 and p14 < 0.05,
        note="Wilcoxon on non-zero paired differences (zero differences dropped, scipy default 'wilcox')",
    ),
    H15=dict(n=len(h15), count=int(sum(h15)), passed=len(h15) == 9 and sum(h15) >= 5),
    H16=dict(
        n=len(h16), count=int(sum(h16)), passed=bool(h16) and sum(h16) >= 0.7 * len(h16)
    ),
)
json.dump(
    dict(verdict=V, table=out), open(os.path.join(R, "verdict7.json"), "w"), indent=1
)
print(json.dumps(V, indent=1))
for d, v in out.items():
    print(
        d,
        {
            k: (round(t, 3) if isinstance(t, float) else t)
            for k, t in v.items()
            if k not in ("cert", "cert_k0")
        },
    )
