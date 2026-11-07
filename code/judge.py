"""Automatic verdicts for H1-H10 as frozen in PLAN_v2/PLAN_v3_addendum. No model selection happens here."""
import json, os, numpy as np, gam
from scipy.stats import wilcoxon
R = os.path.join(gam.ROOT, "results")
ALL = "Cora CiteSeer PubMed Texas Wisconsin Cornell Actor roman-empire amazon-ratings minesweeper tolokers".split()
HET = ALL[3:]
def load(ds, m, stage="s03"):
    p = os.path.join(R, stage, f"{ds}__{m}.json")
    return json.load(open(p)) if os.path.exists(p) else None
def mean(d, k="acc"):
    return float(np.mean([r[k] for r in d["rows"]])) if d and d.get("status") == "ok" else None
def certm(d, b="1"):
    return float(np.mean([r["cert"][b] for r in d["rows"]])) if d and d.get("status") == "ok" else None
def best_gf(ds):
    c = {m: load(ds, m, "s02") for m in ("logreg", "mlp", "egogam")}
    m = max((k for k, v in c.items() if v and v.get("status") == "ok"), key=lambda k: c[k]["best_val"])
    return m, mean(load(ds, m))
def pistar(clean, cert_, gf):
    if clean <= gf: return 0.0
    if cert_ >= gf: return 1.0
    return min(1.0, (clean - gf) / (clean - cert_))
V = {}; T = {}
for ds in ALL:
    g, r, s, gm, gc = (load(ds, m) for m in ("certgam", "certrob", "smoothgcn", "gnamrff", "gcn"))
    gfn, gf = best_gf(ds) if all(load(ds, m, "s02") for m in ("logreg", "mlp", "egogam")) else (None, None)
    T[ds] = dict(gf_model=gfn, gf=gf, clean=mean(g), cert=certm(g), rob_clean=mean(r), rob_cert=certm(r), smooth_cert=certm(s), smooth_clean=mean(s), gnam=mean(gm), gcn=mean(gc),
                 rho=float(np.mean([x["spearman_deg"] for x in g["rows"]])) if mean(g) is not None else None,
                 top1=float(np.mean([x["top1_stable"]["1"] for x in g["rows"]])) if mean(g) is not None else None,
                 sec=float(np.median([x["sec_per_node"] for x in g["rows"]])) if mean(g) is not None else None,
                 sec_s=float(np.median([x["sec_per_node"] for x in s["rows"]])) if mean(s) is not None else None)
    t = T[ds]
    if None not in (t["clean"], t["cert"], gf): t["pi"] = pistar(t["clean"], t["cert"], gf)
    if None not in (t["rob_clean"], t["rob_cert"], gf): t["pi_rob"] = pistar(t["rob_clean"], t["rob_cert"], gf)
    t["graph_useful"] = None if t["clean"] is None or gf is None else t["clean"] - gf >= 0.03
def paired(a, b):
    a, b = np.array(a), np.array(b)
    return float(wilcoxon(a, b).pvalue) if len(a) >= 6 and np.any(a != b) else None
# H1
tried = broken = 0
for ds in ALL:
    for m in ("certgam", "certrob"):
        d = load(ds, m)
        if d and d.get("status") == "ok":
            for r in d["rows"]: tried += r.get("fuzz_tried", 0); broken += r.get("fuzz_broken", 0)
V["H1"] = dict(passed=broken == 0 and tried > 0, tried=tried, broken=broken)
# H2 (secondary)
ok = [d for d in ALL if T[d]["rob_cert"] is not None and T[d]["smooth_cert"] is not None]
wins = sum(T[d]["rob_cert"] > T[d]["smooth_cert"] for d in ok); p = paired([T[d]["rob_cert"] for d in ok], [T[d]["smooth_cert"] for d in ok]) if ok else None
V["H2"] = dict(n=len(ok), wins=wins, p=p, passed=len(ok) == 11 and wins >= 7 and p is not None and p < 0.05, note="secondary; needs all 11 datasets" )
# H3
ok = [d for d in ALL if T[d]["clean"] is not None and T[d]["gcn"] is not None]; gap = float(np.median([T[d]["gcn"] - T[d]["clean"] for d in ok])) if ok else None
ok2 = [d for d in HET if T[d]["clean"] is not None and T[d]["gnam"] is not None]; w2 = sum(T[d]["clean"] > T[d]["gnam"] for d in ok2); p2 = paired([T[d]["clean"] for d in ok2], [T[d]["gnam"] for d in ok2]) if ok2 else None
V["H3"] = dict(median_gap_gcn=gap, het_n=len(ok2), het_wins=w2, p=p2, missing=len(HET) - len(ok2), passed=gap is not None and gap <= 0.05 and w2 >= 6 and p2 is not None and p2 < 0.05)
# H4
ok = [d for d in ALL if T[d]["rob_cert"] is not None and T[d]["cert"] is not None]; w = sum(T[d]["rob_cert"] - T[d]["cert"] >= 0.03 and T[d]["clean"] - T[d]["rob_clean"] <= 0.03 for d in ok)
V["H4"] = dict(n=len(ok), count=w, passed=w >= 6)
# H5: comparison with the G-NAMRFF sound bound is added by audit_gnam_cert.py
ok = [d for d in ALL if T[d]["top1"] is not None]; w = sum(T[d]["top1"] >= 0.5 for d in ok)
V["H5"] = dict(n=len(ok), count=w, passed_absolute=len(ok) > 0 and w >= len(ALL) / 2)
# H6
ok = [d for d in ALL if T[d]["sec"] and T[d]["sec_s"]]; ratio = float(np.median([T[d]["sec_s"] / T[d]["sec"] for d in ok])) if ok else None
V["H6"] = dict(n=len(ok), median_speedup=ratio, passed=ratio is not None and ratio >= 100)
# H7
p = os.path.join(R, "s03", "synthetic.json")
if os.path.exists(p):
    d = json.load(open(p)); a, b = np.mean([x["certgam_recall"] for x in d]), np.mean([x["gnamrff_recall"] for x in d])
    V["H7"] = dict(certgam=float(a), gnamrff=float(b), passed=a - b >= 0.10)
# H8, H9
us = [d for d in ALL if T[d]["graph_useful"]]; pi = [T[d].get("pi_rob") for d in us]
have = [x for x in pi if x is not None]; c = sum(x >= 0.5 for x in have)
V["H8"] = dict(useful=us, n=len(have), count=c, passed=len(us) > 0 and len(have) == len(us) and c >= len(us) / 2)
pairs = [(T[d]["pi_rob"], T[d]["pi"]) for d in us if T[d].get("pi_rob") is not None and T[d].get("pi") is not None]; c = sum(a > b for a, b in pairs); p9 = paired([a for a, _ in pairs], [b for _, b in pairs]) if pairs else None
V["H9"] = dict(n=len(pairs), count=c, p=p9, passed=len(pairs) > 0 and c >= 2 * len(us) / 3 and (len(pairs) < 6 or (p9 is not None and p9 < 0.05)))
# H10
ok = [d for d in ALL if T[d]["rho"] is not None]; c = sum(T[d]["rho"] > 0.2 for d in ok)
V["H10"] = dict(n=len(ok), count=c, passed=len(ok) > 0 and c >= 2 * len(ALL) / 3)
conv = lambda o: o.item() if hasattr(o, "item") else str(o)
json.dump(dict(verdict=V, table=T), open(os.path.join(R, "verdict.json"), "w"), indent=1, default=conv)
print(json.dumps(V, indent=0, default=conv))
