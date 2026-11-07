"""E5 (PLAN_v4): split-level bootstrap intervals for pi*, sensitivity to each graph-free reference, and same-class references."""
import json, os, numpy as np, gam
R = os.path.join(gam.ROOT, "results"); ALL = "Cora CiteSeer PubMed Texas Wisconsin Cornell Actor roman-empire amazon-ratings minesweeper tolokers".split()
L = lambda d, m: json.load(open(os.path.join(R, "s03", f"{d}__{m}.json")))
def pi(c, ce, g): return 0.0 if c <= g else (1.0 if ce >= g else min(1.0, (c - g) / (c - ce)))
rng = np.random.default_rng(0); out = {}
for d in ALL:
    a, b = L(d, "certgam")["rows"], L(d, "certrob")["rows"]; refs = {m: [r["acc"] for r in L(d, m)["rows"]] for m in ("logreg", "mlp", "egogam")}
    arr = lambda rows, f: np.array([f(r) for r in rows]); c, ce, rc, rce = arr(a, lambda r: r["acc"]), arr(a, lambda r: r["cert"]["1"]), arr(b, lambda r: r["acc"]), arr(b, lambda r: r["cert"]["1"])
    sel = json.load(open(os.path.join(R, "verdict.json")))["table"][d]["gf_model"]; o = dict(selected=sel, by_ref={})
    for m, v in refs.items():
        v = np.array(v); o["by_ref"][m] = dict(acc=float(v.mean()), pi=pi(c.mean(), ce.mean(), v.mean()), pi_rob=pi(rc.mean(), rce.mean(), v.mean()), gain=float(c.mean() - v.mean()), gain_rob=float(rc.mean() - v.mean()))
    v = np.array(refs[sel]); n = len(v); bs = rng.integers(0, n, (2000, n)); samp = lambda arr_: arr_[bs].mean(1)
    pis = np.array([pi(x, y, z) for x, y, z in zip(samp(rc), samp(rce), samp(v))]); pip = np.array([pi(x, y, z) for x, y, z in zip(samp(c), samp(ce), samp(v))])
    o.update(ci_rob=[float(np.quantile(pis, .025)), float(np.quantile(pis, .975))], ci=[float(np.quantile(pip, .025)), float(np.quantile(pip, .975))], per_split_rob=[pi(x, y, z) for x, y, z in zip(rc, rce, v)], share=float(np.mean([r["graph_share"] for r in b])))
    best = {m: refs[m] for m in refs}; o["best_test_ref"] = max(np.mean(x) for x in best.values()); o["rob_cert"] = float(rce.mean()); o["ego_gain"] = float(c.mean() - np.mean(refs["egogam"])); out[d] = o
    print(d, "sel", sel, "pi_rob ci", [round(x, 2) for x in o["ci_rob"]], "by_ref", {m: round(x["pi_rob"], 2) for m, x in o["by_ref"].items()}, "ego_gain", round(o["ego_gain"], 3), "share", round(o["share"], 2))
json.dump(out, open(os.path.join(R, "analysis.json"), "w"), indent=1)
print("rob cert below best-test graph-free on", sum(o["rob_cert"] < o["best_test_ref"] for o in out.values()), "of 11; graph useful vs ego", sum(o["ego_gain"] >= .03 for o in out.values()))
