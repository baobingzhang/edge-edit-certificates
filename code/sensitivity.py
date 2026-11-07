"""Pi* with clean and certified accuracy from the same test nodes, and the joint explanation certificate. Main verdict formula is untouched."""
import json, os, numpy as np, gam
R = os.path.join(gam.ROOT, "results"); P = json.load(open(os.path.join(R, "s03/post.json"))); V = json.load(open(os.path.join(R, "verdict.json")))["table"]
def pi(c, ce, g): return 0.0 if c <= g else (1.0 if ce >= g else min(1.0, (c - g) / (c - ce)))
out = {}
for ds, t in V.items():
    a, b = P[f"{ds}__certgam"], P[f"{ds}__certrob"]; m = lambda rows, f: float(np.mean([f(r) for r in rows]))
    c, ce, rc, rce = m(a, lambda r: r["clean_same_nodes"]), m(a, lambda r: r["cert"]["1"]), m(b, lambda r: r["clean_same_nodes"]), m(b, lambda r: r["cert"]["1"])
    out[ds] = dict(gf=t["gf"], clean=c, cert=ce, rob_clean=rc, rob_cert=rce, useful_same=c - t["gf"] >= 0.03, pi=pi(c, ce, t["gf"]), pi_rob=pi(rc, rce, t["gf"]), main_pi=t.get("pi"), main_pi_rob=t.get("pi_rob"), top1=m(a, lambda r: r["top1"]["1"]), joint=m(a, lambda r: r["joint"]["1"]), rob_top1=m(b, lambda r: r["top1"]["1"]), rob_joint=m(b, lambda r: r["joint"]["1"]), n=a[0]["n"])
json.dump(out, open(os.path.join(R, "sensitivity.json"), "w"), indent=1)
us = [d for d in out if out[d]["useful_same"]]; print("useful", us, "pi_rob>=.5", sum(out[d]["pi_rob"] >= .5 for d in us), "rob>plain", sum(out[d]["pi_rob"] > out[d]["pi"] for d in us))
print("ds n gf clean cert rclean rcert | pi(main) pi_rob(main) | top1 joint rtop1 rjoint")
for d, o in out.items(): print(d, o["n"], *(round(o[k], 3) for k in ("gf", "clean", "cert", "rob_clean", "rob_cert")), "|", round(o["pi"], 3), f"({o['main_pi']:.3f})", round(o["pi_rob"], 3), f"({o['main_pi_rob']:.3f})", "|", *(round(o[k], 3) for k in ("top1", "joint", "rob_top1", "rob_joint")))
