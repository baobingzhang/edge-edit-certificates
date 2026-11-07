"""E10e (PLAN_v8b): pi-PPNP selected and evaluated with the exact single-toggle certificate (ppr.exact_b1).
usage: exp10e.py DATASET -> results/s09/DATASET__pippnp_exact.json
"""

import json
import os
import sys
import time
import traceback

import torch

import exp
import ppr
from exp10 import ALPHAS, LIMIT_SEC, LOSSES, OUT, sample, train


def evaluate(ds, G, m, x, y, alpha, mask, cap, seed):
    nclass = int(y.max()) + 1
    with torch.no_grad():
        H = m(x)
    nodes = sample(mask, cap, seed)
    t0 = time.time()
    pred, worst, arg, logits = ppr.exact_b1(G, H, alpha, nodes, nclass)
    s = exp.scores(ds, logits.float(), mask, y)
    ok = pred.cpu().numpy()[nodes] == y.cpu().numpy()[nodes]
    rob = worst.cpu().numpy() > 0
    return dict(
        acc=s["acc"],
        metric=s["metric"],
        nodes=nodes.tolist(),
        correct=ok.tolist(),
        robust=rob.tolist(),
        worst=worst.cpu().numpy().tolist(),
        arg=arg.cpu().numpy().tolist(),
        deg=G.deg.cpu().numpy()[nodes].astype(int).tolist(),
        cert1=float((ok & rob).mean()),
        cert_sec=time.time() - t0,
    )


def main():
    ds = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{ds}__pippnp_exact.json")
    if os.path.exists(path) and json.load(open(path)).get("status") == "ok":
        print(ds, "skip")
        return
    t0 = time.time()
    out = dict(ds=ds, status="running")
    try:
        g, ms = exp.splits(ds)
        G = ppr.Graph(g.edge_index.cuda(), g.num_nodes, "cuda")
        x, y = g.x.cuda().float(), g.y.cuda()
        tr0, va0 = (ms[k][:, 0].nonzero().flatten().cuda() for k in (0, 1))
        timing = {}
        for a in ALPHAS:  # PLAN_v8 timing rule, identical to exp10
            _, info = train(G, x, y, tr0, va0, a, "ce", 0)
            for k in ("rce", "cem"):
                _, ti = train(G, x, y, tr0, va0, a, k, 0, epochs_cap=5)
                timing[f"{a}_{k}"] = dict(
                    per_epoch=ti["sec"] / 5,
                    ce_epochs=info["epochs"],
                    est=ti["sec"] / 5 * info["epochs"],
                )
        robust = all(v["est"] <= LIMIT_SEC for v in timing.values())
        out.update(timing=timing, robust=robust)
        trials = []
        for a in ALPHAS:
            for k in LOSSES if robust else ["ce"]:
                m, info = train(G, x, y, tr0, va0, a, k, 0)
                e = evaluate(ds, G, m, x, y, a, ms[1][:, 0], 1000, 0)
                trials.append(
                    dict(
                        alpha=a,
                        loss=k,
                        acc=e["acc"],
                        c1=e["cert1"],
                        val=0.5 * e["acc"] + 0.5 * e["cert1"],
                        **info,
                    )
                )
                print(
                    ds,
                    "val",
                    {
                        kk: (round(v, 4) if isinstance(v, float) else v)
                        for kk, v in trials[-1].items()
                    },
                    flush=True,
                )
        best = max(
            trials, key=lambda t: (t["val"], -LOSSES.index(t["loss"]), t["alpha"])
        )
        out.update(
            trials=trials, cfg=dict(alpha=best["alpha"], loss=best["loss"]), rows=[]
        )
        for s in range(3, ms[2].size(1)):
            tr, va = (ms[k][:, s].nonzero().flatten().cuda() for k in (0, 1))
            m, info = train(G, x, y, tr, va, best["alpha"], best["loss"], s)
            r = evaluate(ds, G, m, x, y, best["alpha"], ms[2][:, s], 1000, s)
            r.update(split=s, **info)
            out["rows"].append(r)
            print(
                ds,
                s,
                out["cfg"],
                f"acc={r['acc']:.3f} c1={r['cert1']:.3f} ({r['cert_sec']:.0f}s)",
                flush=True,
            )
        out["status"] = "ok"
    except Exception:
        out.update(status="error", trace=traceback.format_exc())
        print(out["trace"])
    out["sec"] = time.time() - t0
    json.dump(out, open(path, "w"))
    print(ds, out["status"], f"{out['sec']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
