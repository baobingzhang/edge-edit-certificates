"""E8 (PLAN_v7): ShrinkGAM-R. Select kappa on split-0 validation, then evaluate the chosen kappa and the paired
kappa=0 baseline on splits 3-9 with exact certificates. usage: exp8.py DATASET -> results/s07/DATASET__shrink.json
"""

import json
import os
import sys
import time
import traceback

import numpy as np
import torch

import cert
import exp
import gam
import shrink

KAPPAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
OUT = os.path.join(gam.ROOT, "results", "s07")


def train(m, tr_mask, y, cfg):
    idx = tr_mask.nonzero().flatten()
    if cfg.get("solver") == "cutting":
        shrink.fit_cutting(m, idx, y, cfg["lam"], cfg["B"])
    else:
        shrink.fit_robust(m, idx, y, cfg["lam"], cfg["B"], cfg["lr"])


def evaluate(ds, m, mask, y, cap, seed, fuzz=True):
    rng = np.random.default_rng(seed)
    nodes = mask.nonzero().flatten().cpu().numpy()
    if len(nodes) > cap:
        nodes = rng.choice(nodes, cap, replace=False)
    with torch.no_grad():
        s = exp.scores(ds, m(), mask, y)
    pred, rad = shrink.radii(m, nodes, exp.BUD)
    ok = pred == y[nodes].cpu().numpy()
    out = dict(
        acc=s["acc"],
        metric=s["metric"],
        cert={str(b): float((ok & (rad >= b)).mean()) for b in exp.BUD},
        nodes=nodes.tolist(),
        deg=m.deg[nodes].cpu().numpy().astype(int).tolist(),
        correct=ok.tolist(),
        rad=rad.tolist(),
    )
    if (
        fuzz
    ):  # real edits on certified nodes, recomputed from scratch through an independent path
        with torch.no_grad():
            S0, S1 = (t.cpu().numpy() for t in m.scores())
        start, dst = cert.neighbour_lists(m)
        tried = broke = 0

        def logit(i, cur):
            den = len(cur) + m.kappa
            return S0[i] + (S1[cur].sum(0) / den if den > 0 else 0.0)

        for t in np.where(rad >= 1)[0][:150]:
            i, r, y0 = int(nodes[t]), int(rad[t]), int(pred[t])
            nb = set(dst[start[i] : start[i + 1]].tolist())
            for _ in range(40):
                e = set(
                    rng.choice(m.n, int(rng.integers(1, r + 1)), replace=False).tolist()
                ) - {i}
                tried += 1
                broke += int(logit(i, sorted(nb ^ e)).argmax() != y0)
            for c in range(m.c):
                if c == y0:
                    continue
                sc = S1[:, y0] - S1[:, c]
                add = [u for u in np.argsort(sc) if u not in nb and u != i][:r]
                drop = sorted(nb, key=lambda u: -sc[u])
                for p in range(0, min(r, len(drop)) + 1):
                    cur = sorted((nb - set(drop[:p])) | set(add[: r - p]))
                    tried += 1
                    broke += int(logit(i, cur).argmax() != y0)
        out.update(fuzz_tried=tried, fuzz_broken=broke)
    return out


def main():
    ds = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{ds}__shrink.json")
    if os.path.exists(path) and json.load(open(path)).get("status") == "ok":
        print(ds, "skip")
        return
    t0 = time.time()
    out = dict(ds=ds, status="running")
    try:
        g, ms = exp.splits(ds)
        g = g.to("cuda")
        ms = [x.to("cuda") for x in ms]
        c, y = int(g.y.max()) + 1, g.y
        cfg = json.load(
            open(os.path.join(gam.ROOT, "results", "s02", f"{ds}__certrob.json"))
        )["cfg"]
        m = shrink.ShrinkGAM(g, c, K=cfg["K"], theta=cfg["theta"]).cuda()
        trials = []
        for k in KAPPAS:
            m.set_kappa(k)
            train(m, ms[0][:, 0], y, cfg)
            e = evaluate(ds, m, ms[1][:, 0], y, 1000, 0, fuzz=False)
            trials.append(
                dict(
                    kappa=k,
                    val=0.5 * e["acc"] + 0.5 * e["cert"]["1"],
                    acc=e["acc"],
                    c1=e["cert"]["1"],
                )
            )
            print(ds, "val", trials[-1], flush=True)
        best = max(
            trials, key=lambda t: (t["val"], -t["kappa"])
        )  # ties -> smaller kappa
        out.update(
            cfg=cfg,
            kappa=best["kappa"],
            trials=trials,
            rows={"chosen": [], "kappa0": []},
        )
        for s in range(3, ms[2].size(1)):
            for tag, k in (("chosen", best["kappa"]), ("kappa0", 0.0)):
                if tag == "kappa0" and best["kappa"] == 0.0:
                    out["rows"]["kappa0"].append(out["rows"]["chosen"][-1])
                    continue
                m.set_kappa(k)
                t1 = time.time()
                train(m, ms[0][:, s], y, cfg)
                r = evaluate(ds, m, ms[2][:, s], y, 2000, s)
                r.update(split=s, kappa=k, train_sec=time.time() - t1)
                out["rows"][tag].append(r)
                print(
                    ds,
                    tag,
                    s,
                    k,
                    f"acc={r['acc']:.3f} c1={r['cert']['1']:.3f} fuzz={r['fuzz_broken']}/{r['fuzz_tried']}",
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
