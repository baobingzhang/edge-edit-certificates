"""E6 (PLAN_v5): exact b=1 certificates for two-layer mean-aggregation GraphSAGE.
usage: exp6.py DATASET. Writes results/s05/DATASET__sage2.json; a finished file is skipped.
"""

import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

import cert2
import exp
import gam
import run
import smooth

OUT = os.path.join(gam.ROOT, "results", "s05")
SPLIT_CAP = 3 * 3600  # PLAN_v5: a split taking longer marks the dataset as timed out


def fit_aug(ds, m, g, tr, va, cfg, seed, epochs=600, patience=200):
    """Train on randomly flipped graphs (Smooth-SAGE's selected pa, pd); keep the epoch best on the clean validation graph."""
    ei = g.edge_index.cpu().numpy()
    und = ei[:, ei[0] < ei[1]]
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best, state, wait = -1, None, 0
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        noisy = torch.as_tensor(
            smooth.sample_edges(und, g.num_nodes, cfg["pa"], cfg["pd"], rng),
            device="cuda",
            dtype=torch.long,
        )
        F.cross_entropy(m(g.x, noisy)[tr], g.y[tr]).backward()
        opt.step()
        m.eval()
        with torch.no_grad():
            v = gam.metric(ds, m(g.x, g.edge_index)[va], g.y[va])
        if v > best:
            best, wait, state = (
                v,
                0,
                {k: t.detach().clone() for k, t in m.state_dict().items()},
            )
        else:
            wait += 1
            if wait >= patience:
                break
    m.load_state_dict(state)
    return best


def train(ds, variant, g, c, ms, s, cfgs):
    torch.manual_seed(s)
    cfg = cfgs[variant]
    m = gam.Conv2(
        g.x.size(1), c, kind="sage", hidden=cfg["hidden"], drop=cfg["drop"]
    ).cuda()
    tr, va = ms[0][:, s].cuda(), ms[1][:, s].cuda()
    if variant == "sage":
        run.fit_sgd(ds, m, g, tr, va, cfg)
    else:
        fit_aug(ds, m, g, tr, va, cfg, seed=s)
    return m.eval()


def evaluate(m, g, mask, cap, seed):
    idx = mask.nonzero().flatten().numpy()
    if len(idx) > cap:
        idx = np.sort(np.random.default_rng(seed).choice(idx, cap, replace=False))
    with torch.no_grad():
        P = cert2.precompute(m, g.x, g.edge_index)
    pred = P["out"].argmax(1).cpu()
    correct = (pred[idx] == g.y.cpu()[idx]).numpy()
    ok = np.array(cert2.certify_b1(P, idx))
    deg = P["deg"].cpu().numpy()[idx].astype(int)
    return dict(
        nodes=idx.tolist(),
        correct=correct.tolist(),
        robust=ok.tolist(),
        deg=deg.tolist(),
        acc=float(correct.mean()),
        cert1=float((correct & ok).mean()),
        n=len(idx),
    )


def main():
    ds = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{ds}__sage2.json")
    if os.path.exists(path) and json.load(open(path)).get("status") == "ok":
        print(ds, "skip")
        return
    out = dict(ds=ds, status="running")
    try:
        g, ms = exp.splits(ds)
        g = g.to("cuda")
        c = int(g.y.max()) + 1
        base = json.load(
            open(os.path.join(gam.ROOT, "results", "s02", f"{ds}__sage.json"))
        )["cfg"]
        sm = json.load(
            open(os.path.join(gam.ROOT, "results", "s04", f"{ds}__smoothsage.json"))
        )["cfg"]
        cfgs = dict(sage=base, sageaug=dict(base, pa=sm["pa"], pd=sm["pd"]))
        sel = {}
        for (
            v
        ) in (
            cfgs
        ):  # selection: validation of split 0 only, score 0.5 clean + 0.5 cert_1
            r = evaluate(train(ds, v, g, c, ms, 0, cfgs), g, ms[1][:, 0], 500, 0)
            sel[v] = dict(
                acc=r["acc"], cert1=r["cert1"], score=0.5 * r["acc"] + 0.5 * r["cert1"]
            )
            print(ds, "select", v, sel[v], flush=True)
        chosen = max(sel, key=lambda v: sel[v]["score"])
        out.update(cfgs=cfgs, selection=sel, chosen=chosen, rows={})
        for (
            v
        ) in (
            cfgs
        ):  # both variants on the confirmation splits; H11 uses only the chosen one
            rows = []
            for s in range(3, ms[2].size(1)):
                t0 = time.time()
                m = train(ds, v, g, c, ms, s, cfgs)
                r = evaluate(m, g, ms[2][:, s], 1000, s)
                with torch.no_grad():
                    te = ms[2][:, s].cuda()
                    r["metric"] = gam.metric(ds, m(g.x, g.edge_index)[te], g.y[te])
                r.update(split=s, seconds=time.time() - t0)
                rows.append(r)
                print(
                    ds,
                    v,
                    s,
                    f"acc={r['acc']:.3f} cert1={r['cert1']:.3f} {r['seconds']:.0f}s",
                    flush=True,
                )
                if r["seconds"] > SPLIT_CAP:
                    out.update(status="timeout")
                    break
            out["rows"][v] = rows
            if out["status"] == "timeout":
                break
        if out["status"] == "running":
            out["status"] = "ok"
    except Exception:
        out.update(status="error", trace=traceback.format_exc())
        print(out["trace"])
    json.dump(out, open(path, "w"))
    print(ds, out["status"])


if __name__ == "__main__":
    main()
