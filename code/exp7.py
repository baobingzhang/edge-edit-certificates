"""E7 (PLAN_v6, exploratory): two-layer SAGE trained against the exact worst single edge edit, certified exactly.
usage: exp7.py DATASET. Writes results/s06/DATASET__sageadv.json; a finished file is skipped.
"""

import json
import os
import sys
import time
import traceback

import cert2
import exp
import gam
import numpy as np
import torch
import torch.nn.functional as F
from exp6 import evaluate

OUT = os.path.join(gam.ROOT, "results", "s06")


def worst_ce(P, i, y, rng, cap_u=32):
    """Cross-entropy of node i under its worst single toggle (exact over v; u subsampled above cap_u during training)."""
    U = torch.unique(
        torch.cat([cert2.neighbours(P, i), torch.tensor([i], device=P["Z"].device)])
    )
    if len(U) > cap_u:
        U = U[
            torch.as_tensor(rng.choice(len(U), cap_u, replace=False), device=U.device)
        ]
    us = U.tolist()
    allv = torch.arange(P["n"], device=U.device)
    o = cert2.toggled_logits(P, i, us, allv)  # k x n x C
    ce = F.cross_entropy(
        o.reshape(-1, o.size(-1)), y.expand(o.size(0) * o.size(1)), reduction="none"
    ).view(o.size(0), -1)
    ce[torch.arange(len(us)), U] = -1.0  # u == v is not an edit
    return torch.maximum(ce.max(), F.cross_entropy(P["out"][i : i + 1], y.view(1)))


def train_adv(ds, g, c, ms, s, cfg, steps=400, batch=16):
    torch.manual_seed(s)
    rng = np.random.default_rng(s)
    m = gam.Conv2(
        g.x.size(1), c, kind="sage", hidden=cfg["hidden"], drop=cfg["drop"]
    ).cuda()
    tr = ms[0][:, s].nonzero().flatten().numpy()
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best, state = -1.0, None
    for t in range(1, steps + 1):
        m.train()
        opt.zero_grad()
        clean = F.cross_entropy(m(g.x, g.edge_index)[tr], g.y[tr])
        m.eval()  # the adversary attacks the deterministic model that will be certified
        P = cert2.precompute(m, g.x, g.edge_index)
        adv = torch.stack(
            [
                worst_ce(P, int(i), g.y[int(i)], rng)
                for i in rng.choice(tr, min(batch, len(tr)), replace=False)
            ]
        ).mean()
        (0.5 * clean + 0.5 * adv).backward()
        opt.step()
        if t % 25 == 0:
            r = evaluate(m.eval(), g, ms[1][:, s], 300, s)
            sc = 0.5 * r["acc"] + 0.5 * r["cert1"]
            if sc > best:
                best, state = sc, {
                    k: v.detach().clone() for k, v in m.state_dict().items()
                }
    m.load_state_dict(state)
    return m.eval(), best


def main():
    ds = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{ds}__sageadv.json")
    if os.path.exists(path) and json.load(open(path)).get("status") == "ok":
        print(ds, "skip")
        return
    out = dict(ds=ds, status="running", rows=[])
    try:
        g, ms = exp.splits(ds)
        g = g.to("cuda")
        c = int(g.y.max()) + 1
        cfg = json.load(
            open(os.path.join(gam.ROOT, "results", "s02", f"{ds}__sage.json"))
        )["cfg"]
        out["cfg"] = cfg
        for s in range(3, ms[2].size(1)):
            t0 = time.time()
            m, vbest = train_adv(ds, g, c, ms, s, cfg)
            r = evaluate(m, g, ms[2][:, s], 1000, s)
            r.update(split=s, val_score=vbest, seconds=time.time() - t0)
            out["rows"].append(r)
            print(
                ds,
                "sageadv",
                s,
                f"acc={r['acc']:.3f} cert1={r['cert1']:.3f} {r['seconds']:.0f}s",
                flush=True,
            )
        out["status"] = "ok"
    except Exception:
        out.update(status="error", trace=traceback.format_exc())
        print(out["trace"])
    json.dump(out, open(path, "w"))
    print(ds, out["status"])


if __name__ == "__main__":
    main()
