"""PLAN_v8c fixes must not change what RCE/CEM optimise. The robust loss is built from adv_logits, so the patched
version (pair chunks, hand-written backward, stall stop in policy iteration) must give the same worst-case logits
and the same gradient w.r.t. H as the version that produced the E10 results already on disk; a wrong transpose in
the backward or a wrong column index after chunking would show up here as a mismatch.
"""

import importlib.util
import json
import os

import torch

import exp
import gam
import ppr

spec = importlib.util.spec_from_file_location(
    "ppr_old", os.path.join(gam.ROOT, "results", "ppr_at_s09.py")
)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def check(ds, alpha, chunk):
    g, ms = exp.splits(ds)
    G = ppr.Graph(g.edge_index.cuda(), g.num_nodes, "cuda")
    y = g.y.cuda()
    c = int(y.max()) + 1
    obs = torch.cat(
        [ms[0][:, 0].nonzero().flatten(), ms[1][:, 0].nonzero().flatten()]
    ).cuda()
    H0 = (
        torch.randn(
            g.num_nodes,
            c,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(7),
        )
        * 2
    )
    W = torch.randn(
        len(obs),
        c,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(8),
    )
    out = []
    for f in (
        lambda H: old.adv_logits(G, H, alpha, y[obs], obs, b=1),
        lambda H: ppr.adv_logits(G, H, alpha, y[obs], obs, b=1, chunk_states=chunk),
    ):
        H = H0.clone().requires_grad_(True)
        A = f(H)
        (A * W).sum().backward()
        out.append((A.detach(), H.grad.detach()))
    dv = float((out[0][0] - out[1][0]).abs().max())
    dg = float((out[0][1] - out[1][1]).abs().max())
    scale = float(out[0][1].abs().max())
    assert dv < 1e-4 and dg < 1e-4 * (1 + scale), (ds, alpha, dv, dg)
    return dict(
        ds=ds,
        alpha=alpha,
        chunk=chunk,
        max_value_diff=dv,
        max_grad_diff=dg,
        grad_scale=scale,
    )


if __name__ == "__main__":
    res = []
    for ds in ("Texas", "Cora"):
        for alpha in (0.1, 0.85):
            for chunk in (
                4_000_000,
                3 * 2708,
            ):  # one chunk, and several chunks of 3 pairs on Cora
                r = check(ds, alpha, chunk)
                print("PASS", r, flush=True)
                res.append(r)
    json.dump(
        dict(status="PASS", cases=res),
        open(os.path.join(gam.ROOT, "results", "g10c.json"), "w"),
        indent=1,
    )
    print("G10c PASS")
