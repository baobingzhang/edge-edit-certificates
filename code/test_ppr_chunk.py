"""exact_b1 processes rows of M in chunks; an offset error in the chunk loop would make different chunk sizes give
different worst margins. Brute force on tiny graphs never uses more than one chunk, so compare chunk sizes on real
graphs, where several chunks are used, against one chunk covering all rows."""

import json
import os

import torch

import exp
import gam
import ppr

res = []
for ds in ("Texas", "Cora", "CiteSeer"):
    g, _ = exp.splits(ds)
    n = g.num_nodes
    G = ppr.Graph(g.edge_index.cuda(), n, "cuda")
    c = int(g.y.max()) + 1
    H = torch.randn(n, c, device="cuda", dtype=torch.float64, generator=torch.Generator(device="cuda").manual_seed(9)) * 3
    nodes = list(range(0, n, max(1, n // 300)))
    for alpha in (0.1, 0.85):
        _, w0, a0, _ = ppr.exact_b1(G, H, alpha, nodes, c, row_chunk=n)
        for ch in (97, 1000):
            _, w1, a1, _ = ppr.exact_b1(G, H, alpha, nodes, c, row_chunk=ch)
            d = float((w0 - w1).abs().max())
            assert d < 1e-9, (ds, alpha, ch, d)
            res.append(dict(ds=ds, alpha=alpha, chunk=ch, nodes=len(nodes), max_diff=d))
            print("PASS", res[-1], flush=True)
json.dump(dict(status="PASS", cases=res), open(os.path.join(gam.ROOT, "results", "g10d.json"), "w"), indent=1)
print("G10d PASS")
