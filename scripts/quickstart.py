"""Minimal example (CPU, under a minute): train CertGAM on one split with its frozen configuration, certify every
test node exactly against 1, 2, 3, 5 and 10 edge edits, and put the result next to the graph-free reference.

usage: python scripts/quickstart.py [DATASET] [SPLIT]      (default: Texas 3)
"""

import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code")
)
import torch  # noqa: E402

import cert  # noqa: E402
import exp  # noqa: E402
import gam  # noqa: E402

ds = sys.argv[1] if len(sys.argv) > 1 else "Texas"
sp = int(sys.argv[2]) if len(sys.argv) > 2 else 3
R = os.path.join(gam.ROOT, "results")

g, masks = exp.splits(ds)
c, y = int(g.y.max()) + 1, g.y
tr, te = masks[0][:, sp], masks[2][:, sp]
cfg = json.load(open(os.path.join(R, "s02", f"{ds}__certgam.json")))[
    "cfg"
]  # chosen on validation of split 0

m = cert.NeighbourGAM(g, c, K=cfg["K"], theta=cfg["theta"])
cert.fit(m, tr, y, cfg["lam"])  # convex ridge training
nodes = te.nonzero().flatten().numpy()
pred, rad = cert.radii(m, nodes, exp.BUD)  # exact certified radius of every test node
ok = pred == y[nodes].numpy()
with torch.no_grad():
    S0, S1 = m.scores()
print(f"{ds}, split {sp}: CertGAM {cfg}, {len(nodes)} test nodes")
print(f"  clean accuracy          {100 * ok.mean():5.1f}")
for b in exp.BUD:
    print(f"  certified, {b:2d} edit(s)   {100 * (ok & (rad >= b)).mean():5.1f}")
print(
    f"  neighbour share of score variation {float(S1.std() / (S0.std() + S1.std())):.2f}"
)

an = json.load(open(os.path.join(R, "analysis.json")))
if ds in an:
    ref = an[ds]["by_ref"][an[ds]["selected"]]
    print(
        f"  graph-free reference ({an[ds]['selected']}, mean over splits 3-9): {100 * ref['acc']:.1f}"
        " -- edge edits cannot change it"
    )
