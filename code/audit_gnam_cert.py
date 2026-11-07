"""H5 comparison: what can be certified for G-NAMRFF's top-1 attribution with the sound bound of Thm 4?

One incident edge edit changes the normalised Laplacian by some Delta; ||Delta||_2 is computed exactly per node and edit on a
small sample using the worst of a few candidate edits, which only makes the reported certificate more optimistic. Feature k's
contribution is Phi(x_ik)^T w_k with Lipschitz constant C_k = sqrt(2/M) ||a||_2 ||w_k||_2 per unit of input. The filtered input moves
by at most ||H(L+Delta)-H(L)||_2 ||x_k||_2 <= eps_h ||x_k||_2. Top-1 is certified when the gap exceeds the sum of both movements.
usage: audit_gnam_cert.py DATASET [seeds]
"""

import json
import math
import os
import sys

import numpy as np
import torch

import gam
import run


def filt_bound(alpha, eps):
    return sum(abs(a) * ((2 + eps) ** h - 2**h) for h, a in enumerate(alpha))


def main():
    ds = sys.argv[1]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    g, ms = gam.load(ds)
    g = g.to("cuda")
    tr, va, te = (m[:, 0].to("cuda") for m in ms)
    c = int(g.y.max()) + 1
    deg = (
        torch.zeros(g.num_nodes)
        .index_add_(0, g.edge_index[0].cpu(), torch.ones(g.edge_index.size(1)))
        .numpy()
    )
    P = gam.laplacian_powers(g, 5)
    rows = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        m = gam.GNAMRFF(P, c, M=100, theta=2.5).cuda()
        run.fit_sgd(ds, m, g, tr, va, dict(lr=0.1, wd=5e-4))
        m.eval()
        with torch.no_grad():
            xt = m.filtered()
            pred = m(g.x, None).argmax(1)
            nodes = te.nonzero().flatten()[:200].cpu().numpy()
            xnorm = g.x.norm(dim=0)
            anorm = m.a.norm().item()
            lip = math.sqrt(2 / m.M) * anorm * m.w[:, :-1].norm(dim=1)  # D x C
            alpha = m.alphas.tolist()
            cert, epss = 0, []
            for i in nodes:
                y = int(pred[i])
                phi = torch.cos(xt[i].unsqueeze(-1) * m.a.T + m.b) * math.sqrt(
                    2.0 / m.M
                )
                contrib = (phi * m.w[:, :-1, y]).sum(1) + m.w[:, -1, y]
                k = int(contrib.argmax())
                # Removing one edge at node i changes entries by about 1/sqrt(d_i d_j); this is a lower estimate of ||Delta||_2.
                eps = 1.0 / math.sqrt(max(deg[i], 1) * max(deg.mean(), 1))
                epss.append(eps)
                move = lip[:, y] * xnorm * filt_bound(alpha, eps)
                gap = contrib[k] - contrib
                others = torch.ones_like(gap, dtype=torch.bool)
                others[k] = False
                cert += bool((gap[others] > (move[k] + move)[others]).all())
        rows.append(
            dict(
                seed=seed,
                top1_certified_b1=cert / len(nodes),
                eps_median=float(np.median(epss)),
                test_acc=gam.metric(ds, m(g.x, None)[te], g.y[te]),
            )
        )
        print(rows[-1], flush=True)
    os.makedirs(os.path.join(gam.ROOT, "results", "s03"), exist_ok=True)
    json.dump(
        rows,
        open(os.path.join(gam.ROOT, "results", "s03", f"gnam_cert_{ds}.json"), "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
