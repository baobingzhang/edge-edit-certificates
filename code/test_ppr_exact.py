"""Gate G10b (PLAN_v8b). exact_b1 claims the exact minimum margin of pi-PPNP over every single undirected toggle.
A wrong Woodbury term, a wrong sign for deletions, or a missed toggle would make it disagree with recomputing the
propagation from scratch on each perturbed graph, which is what these checks do.
"""

import itertools
import json
import os

import numpy as np
import torch

import exp
import gam
import ppr
from exp10 import ppr_rows_after_toggle

DEV = "cuda"


def brute(A, H, alpha, t, y):
    """Min over all toggles of min_c margin (pi'_t H)_y - (pi'_t H)_c, by dense solves."""
    n = A.shape[0]
    best = np.inf
    for u, v in itertools.combinations(range(n), 2):
        B = A.copy()
        B[u, v] = B[v, u] = 1 - B[u, v]
        Bh = B + np.eye(n)
        Mt = np.linalg.solve(
            (np.eye(n) - alpha * Bh / Bh.sum(1, keepdims=True)).T, np.eye(n)[t]
        )
        z = (1 - alpha) * Mt @ H
        best = min(best, min(z[y] - z[c] for c in range(H.shape[1]) if c != y))
    return best


def test_tiny():
    rng = np.random.default_rng(3)
    k = 0
    for _ in range(60):
        n = int(rng.integers(3, 9))
        alpha = float(rng.choice([0.1, 0.5, 0.85]))
        pairs = [
            (a, b) for a, b in itertools.combinations(range(n), 2) if rng.random() < 0.4
        ]
        A = np.zeros((n, n))
        for a, b in pairs:
            A[a, b] = A[b, a] = 1
        ei = (
            torch.tensor(pairs + [(b, a) for a, b in pairs], dtype=torch.long).T
            if pairs
            else torch.zeros(2, 0, dtype=torch.long)
        )
        H = rng.normal(size=(n, 3)) * 2
        G = ppr.Graph(ei, n, DEV)
        pred, worst, _, _ = ppr.exact_b1(
            G, torch.as_tensor(H, device=DEV), alpha, list(range(n)), 3
        )
        for t in range(n):
            ref = brute(A, H, alpha, t, int(pred[t]))
            assert abs(float(worst[t]) - ref) < 1e-9, (
                n,
                alpha,
                t,
                float(worst[t]),
                ref,
            )
            k += 1
    print(f"PASS exact_b1 equals brute force on {k} (graph, node) cases, n<=8")
    return k


def test_texas():
    g, _ = exp.splits("Texas")
    n = g.num_nodes
    G = ppr.Graph(g.edge_index.to(DEV), n, DEV)
    A = np.zeros((n, n))
    A[G.row.cpu().numpy(), G.col.cpu().numpy()] = 1
    rng = np.random.default_rng(4)
    H = rng.normal(size=(n, 5))
    worst_diff, k = 0.0, 0
    for alpha in (0.5, 0.85):
        targets = rng.choice(n, 6, replace=False).tolist()
        pred, worst, _, _ = ppr.exact_b1(
            G, torch.as_tensor(H, device=DEV), alpha, targets, 5
        )
        for q, t in enumerate(targets):
            ref = brute(A, H, alpha, t, int(pred[t]))
            worst_diff = max(worst_diff, abs(float(worst[q]) - ref))
            k += 1
    assert worst_diff < 1e-8, worst_diff
    print(
        f"PASS exact_b1 equals exhaustive recomputation of all {n * (n - 1) // 2} toggles on Texas, {k} targets (max diff {worst_diff:.1e})"
    )
    return k, worst_diff


def test_argmin_cora():
    g, _ = exp.splits("Cora")
    n = g.num_nodes
    G = ppr.Graph(g.edge_index.to(DEV), n, DEV)
    H = (
        torch.randn(
            n,
            7,
            device=DEV,
            dtype=torch.float64,
            generator=torch.Generator(device=DEV).manual_seed(5),
        )
        * 3
    )
    nodes = list(range(0, n, 54))[:50]
    alpha = 0.85
    pred, worst, arg, _ = ppr.exact_b1(G, H, alpha, nodes, 7)
    rows = ppr_rows_after_toggle(
        G, arg, torch.as_tensor(nodes, device=DEV), alpha, tol=1e-13
    )
    Z = rows @ H
    d = 0.0
    for q, t in enumerate(nodes):
        y = int(pred[t])
        m = min(float(Z[q, y] - Z[q, c]) for c in range(7) if c != y)
        d = max(d, abs(m - float(worst[q])))
    assert d < 1e-8, d
    print(
        f"PASS argmin toggles recomputed from scratch reproduce the exact worst margin on Cora ({len(nodes)} nodes, max diff {d:.1e})"
    )
    return len(nodes), d


if __name__ == "__main__":
    a = test_tiny()
    b, bd = test_texas()
    c, cd = test_argmin_cora()
    json.dump(
        dict(
            status="PASS",
            tiny_cases=a,
            texas_targets=b,
            texas_max_diff=bd,
            cora_nodes=c,
            cora_max_diff=cd,
        ),
        open(os.path.join(gam.ROOT, "results", "g10b.json"), "w"),
    )
    print("G10b PASS")
