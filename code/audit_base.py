"""G0: numerical audit of claims in the G-NAMRFF paper. Writes results/audit_base.json.

A. Lemma A.1 / Thm 4.2 constant: compare the claimed bound to the true filter perturbation.
B. Parameter counts: paper formula vs. actual implementation, against a 2-layer GCN.
C. Monomial coefficients: size of L^h shows alpha_h is not comparable across h.
D. Effective rank of the shared RFF basis on the actual range of filtered features.
"""

import json
import os
import sys

import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = {}


def claimed_factor(R):
    """K/||alpha||_1 as printed in Lemma A.1: (1/4)(R^2-1)((R-1)/(R+1))^R."""
    return 0.25 * (R**2 - 1) * ((R - 1) / (R + 1)) ** R


def filt(alpha, L):
    H, P = np.zeros_like(L), np.eye(L.shape[0])
    for a in alpha:
        H += a * P
        P = P @ L
    return H


# A. Bipartite path graph: normalized Laplacian has eigenvalue exactly 2.
n = 40
A = np.zeros((n, n))
for i in range(n - 1):
    A[i, i + 1] = A[i + 1, i] = 1
d = A.sum(1)
L = np.eye(n) - A / np.sqrt(np.outer(d, d))
rng = np.random.default_rng(0)
E = rng.standard_normal((n, n))
E = (E + E.T) / 2
eps = 1e-3
dL = eps * E / np.linalg.norm(E, 2)
rows = []
for R in range(1, 8):
    alpha = np.zeros(R + 1)
    alpha[R] = 1.0  # single monomial so ||alpha||_1 = 1
    actual = np.linalg.norm(filt(alpha, L + dL) - filt(alpha, L), 2)
    rows.append(
        dict(
            R=R,
            claimed=claimed_factor(R) * eps,
            actual=float(actual),
            ratio=float(actual / max(claimed_factor(R) * eps, 1e-300)),
        )
    )
out["lemma"] = rows

# B/C/D on the Planetoid graphs.
out["params"], out["L_power_norm"], out["rff_rank"] = {}, {}, {}
for name, M, R_code, theta in [
    ("Cora", 100, 7, 2.2),
    ("CiteSeer", 100, 5, 2.5),
    ("PubMed", 100, 5, 3.8),
]:
    ds = Planetoid(
        os.path.join(ROOT, "data", "planetoid"), name, transform=NormalizeFeatures()
    )
    g = ds[0]
    D, C = ds.num_features, ds.num_classes
    out["params"][name] = dict(
        D=D,
        C=C,
        paper_formula=D * M + R_code,
        code_actual=(M + 1) * C * D + R_code,
        gcn16=(D * 16 + 16) + (16 * C + C),
        gcn64=(D * 64 + 64) + (64 * C + C),
    )
    ei, ew = get_laplacian(g.edge_index, normalization="sym", num_nodes=g.num_nodes)
    Ls = to_scipy_sparse_matrix(ei, ew, g.num_nodes).tocsr()
    X = g.x.numpy()

    # Size of each monomial's response on the real features.
    Pk, norms = X.copy(), []
    for h in range(R_code):
        norms.append(float(np.linalg.norm(Pk) / np.linalg.norm(X)))
        Pk = Ls @ Pk
    out["L_power_norm"][name] = norms

    # D. Rank of Phi on an input range. Use a range matching filtered features with alpha ~ 1.
    torch.manual_seed(0)
    a = (torch.randn(M, 1) / theta).numpy()
    b = (2 * np.pi * torch.rand(M)).numpy()
    q = np.quantile(np.abs(X[X != 0]), [0.5, 0.99, 1.0])
    res = {"nonzero_abs_quantiles_50_99_100": [float(v) for v in q]}
    for label, hi in [
        ("data_max", float(np.abs(X).max())),
        ("x10", 10 * float(np.abs(X).max())),
    ]:
        grid = np.linspace(-hi, hi, 2001)[:, None]
        Phi = np.cos(grid @ a.T + b) * np.sqrt(2 / M)
        Phi = np.column_stack([Phi, np.ones(len(grid))])
        s = np.linalg.svd(Phi, compute_uv=False)
        e = np.cumsum(s**2) / np.sum(s**2)
        res[label] = dict(
            hi=hi,
            rank_999=int(np.searchsorted(e, 0.999) + 1),
            rank_999999=int(np.searchsorted(e, 0.999999) + 1),
            top5=[float(v) for v in e[:5]],
        )
    out["rff_rank"][name] = res

json.dump(out, open(os.path.join(ROOT, "results", "audit_base.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
sys.exit(0)
