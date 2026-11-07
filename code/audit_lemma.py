"""Printed Lemma A.1 constant versus real single-edge flips between normalised Laplacians (not synthetic perturbations)."""
import json, os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def lap(A):
    d = A.sum(1); s = np.where(d > 0, 1 / np.sqrt(np.maximum(d, 1)), 0); L = np.diag((d > 0).astype(float)) - s[:, None] * A * s[None, :]
    return L
def mono(L, h): return np.linalg.matrix_power(L, h)
rng = np.random.default_rng(0); rows = []
for R in range(1, 8):
    ratio, viol, ex = [], 0, None
    for _ in range(1500):
        n = int(rng.integers(6, 30)); A = np.triu((rng.random((n, n)) < rng.uniform(.1, .6)).astype(float), 1); A = A + A.T
        i, j = rng.choice(n, 2, replace=False); B = A.copy(); B[i, j] = B[j, i] = 1 - A[i, j]
        L, Lp = lap(A), lap(B); eps = np.linalg.norm(Lp - L, 2)
        if eps < 1e-9: continue
        true = np.linalg.norm(mono(Lp, R) - mono(L, R), 2)
        printed = .25 * (R ** 2 - 1) * ((R - 1) / (R + 1)) ** R * eps
        ours = R * 2 ** (R - 1) * eps
        viol += true > ours * (1 + 1e-9)
        ratio.append(true / printed if printed > 0 else np.inf)
    finite = [x for x in ratio if np.isfinite(x)]
    rows.append(dict(R=R, n=len(ratio), printed_zero=R == 1, min_ratio=min(finite) if finite else None, median_ratio=float(np.median(finite)) if finite else None, laplacian_bound_violations=int(viol)))
    print(rows[-1], flush=True)
assert all(r["laplacian_bound_violations"] == 0 for r in rows), "Laplacian-pair bound violated"
json.dump(rows, open(os.path.join(ROOT, "results", "audit_lemma.json"), "w"), indent=1)
