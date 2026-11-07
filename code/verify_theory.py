"""Numerical checks of T1 and T2. A single violated instance falsifies the theorem, so the script fails loudly."""
import json, math, os
import numpy as np

rng = np.random.default_rng(0)
out = {"T1": [], "T2": []}

def H(alpha, L):
    A, P = np.zeros_like(L), np.eye(len(L))
    for a in alpha:
        A, P = A + a * P, P @ L
    return A

def bound_t1(alpha, eps):
    return sum(abs(a) * ((2 + eps) ** h - 2 ** h) for h, a in enumerate(alpha))

worst = 0.0
for _ in range(2000):
    n = rng.integers(4, 30)
    A = (rng.random((n, n)) < rng.uniform(0.1, 0.7)).astype(float); A = np.triu(A, 1); A = A + A.T
    d = A.sum(1); d[d == 0] = 1
    L = np.eye(n) - A / np.sqrt(np.outer(d, d))
    E = rng.standard_normal((n, n)); E = E + E.T
    eps = 10 ** rng.uniform(-4, 0)
    dL = eps * E / np.linalg.norm(E, 2)
    alpha = rng.standard_normal(rng.integers(1, 9))
    lhs, rhs = np.linalg.norm(H(alpha, L + dL) - H(alpha, L), 2), bound_t1(alpha, eps)
    assert lhs <= rhs * (1 + 1e-9) + 1e-12, ("T1 violated", lhs, rhs)
    worst = max(worst, lhs / max(rhs, 1e-300))
out["T1_random_max_ratio"] = worst

# Tightness: bipartite path graph, perturbation along the eigenvector with eigenvalue 2, alpha >= 0.
n = 20
A = np.zeros((n, n))
for i in range(n - 1):
    A[i, i + 1] = A[i + 1, i] = 1
d = A.sum(1)
L = np.eye(n) - A / np.sqrt(np.outer(d, d))
w, V = np.linalg.eigh(L)
assert abs(w[-1] - 2) < 1e-9
v = V[:, -1:]
for R in range(1, 8):
    alpha = rng.random(R + 1)
    eps = 1e-2
    lhs = np.linalg.norm(H(alpha, L + eps * v @ v.T) - H(alpha, L), 2)
    rhs = bound_t1(alpha, eps)
    assert abs(lhs - rhs) < 1e-8 * rhs, ("T1 not tight", lhs, rhs)
    out["T1"].append(dict(R=R, lhs=lhs, rhs=rhs))

# T2: explicit Taylor polynomial of degree < r stays within sqrt(2)*||w||*(a_max rho)^r / r!.
for M in (20, 100, 500):
    for theta in (0.5, 1.0, 2.2, 8.0):
        for rho in (0.05, 0.2, 1.0, 3.0):
            for r in (1, 2, 3, 4, 6):
                a, b = rng.standard_normal(M) / theta, 2 * math.pi * rng.random(M)
                w = rng.standard_normal(M)
                x = np.linspace(-rho, rho, 801)
                f = np.sqrt(2 / M) * np.cos(np.outer(x, a) + b) @ w
                p = np.zeros_like(x)
                for j in range(r):
                    dj = math.cos(j * math.pi / 2) * np.cos(b) - math.sin(j * math.pi / 2) * np.sin(b)
                    p += np.sqrt(2 / M) * ((a ** j * dj) @ w) * x ** j / math.factorial(j)
                err = np.abs(f - p).max()
                bd = math.sqrt(2) * np.linalg.norm(w) * (np.abs(a).max() * rho) ** r / math.factorial(r)
                assert err <= bd * (1 + 1e-9) + 1e-12, ("T2 violated", M, theta, rho, r, err, bd)
                out["T2"].append(dict(M=M, theta=theta, rho=rho, r=r, err=float(err), bound=float(bd)))
path = os.path.join(os.path.dirname(__file__), "..", "results", "verify_theory.json")
json.dump(out, open(path, "w"), indent=0)
r = [t["err"] / t["bound"] for t in out["T2"]]
print("PASS T1 random max lhs/rhs", round(worst, 4), "tight on path graph; T2 cases", len(r), "max err/bound", round(max(r), 4))
