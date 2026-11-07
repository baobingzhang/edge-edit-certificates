"""Numerical check of Proposition 3 (PLAN_v7): one edit changes the shrunk neighbour term by at most
2R/(d+kappa+1) for an insertion and 2R/(d+kappa-1) for a deletion (d+kappa>1); R/(1+kappa) if d=0."""
import json, os
import numpy as np
import gam

rng = np.random.default_rng(0)
worst_ratio, n = 0.0, 0
for _ in range(200000):
    d, k, R = int(rng.integers(0, 12)), float(rng.choice([0, 0.25, 0.5, 1, 2, 4, 16])), float(rng.uniform(0.1, 5))
    s = rng.uniform(-R, R, d)
    if rng.random() < 0.3: s = np.sign(rng.normal(size=d)) * R  # extreme points
    base = s.sum() / (d + k) if d + k > 0 else 0.0
    u = float(rng.choice([-R, R, rng.uniform(-R, R)]))
    ins = (s.sum() + u) / (d + 1 + k)
    bound = R / (1 + k) if d == 0 else 2 * R / (d + k + 1)
    worst_ratio = max(worst_ratio, abs(ins - base) / bound); n += 1
    if d >= 1 and d + k > 1:
        j = int(rng.integers(d)); dl = (s.sum() - s[j]) / (d - 1 + k)
        worst_ratio = max(worst_ratio, abs(dl - base) / (2 * R / (d + k - 1))); n += 1
ok = worst_ratio <= 1 + 1e-9
json.dump(dict(checked=n, max_ratio=float(worst_ratio), passed=bool(ok)), open(os.path.join(gam.ROOT, "results", "verify_shrink.json"), "w"))
print(n, worst_ratio, "PASS" if ok else "FAIL")
