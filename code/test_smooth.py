"""Baseline certificates must be valid, or comparing against them is meaningless."""
import numpy as np
import smooth


def main():
    rng = np.random.default_rng(0)
    # Exhaustive check on few bits: enumerate all samples, take the true Neyman-Pearson optimum.
    for _ in range(200):
        pa, pd = rng.uniform(0.01, 0.4), rng.uniform(0.1, 0.8)
        ra, rd = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        if ra + rd == 0:
            continue
        pA = rng.uniform(0.5, 0.999)
        cells = []
        n = ra + rd
        for bits in range(2**n):
            z = [(bits >> i) & 1 for i in range(n)]
            pc = np.prod([pa if z[i] else 1 - pa for i in range(ra)] + [1 - pd if z[i] else pd for i in range(ra, n)])
            px = np.prod([1 - pd if z[i] else pd for i in range(ra)] + [pa if z[i] else 1 - pa for i in range(ra, n)])
            cells.append((px / pc, pc, px))
        cells.sort(key=lambda t: t[0])
        left, out = pA, 0.0
        for _, c, x in cells:
            take = min(left, c)
            out += x * take / c
            left -= take
        assert abs(out - smooth.worst_pA(pA, pa, pd, ra, rd)) < 1e-9
    # Monotone in budget: more edits can never certify more.
    t = smooth.threshold_table(0.01, 0.6, [1, 2, 3, 5])
    assert t[1] <= t[2] <= t[3] <= t[5], t
    # Clopper-Pearson must cover: empirical failure rate at most alpha (with slack).
    fails = sum(smooth.lower_pA(rng.binomial(1000, 0.9), 1000, 0.05) > 0.9 for _ in range(3000))
    assert fails / 3000 < 0.07, fails
    print("PASS smoothing matches exhaustive Neyman-Pearson; thresholds", {k: round(v, 4) for k, v in t.items()})


main()
