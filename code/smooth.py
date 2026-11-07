"""Sparse randomized smoothing for edge edits, following the regions argument of Bojchevski et al. (2020).

An edge pair flips 0->1 with prob pa and 1->0 with prob pd. For an adversary with ra additions and rd
deletions, only the edited bits change the likelihood ratio, so Neyman-Pearson is solved by enumerating
(u, v) = number of added/deleted bits that are 1 in the sample.
"""
import math

import numpy as np
from scipy.stats import beta, binom


def worst_pA(pA, pa, pd, ra, rd):
    """Smallest probability of the top class under the adversarial graph, given probability pA under the clean one."""
    cells = []
    for u in range(ra + 1):
        for v in range(rd + 1):
            clean = binom.pmf(u, ra, pa) * binom.pmf(v, rd, 1 - pd)
            adv = binom.pmf(u, ra, 1 - pd) * binom.pmf(v, rd, pa)
            cells.append((adv / max(clean, 1e-300), clean, adv))
    cells.sort(key=lambda t: t[0])
    left, out = pA, 0.0
    for _, clean, adv in cells:
        take = min(left, clean)
        if clean > 0:
            out += adv * take / clean
        left -= take
        if left <= 1e-15:
            break
    return out


def certified(pA, pa, pd, b):
    """True if the top class stays above 1/2 for every split of b edits into additions and deletions."""
    return all(worst_pA(pA, pa, pd, ra, b - ra) > 0.5 for ra in range(b + 1))


def lower_pA(k, n, alpha=0.01):
    """One-sided Clopper-Pearson lower confidence bound."""
    return 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))


def threshold_table(pa, pd, budgets, grid=2001):
    """Smallest pA certifying each budget, found by bisection since worst_pA is monotone in pA."""
    out = {}
    for b in budgets:
        lo, hi = 0.5, 1.0
        if not certified(1 - 1e-12, pa, pd, b):
            out[b] = math.inf
            continue
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if certified(mid, pa, pd, b) else (mid, hi)
        out[b] = hi
    return out


def sample_edges(ei_und, n, pa, pd, rng):
    """Undirected edge list after random flips. Insertions are sampled sparsely; rare duplicates of existing edges are kept, so such an edge counts twice in a mean."""
    keep = rng.random(ei_und.shape[1]) > pd
    kept = ei_und[:, keep]
    pairs = n * (n - 1) // 2
    m = rng.binomial(pairs, pa)
    if m:
        a = rng.integers(0, n, m)
        b = rng.integers(0, n, m)
        ok = a != b
        new = np.stack([np.minimum(a[ok], b[ok]), np.maximum(a[ok], b[ok])])
        kept = np.concatenate([kept, new], axis=1)
    both = np.concatenate([kept, kept[::-1]], axis=1)
    return both
