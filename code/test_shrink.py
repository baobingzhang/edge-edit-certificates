"""Gate G9 (PLAN_v7). ShrinkGAM certificates are claimed exact for every kappa; each check would catch a false claim:
a real edit set that flips a "certified" node, a certified radius smaller than the true one, a training
worst case that is not the true minimum, or kappa=0 drifting away from the frozen CertGAM code.
"""

import itertools
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

import cert
import exp
import gam
import shrink

KAPPAS = [0.0, 0.5, 3.0]


def tiny(seed, kappa, n=9, d=3, c=3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    edges = [
        (a, b) for a, b in itertools.combinations(range(n), 2) if rng.random() < 0.35
    ] or [(0, 1)]
    ei = torch.tensor(edges + [(b, a) for a, b in edges]).T
    m = shrink.ShrinkGAM(
        SimpleNamespace(x=torch.randn(n, d), edge_index=ei),
        c,
        K=2,
        theta=2.0,
        kappa=kappa,
    )
    for p, sd in ((m.W0, 1.0), (m.W1, 1.0), (m.b, 0.3)):
        torch.nn.init.normal_(p, std=sd)
    return m, n


def logits_after(S0, S1, i, nb, edit, kappa):
    """Independent recomputation: aggregate from scratch over the toggled neighbour set."""
    cur = sorted(set(nb) ^ set(edit))
    den = len(cur) + kappa
    return S0[i] + (S1[cur].sum(0) / den if den > 0 else 0.0)


def test_worst_mean():
    rng = np.random.default_rng(1)
    for _ in range(3000):
        n = int(rng.integers(3, 11))
        s = rng.normal(size=n) * rng.choice([0.1, 1, 10])
        i = int(rng.integers(n))
        nb = [u for u in range(n) if u != i and rng.random() < rng.uniform(0.1, 0.9)]
        b, base, k = (
            int(rng.integers(0, 4)),
            float(rng.normal()),
            float(rng.choice([0.0, 0.5, 1.0, 3.0, 16.0])),
        )
        cand = [(float(s[u]), int(u)) for u in np.argsort(s) if u != i]
        fast = shrink.worst_mean(
            s[nb] if nb else np.array([]), cand, set(nb), b, base, k
        )
        assert abs(fast - shrink.bruteforce_worst(s, nb, i, b, base, k)) < 1e-9
    print("PASS worst_mean equals brute force on 3000 instances (kappa in 0..16)")


def test_radii_exhaustive():
    nodes_total = breakable = 0
    for kappa in KAPPAS:
        for seed in range(40):
            m, n = tiny(seed, kappa)
            with torch.no_grad():
                S0, S1 = (t.numpy() for t in m.scores())
                full = m().numpy()
            start, dst = cert.neighbour_lists(m)
            pred, rad = shrink.radii(m, np.arange(n), [1, 2])
            for i in range(n):
                nb = dst[start[i] : start[i + 1]].tolist()
                assert np.allclose(
                    full[i], logits_after(S0, S1, i, nb, (), kappa), atol=1e-5
                ), "forward mismatch"
                others = [u for u in range(n) if u != i]
                flip = None
                for b in (1, 2):
                    for e in itertools.combinations(others, b):
                        o = logits_after(S0, S1, i, nb, e, kappa)
                        if (
                            o.argmax() != pred[i] or np.sort(o)[-2] >= o[pred[i]]
                        ):  # ties count as flips
                            flip = b
                            break
                    if flip:
                        break
                true_r = 2 if flip is None else flip - 1
                assert rad[i] == true_r, (kappa, seed, i, rad[i], true_r)
                nodes_total += 1
                breakable += flip is not None
    print(
        f"PASS radii exact on {nodes_total} nodes ({breakable} breakable) for kappa in {KAPPAS}"
    )


def test_worst_active():
    """The edit set returned for training attains the exhaustive minimum margin."""
    checked = 0
    for kappa in KAPPAS:
        for seed in range(25):
            m, n = tiny(seed, kappa)
            y = torch.as_tensor(np.random.default_rng(seed).integers(0, m.c, n))
            tr = np.arange(n)
            with torch.no_grad():
                S0, S1 = (t.numpy() for t in m.scores())
                rows = shrink.worst_active(m, tr, y.numpy(), 2, m.Psi @ m.W1)
                mg, _, _ = shrink.margins_from_rows(m, torch.as_tensor(tr), y, rows)
            start, dst = cert.neighbour_lists(m)
            for (t, c, _), got in zip(rows, mg.numpy()):
                i, yi = int(tr[t]), int(y[t])
                nb = dst[start[i] : start[i + 1]].tolist()
                s = S1[:, yi] - S1[:, c]
                truth = shrink.bruteforce_worst(
                    s, nb, i, 2, S0[i, yi] - S0[i, c], kappa
                )
                assert abs(got - truth) < 1e-4, (kappa, seed, t, c, got, truth)
                checked += 1
    print(
        f"PASS worst_active attains the exhaustive minimum on {checked} (node, class) rows"
    )


def test_kappa0_matches_frozen():
    out = {}
    for ds in ("Cora", "Texas"):
        g, ms = exp.splits(ds)
        g = g.to("cuda")
        c = int(g.y.max()) + 1
        a = cert.NeighbourGAM(g, c, K=2, theta=2.0).cuda()
        b = shrink.ShrinkGAM(g, c, K=2, theta=2.0, kappa=0.0).cuda()
        cert.fit(a, ms[0][:, 0].cuda(), g.y, 1e-2)
        b.load_state_dict(a.state_dict())
        nodes = ms[2][:, 0].nonzero().flatten().numpy()[:500]
        pa, ra = cert.radii(a, nodes, exp.BUD)
        pb, rb = shrink.radii(b, nodes, exp.BUD)
        assert (pa == pb).all() and (ra == rb).all()
        out[ds] = int(len(nodes))
    print(f"PASS kappa=0 radii identical to frozen cert.radii: {out}")


if __name__ == "__main__":
    test_worst_mean()
    test_radii_exhaustive()
    test_worst_active()
    test_kappa0_matches_frozen()
    json.dump(
        dict(status="PASS"), open(os.path.join(gam.ROOT, "results", "g9.json"), "w")
    )
    print("G9 PASS")
    sys.exit(0)
