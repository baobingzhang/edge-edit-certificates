"""Gate G8 (PLAN_v5). The two-layer certificate is claimed exact, so it must agree with recomputing the whole model on
every edited graph: a certified node that some real toggle flips, or an uncertified node no toggle flips, fails.
"""

import itertools
import sys

import cert2
import gam
import numpy as np
import torch


def toggle(ei, u, v):
    e = set(map(tuple, ei.T.tolist()))
    for p in ((u, v), (v, u)):
        e.symmetric_difference_update({p})
    return torch.tensor(sorted(e), dtype=torch.long).T.reshape(2, -1)


def tiny(seed, n=8, d=3, c=3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    e = [(a, b) for a, b in itertools.combinations(range(n), 2) if rng.random() < 0.3]
    e += [
        (a, a) for a in range(n) if rng.random() < 0.2
    ]  # WebKB graphs carry self-loops
    e = e or [(0, 1)]
    ei = torch.tensor(sorted(set(e + [(b, a) for a, b in e]))).T
    m = gam.Conv2(d, c, kind="sage", hidden=5, drop=0.0).eval()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=1.0)
    return m, torch.randn(n, d), ei, n


@torch.no_grad()
def test_exhaustive():
    nodes = mism = flips = 0
    for seed in range(80):
        m, x, ei, n = tiny(seed)
        P = cert2.precompute(m, x, ei)
        assert torch.allclose(
            P["out"], m(x, ei), atol=1e-5
        ), "precompute differs from the model"
        base = m(x, ei).argmax(1)
        broken = torch.zeros(n, dtype=torch.bool)
        for u, v in itertools.combinations(range(n), 2):
            o = m(x, toggle(ei, u, v))
            broken |= o.argmax(1) != base
            for i in range(
                n
            ):  # the local formula must reproduce every logit, also for edits it says are irrelevant
                touches = (
                    u == i
                    or v == i
                    or u in cert2.neighbours(P, i).tolist()
                    or v in cert2.neighbours(P, i).tolist()
                )
                if touches:
                    a, b = (
                        (u, v)
                        if (u == i or u in cert2.neighbours(P, i).tolist())
                        else (v, u)
                    )
                    loc = cert2.toggled_logits(P, i, [a], [b])[0, 0]
                    assert torch.allclose(loc, o[i], atol=1e-4), (seed, i, u, v)
                else:
                    assert torch.allclose(P["out"][i], o[i], atol=1e-5), (
                        "irrelevant edit changed",
                        seed,
                        i,
                        u,
                        v,
                    )
        cert = cert2.certify_b1(P, range(n))
        nodes += n
        flips += int(broken.sum())
        mism += sum(int(c == bool(b)) for c, b in zip(cert, broken))
    assert mism == 0, f"certificate disagrees with exhaustive edits on {mism} nodes"
    assert flips > 0.1 * nodes, "test too easy: almost nothing flips"
    print(f"PASS exact on {nodes} nodes of 80 random graphs ({flips} breakable)")


@torch.no_grad()
def test_real(ds, k=2000):
    import exp

    g, _ = exp.splits(ds)
    g = g.to("cuda")
    torch.manual_seed(0)
    m = (
        gam.Conv2(g.x.size(1), int(g.y.max()) + 1, kind="sage", hidden=64, drop=0.0)
        .cuda()
        .eval()
    )
    P = cert2.precompute(m, g.x, g.edge_index)
    assert torch.allclose(P["out"], m(g.x, g.edge_index), atol=1e-4)
    rng = np.random.default_rng(1)
    ei = g.edge_index.cpu()
    worst = 0.0
    for _ in range(k):
        i = int(rng.integers(g.num_nodes))
        nb = cert2.neighbours(P, i).tolist() + [i]
        u = int(rng.choice(nb))
        v = (
            int(rng.choice(nb))
            if rng.random() < 0.3
            else int(rng.integers(g.num_nodes))
        )
        if u == v:
            continue
        o = m(g.x, toggle(ei, u, v).cuda())[i]
        worst = max(
            worst, float((cert2.toggled_logits(P, i, [u], [v])[0, 0] - o).abs().max())
        )
    assert worst < 1e-4, f"{ds}: max abs logit difference {worst}"
    print(f"PASS {ds}: {k} random toggles, max abs logit difference {worst:.2e}")


if __name__ == "__main__":
    test_exhaustive()
    for ds in sys.argv[1:] or ["Texas", "Cora", "amazon-ratings"]:
        test_real(ds, k=2000 if ds != "amazon-ratings" else 300)
