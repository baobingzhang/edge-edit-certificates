"""Certificates are claimed to be exact and training convex. Each test below would catch a false claim:
a real edit beating a "certified" node, a bound looser than brute force, or a non-convex objective."""
import itertools
import numpy as np
import torch
from types import SimpleNamespace
import cert


def tiny(seed, n=9, d=3, c=3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    edges = [(a, b) for a, b in itertools.combinations(range(n), 2) if rng.random() < 0.35]
    if not edges:
        edges = [(0, 1)]
    ei = torch.tensor(edges + [(b, a) for a, b in edges]).T
    g = SimpleNamespace(x=torch.randn(n, d), edge_index=ei)
    m = cert.NeighbourGAM(g, c, K=2, theta=2.0)
    torch.nn.init.normal_(m.W0, std=1.0), torch.nn.init.normal_(m.W1, std=1.0), torch.nn.init.normal_(m.b, std=0.3)
    return m, g, n, edges


def logits_after(m, i, edit, nb0):
    """Recompute logit_i from scratch after toggling edges (i,u) in `edit`: an independent code path."""
    with torch.no_grad():
        S0, S1 = m.scores()
    cur = sorted(set(nb0) ^ set(edit))
    return (S0[i] + (S1[cur].mean(0) if cur else 0.0)).numpy()


def test_worst_mean():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        n = rng.integers(3, 11)
        s = rng.normal(size=n) * rng.choice([0.1, 1, 10])
        i = int(rng.integers(n))
        nb = [u for u in range(n) if u != i and rng.random() < rng.uniform(0.1, 0.9)]
        b, base = int(rng.integers(0, 5)), float(rng.normal())
        cand = [(float(s[u]), int(u)) for u in np.argsort(s) if u != i]
        fast = cert.worst_mean(s[nb] if nb else np.array([]), cand, set(nb), b, base)
        assert abs(fast - cert.bruteforce_worst(s, nb, i, b, base)) < 1e-9
    print("PASS worst_mean equals brute force on 3000 instances")


def test_radii_and_top1():
    broke_pred, broke_top1, tight_pred, tight_top1, cases = 0, 0, 0, 0, 0
    for seed in range(60):
        m, g, n, _ = tiny(seed)
        nodes = np.arange(n)
        pred, rad = cert.radii(m, nodes, [1, 2, 3])
        top = cert.top1_stable(m, nodes, [1, 2, 3])
        start, dst = cert.neighbour_lists(m)
        for i in nodes:
            nb = dst[start[i]:start[i + 1]].tolist()
            others = [u for u in range(n) if u != i]
            y = int(pred[i])
            with torch.no_grad():
                E, F1 = cert.attribution_scores(m, y)
            ego, f1 = E[i].numpy(), F1.numpy()
            base = (ego + (f1[nb].mean(0) if nb else 0)).argmax()
            flipped_at, top_changed_at = None, None
            for b in (1, 2, 3):
                for edit in itertools.combinations(others, b):
                    if flipped_at is None and logits_after(m, i, edit, nb).argmax() != y:
                        flipped_at = b
                    cur = sorted(set(nb) ^ set(edit))
                    if top_changed_at is None and (ego + (f1[cur].mean(0) if cur else 0)).argmax() != base:
                        top_changed_at = b
                if flipped_at and top_changed_at:
                    break
            cases += 1
            # Soundness: no real edit within the certified budget changes the outcome.
            broke_pred += flipped_at is not None and flipped_at <= rad[i]
            broke_top1 += top_changed_at is not None and top_changed_at <= top[i]
            # Exactness: the first radius not certified must be broken by some real edit.
            if rad[i] < 3:
                tight_pred += flipped_at != rad[i] + 1
            # Thm 2 claims top-1 stability is exact too, so the first uncertified budget must be breakable.
            if top[i] < 3:
                tight_top1 += top_changed_at != top[i] + 1
    assert broke_pred == 0, f"prediction certificate broken {broke_pred}"
    assert broke_top1 == 0, f"attribution certificate broken {broke_top1}"
    assert tight_pred == 0, f"prediction certificate not exact {tight_pred}"
    assert tight_top1 == 0, f"top-1 certificate not exact {tight_top1}"
    print(f"PASS certificates sound and exact against exhaustive edits on {cases} nodes")


def test_convex():
    viol = 0
    for seed in range(20):
        m, g, n, _ = tiny(seed + 100, n=15, d=3, c=3)
        y = torch.randint(0, 3, (n,))
        tr = torch.arange(8)

        def obj(p):
            for t, v in zip((m.W0, m.W1, m.b), p):
                t.data.copy_(v)
            return cert.robust_objective(m, tr, y, 2, 1e-3)[0].item()

        for _ in range(20):
            a = [torch.randn_like(t) for t in (m.W0, m.W1, m.b)]
            b = [torch.randn_like(t) for t in (m.W0, m.W1, m.b)]
            t = float(torch.rand(1))
            mid = [t * u + (1 - t) * v for u, v in zip(a, b)]
            viol += obj(mid) > t * obj(a) + (1 - t) * obj(b) + 1e-5
    assert viol == 0, f"robust objective not convex: {viol}"
    print("PASS robust objective satisfies the convexity inequality on 400 random segments")


if __name__ == "__main__":
    test_worst_mean()
    test_radii_and_top1()
    test_convex()
