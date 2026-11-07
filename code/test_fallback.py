"""The composite's guarantee is new, so check it directly: no real edit within budget b may change a certified output,
including edits that raise the radius and thereby switch the composite back to the graph model."""
import itertools, numpy as np, torch, cert, fallback, test_cert


def rad_of(s0, s1, cur, n, i, B, c):
    """Exact radius by brute force from a given neighbour set."""
    def pred(nb): return int((s0[i] + (s1[sorted(nb)].mean(0) if nb else 0)).argmax())
    y0 = pred(cur); others = [u for u in range(n) if u != i]
    for b in range(1, B + 1):
        for e in itertools.combinations(others, b):
            if pred(set(cur) ^ set(e)) != y0:
                return y0, b - 1
    return y0, B


def main():
    broken, cases, certified = 0, 0, 0
    for seed in range(40):
        m, g, n, _ = test_cert.tiny(seed + 500, n=8, d=3, c=3)
        with torch.no_grad():
            S0, S1 = (t.numpy() for t in m.scores())
        ego = S0.argmax(1)  # the graph-free model here: same ego scores, independent of edges
        start, dst = cert.neighbour_lists(m)
        for i in range(n):
            nb = set(dst[start[i]:start[i + 1]].tolist())
            for t in (1, 2):
                B = 2 * t + 1
                y0, r0 = rad_of(S0, S1, nb, n, i, B, m.c)
                out0 = y0 if r0 >= t else ego[i]
                for b in range(1, t + 1):
                    if not (ego[i] == y0 or r0 >= t + b):
                        continue
                    certified += 1
                    others = [u for u in range(n) if u != i]
                    for k in range(1, b + 1):
                        for e in itertools.combinations(others, k):
                            cur = nb ^ set(e)
                            y1, r1 = rad_of(S0, S1, cur, n, i, t, m.c)
                            cases += 1
                            broken += (y1 if r1 >= t else ego[i]) != out0
    # The production code path must agree with the criterion exercised above.
    pg, r, pe, y = np.array([0, 0, 1, 1]), np.array([0, 3, 1, 3]), np.array([0, 1, 0, 0]), np.array([0, 0, 0, 1])
    res = fallback.evaluate(pg, r, pe, y, 1, [1, 2])
    use = r >= 1; out = np.where(use, pg, pe); cert_ = (pe == pg) | (r >= 2)
    assert abs(res["clean"] - (out == y).mean()) < 1e-12 and abs(res["c1"] - ((out == y) & cert_).mean()) < 1e-12 and "c2" not in res, res
    assert broken == 0, f"composite certificate broken {broken}/{cases}"
    assert certified > 50, "test did not exercise the certificate"
    print(f"PASS composite certificate held on {cases} real edits across {certified} certified cases")


if __name__ == "__main__":
    main()
