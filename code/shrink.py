"""E8 (PLAN_v7): ShrinkGAM, the neighbour-additive GAM with a pseudo-count kappa in the aggregation denominator.

logit_i = b + S0_i + sum_{j in N(i)} S1_j / (|N(i)| + kappa).  kappa = 0 is CertGAM (cert.NeighbourGAM).
For fixed numbers of deletions p and insertions q the denominator is fixed, so deleting the p largest and
inserting the q smallest scores is still the exact worst case. cert.py is frozen, so the few functions whose
denominator changes are re-implemented here with the extra kappa term and otherwise mirror cert.py.
"""

import cert
import numpy as np
import torch


class ShrinkGAM(cert.NeighbourGAM):
    def __init__(self, g, c, K=2, theta=2.0, clip=5.0, kappa=0.0):
        super().__init__(g, c, K=K, theta=theta, clip=clip)
        self.set_kappa(kappa)

    def set_kappa(self, kappa):
        self.kappa = float(kappa)
        den = self.deg.clamp_min(1) if self.kappa == 0 else self.deg + self.kappa
        w = (1 / den)[self.ei[0]]
        A = torch.sparse_coo_tensor(self.ei, w, (self.n, self.n)).coalesce()
        self.Mean = torch.sparse.mm(A, self.Psi)


def agg(s, kappa):
    """Aggregated neighbour score for a set of scores s (numpy, any length)."""
    den = len(s) + kappa
    return float(np.sum(s)) / den if den > 0 else 0.0


def worst_mean(s_nb, s_all_sorted, nb_set, budget, base, kappa):
    d = len(s_nb)
    s_desc = np.sort(s_nb)[::-1]
    pre_del = np.concatenate([[0.0], np.cumsum(s_desc)])
    add, picked = [0.0], 0
    for sc, u in s_all_sorted:
        if picked >= budget:
            break
        if u in nb_set:
            continue
        add.append(add[-1] + sc)
        picked += 1
    total, best = pre_del[-1], np.inf
    for p in range(0, min(budget, d) + 1):
        for q in range(0, min(budget - p, picked) + 1):
            den = d - p + q + kappa
            mean = (total - pre_del[p] + add[q]) / den if den > 0 else 0.0
            best = min(best, base + mean)
    return best


def radii(m, nodes, budgets):
    with torch.no_grad():
        S0, S1 = (t.detach().cpu().numpy() for t in m.scores())
        logits = m().detach().cpu().numpy()
    pred = logits.argmax(1)
    start, dst = cert.neighbour_lists(m)
    out = np.zeros(len(nodes), dtype=int)
    pair_sorted = {}
    for t, i in enumerate(nodes):
        y = pred[i]
        nb = dst[start[i] : start[i + 1]]
        nbs = set(nb.tolist()) | {int(i)}
        r = 0
        for b in budgets:
            ok = True
            for c in range(m.c):
                if c == y:
                    continue
                if (y, c) not in pair_sorted:
                    s = S1[:, y] - S1[:, c]
                    idx = np.argsort(s, kind="stable")
                    pair_sorted[(y, c)] = (s, s[idx], idx)
                s, ss, idx = pair_sorted[(y, c)]
                if (
                    worst_mean(
                        s[nb],
                        zip(ss.tolist(), idx.tolist()),
                        nbs,
                        b,
                        S0[i, y] - S0[i, c],
                        m.kappa,
                    )
                    <= 0
                ):
                    ok = False
                    break
            if not ok:
                break
            r = b
        out[t] = r
    return pred[nodes], out


def worst_active(m, tr, y, B, S1):
    S1n = S1.detach().cpu().numpy()
    start, dst = cert.neighbour_lists(m)
    rows, orders = [], {}
    for t, i in enumerate(tr):
        yi = int(y[i])
        nb = dst[start[i] : start[i + 1]]
        nbs = set(nb.tolist()) | {int(i)}
        for c in range(m.c):
            if c == yi:
                continue
            if (yi, c) not in orders:
                s = S1n[:, yi] - S1n[:, c]
                orders[(yi, c)] = (s, np.argsort(s, kind="stable"))
            s, idx = orders[(yi, c)]
            del_order = nb[np.argsort(-s[nb], kind="stable")]
            pre = np.concatenate([[0.0], np.cumsum(s[del_order])])
            added, add_s = [], [0.0]
            for u in idx:
                if len(added) >= B:
                    break
                if int(u) in nbs:
                    continue
                added.append(int(u))
                add_s.append(add_s[-1] + s[u])
            best, arg, d = np.inf, (0, 0), len(nb)
            for p in range(min(B, d) + 1):
                for q in range(min(B - p, len(added)) + 1):
                    den = d - p + q + m.kappa
                    v = (pre[-1] - pre[p] + add_s[q]) / den if den > 0 else 0.0
                    if v < best:
                        best, arg = v, (p, q)
            p, q = arg
            rows.append(
                (
                    t,
                    c,
                    np.concatenate(
                        [del_order[p:], np.array(added[:q], dtype=np.int64)]
                    ).astype(np.int64),
                )
            )
    return rows


def margins_from_rows(m, tr_t, y, rows):
    """Margins of (train node, wrong class) rows whose neighbour sets are given, with the kappa denominator."""
    S0, S1 = m.Psi[tr_t] @ m.W0 + m.b, m.Psi @ m.W1
    dev = m.Psi.device
    ir, nd, wt = [], [], []
    for r, (_, _, st) in enumerate(rows):
        if len(st):
            ir.append(np.full(len(st), r))
            nd.append(st)
            wt.append(np.full(len(st), 1.0 / (len(st) + m.kappa), dtype=np.float32))
    ts = torch.as_tensor([a for a, _, _ in rows], device=dev)
    cs = torch.as_tensor([b for _, b, _ in rows], device=dev)
    ys = y[tr_t][ts]
    mean = torch.zeros(len(rows), device=dev)
    if ir:
        I = torch.as_tensor(np.concatenate(ir), device=dev)
        N = torch.as_tensor(np.concatenate(nd), device=dev)
        W = torch.as_tensor(np.concatenate(wt), device=dev)
        mean = mean.index_add(0, I, (S1[N, ys[I]] - S1[N, cs[I]]) * W)
    return S0[ts, ys] - S0[ts, cs] + mean, ts, cs


def _pool_loss(m, tr_len, mg, ts):
    # log(1 + sum_c exp(-m_c)); slot c-1 stays 0 and supplies the "1 +".
    pos, seen = [], {}
    for t in ts.tolist():
        pos.append(seen.get(t, 0))
        seen[t] = pos[-1] + 1
    acc = torch.zeros(tr_len, m.c, device=mg.device).index_put(
        (ts, torch.as_tensor(pos, device=mg.device)), -mg
    )
    return torch.logsumexp(acc, 1).mean()


def robust_objective(m, tr_t, y, B, lam):
    rows = worst_active(m, tr_t.cpu().numpy(), y.cpu().numpy(), B, m.Psi @ m.W1)
    mg, ts, _ = margins_from_rows(m, tr_t, y, rows)
    return _pool_loss(m, len(tr_t), mg, ts) + lam * m.penalty(), rows


def reset(m):
    with torch.no_grad():
        m.W0.zero_(), m.W1.zero_(), m.b.zero_()


def fit_robust(m, tr_idx, y, lam, B, lr, steps=200, batch=512):
    """Same schedule as exp.fit_robust (ridge warm start, minibatch Adam, average of the second half)."""
    reset(m)
    mask = torch.zeros(m.n, dtype=torch.bool, device=m.Psi.device)
    mask[tr_idx] = True
    cert.fit(m, mask, y, lam)
    opt = torch.optim.Adam([m.W0, m.W1, m.b], lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(0)
    avg = None
    for it in range(steps):
        for gr in opt.param_groups:
            gr["lr"] = lr / np.sqrt(1 + it / 20)
        sub = (
            tr_idx
            if len(tr_idx) <= batch
            else tr_idx[
                torch.randperm(len(tr_idx), generator=gen)[:batch].to(tr_idx.device)
            ]
        )
        opt.zero_grad()
        obj, _ = robust_objective(m, sub, y, B, lam)
        obj.backward()
        opt.step()
        if it >= steps // 2:
            cur = [t.detach().clone() for t in (m.W0, m.W1, m.b)]
            k = it - steps // 2
            avg = (
                cur
                if avg is None
                else [(a * k + c) / (k + 1) for a, c in zip(avg, cur)]
            )
    with torch.no_grad():
        for t, a in zip((m.W0, m.W1, m.b), avg):
            t.copy_(a)


def fit_cutting(m, tr_t, y, lam, B, rounds=12, tau=0.02):
    """Mirror of alt_solver.fit_cutting with the kappa denominator."""
    reset(m)
    mask = torch.zeros(m.n, dtype=torch.bool, device=m.Psi.device)
    mask[tr_t] = True
    cert.fit(m, mask, y, lam)
    tr, ynp = tr_t.cpu().numpy(), y.cpu().numpy()
    pool, seen, best, state = [], set(), np.inf, None
    for _ in range(rounds):
        with torch.no_grad():
            S1 = m.Psi @ m.W1
        for t, c, st in worst_active(m, tr, ynp, B, S1):
            k = (t, c, tuple(sorted(st.tolist())))
            if k not in seen:
                seen.add(k)
                pool.append((t, c, st))
        keys = {}
        gid = torch.as_tensor(
            [keys.setdefault((t, c), len(keys)) for t, c, _ in pool],
            device=m.Psi.device,
        )
        G = len(keys)
        kt = torch.as_tensor([k[0] for k in keys], device=m.Psi.device)
        opt = torch.optim.LBFGS(
            [m.W0, m.W1, m.b],
            lr=1.0,
            max_iter=60,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            mg, _, _ = margins_from_rows(m, tr_t, y, pool)
            big = torch.zeros(G, device=mg.device).scatter_reduce(
                0, gid, -mg / tau, "amax", include_self=False
            )
            sm = torch.zeros(G, device=mg.device).index_add(
                0, gid, torch.exp(-mg / tau - big[gid])
            )
            L = (
                _pool_loss(m, len(tr), -tau * (big + torch.log(sm)), kt)
                + lam * m.penalty()
            )
            L.backward()
            return L

        opt.step(closure)
        with torch.no_grad():
            v = robust_objective(m, tr_t, y, B, lam)[0].item()
        if v < best:
            best, state = v, [t.detach().clone() for t in (m.W0, m.W1, m.b)]
    with torch.no_grad():
        for t, s in zip((m.W0, m.W1, m.b), state):
            t.copy_(s)


def bruteforce_worst(s, nb, i, budget, base, kappa):
    import itertools

    nb = set(nb)
    cand = [u for u in range(len(s)) if u != i]
    best = np.inf
    for b in range(budget + 1):
        for edits in itertools.combinations(cand, b):
            cur = sorted(nb ^ set(edits))
            best = min(best, base + agg(np.array([s[u] for u in cur]), kappa))
    return best
