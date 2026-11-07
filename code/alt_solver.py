"""Cutting-plane solver for certified training (ALT-1)."""
import numpy as np, torch, cert

def margins_from_rows(m, tr_t, y, rows):
    S0, S1 = m.Psi[tr_t] @ m.W0 + m.b, m.Psi @ m.W1
    dev = m.Psi.device; ir, nd, wt = [], [], []
    for r, (_, _, st) in enumerate(rows):
        if len(st):
            ir.append(np.full(len(st), r)); nd.append(st); wt.append(np.full(len(st), 1.0 / len(st), dtype=np.float32))
    ts = torch.as_tensor([a for a, _, _ in rows], device=dev); cs = torch.as_tensor([b for _, b, _ in rows], device=dev); ys = y[tr_t][ts]
    mean = torch.zeros(len(rows), device=dev)
    if ir:
        I = torch.as_tensor(np.concatenate(ir), device=dev); N = torch.as_tensor(np.concatenate(nd), device=dev); W = torch.as_tensor(np.concatenate(wt), device=dev)
        mean = mean.index_add(0, I, (S1[N, ys[I]] - S1[N, cs[I]]) * W)
    return S0[ts, ys] - S0[ts, cs] + mean, ts, cs

def true_obj(m, tr_t, y, B, lam):
    with torch.no_grad(): return cert.robust_objective(m, tr_t, y, B, lam)[0].item()

def fit_cutting(m, tr_t, y, lam, B, rounds=12, tau=0.02):
    """Accumulate worst-case sets; between rounds minimise a smooth convex soft-min surrogate with L-BFGS."""
    tr = tr_t.cpu().numpy(); ynp = y.cpu().numpy(); pool, seen, best, state, hist = [], set(), np.inf, None, []
    for _ in range(rounds):
        with torch.no_grad(): S1 = m.Psi @ m.W1
        for t, c, st in cert.worst_active(m, tr, ynp, B, S1):
            k = (t, c, tuple(sorted(st.tolist())))
            if k not in seen: seen.add(k); pool.append((t, c, st))
        keys = {}
        gid = torch.as_tensor([keys.setdefault((t, c), len(keys)) for t, c, _ in pool], device=m.Psi.device)
        G = len(keys); kt = torch.as_tensor([k[0] for k in keys], device=m.Psi.device)
        opt = torch.optim.LBFGS([m.W0, m.W1, m.b], lr=1.0, max_iter=60, history_size=20, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); mg, _, _ = margins_from_rows(m, tr_t, y, pool)
            big = torch.zeros(G, device=mg.device).scatter_reduce(0, gid, -mg / tau, "amax", include_self=False)
            sm = torch.zeros(G, device=mg.device).index_add(0, gid, torch.exp(-mg / tau - big[gid]))
            softmin = -tau * (big + torch.log(sm))
            acc = torch.zeros(len(tr), m.c, device=mg.device)
            pos, cnt = [], {}
            for k in keys: pos.append(cnt.get(k[0], 0)); cnt[k[0]] = pos[-1] + 1
            acc = acc.index_put((kt, torch.as_tensor(pos, device=mg.device)), -softmin)
            L = torch.logsumexp(acc, 1).mean() + lam * m.penalty(); L.backward(); return L
        opt.step(closure)
        v = true_obj(m, tr_t, y, B, lam); hist.append(v)
        if v < best: best, state = v, [t.detach().clone() for t in (m.W0, m.W1, m.b)]
    with torch.no_grad():
        for t, s in zip((m.W0, m.W1, m.b), state): t.copy_(s)
    return best, hist, len(pool)

