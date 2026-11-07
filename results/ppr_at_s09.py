"""E10 (PLAN_v8): pi-PPNP (Bojchevski & Guennemann, NeurIPS 2019) with self-loops, and its PPR certificate.

Model: logits = Pi H, Pi = (1-alpha)(I - alpha D^-1 (A+I))^-1, H = MLP(x). Self-loops are fixed (never edited).
Threat model of the certificate: every ordered pair (i, j), i != j, is fragile; every node may flip at most b of
its out-edges. Policy iteration (their Alg. 1) gives the exact worst case for this local-budget problem; since b
undirected toggles flip at most b out-edges per node, the certificate is sound (not exact) for b global edits.

Policy iteration is batched over class pairs on the GPU. For row i the improving flips are additions of
non-neighbours j with v_j > t_i and deletions of neighbours with v_j < t_i, t_i = (v_i - r_i)/alpha, which is the
rule of the official policy_iteration with fragile = all pairs. The best b additions are the b largest v among
non-members of N[i]; with neighbour ranks s_(1)<s_(2)<... in the global descending order and g_m = s_(m)-(m-1),
the k-th eligible rank is (k-1) + #{m : g_m <= k-1}, so no per-row scan is needed.
"""

import math

import torch


class Graph:
    """Undirected graph without self-loops (they are added implicitly as fixed out-edges)."""

    def __init__(self, edge_index, n, device="cuda"):
        ei = edge_index[:, edge_index[0] != edge_index[1]]
        key = torch.unique(ei[0].long() * n + ei[1].long())
        self.n, self.dev = n, device
        self.row = (key // n).to(device)
        self.col = (key % n).to(device)
        self.deg = torch.bincount(self.row, minlength=n).to(device)
        self.start = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(self.deg, 0)]
        )


def propagate(G, H, alpha, steps=None):
    """(1-alpha)(I - alpha P)^-1 H on the clean graph by a truncated Neumann series (differentiable)."""
    steps = steps or max(1, math.ceil(math.log(1e-6) / math.log(alpha)))
    w = 1.0 / (G.deg + 1).to(H.dtype)
    Z = H
    for _ in range(steps):
        agg = torch.zeros_like(H).index_add_(0, G.row, Z[G.col]) + Z
        Z = H + alpha * agg * w[:, None]
    return (1 - alpha) * Z


def _values(n, P, rows, cols, deg_new, R, alpha, V0, tol=1e-9, max_it=5000):
    """Solve v = r + alpha P' v for P pairs at once. rows/cols index the flattened (P*n) state space; self-loops
    are included in rows/cols. deg_new is the out-degree (incl. self-loop) of each flattened state.
    """
    w = (1.0 / deg_new)[rows]
    V = V0.reshape(-1).clone()
    r = R.reshape(-1)
    scale = 1.0 + r.abs().max()
    for _ in range(max_it):
        agg = torch.zeros_like(V).index_add_(0, rows, V[cols] * w)
        Vn = r + alpha * agg
        if (Vn - V).abs().max() < tol * scale * (1 - alpha):
            V = Vn
            break
        V = Vn
    else:
        raise RuntimeError("value iteration did not converge")
    return V.reshape(P, n)


def _select(G, V, R, alpha, b):
    """Greedy policy step for all pairs. Returns (pair, i, j) for additions and deletions (relative to clean)."""
    P, n = V.shape
    dev = V.device
    t = (V - R) / alpha
    order = torch.argsort(-V, dim=1)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(n, device=dev).expand(P, n))
    # rows' members N[i] = neighbours + self, sorted by rank
    mr = torch.cat([G.row, torch.arange(n, device=dev)])
    mc = torch.cat([G.col, torch.arange(n, device=dev)])
    mstart = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(G.deg + 1, 0)]
    )
    rk = rank[:, mc]  # P x M
    key = mr[None, :] * (n + 1) + rk
    srt = torch.argsort(key, dim=1)
    rk_s = torch.gather(rk, 1, srt)
    row_s = mr[srt]
    m = torch.arange(len(mr), device=dev)[None, :] - mstart[row_s]
    g = rk_s - m
    hist = torch.zeros(P, n, b + 1, dtype=torch.long, device=dev)
    ok = g <= b - 1
    pidx = torch.arange(P, device=dev)[:, None].expand_as(g)
    hist.index_put_(
        (pidx[ok], row_s[ok], g[ok]),
        torch.ones(int(ok.sum()), dtype=torch.long, device=dev),
        accumulate=True,
    )
    cnt = torch.cumsum(hist[:, :, :b], 2)
    pos = (
        torch.arange(b, device=dev)[None, None, :] + cnt
    )  # P x n x b, ranks of the b best eligible
    valid_add = pos < n
    add_j = torch.gather(order, 1, pos.clamp(max=n - 1).reshape(P, -1)).reshape(P, n, b)
    add_gain = torch.gather(V, 1, add_j.reshape(P, -1)).reshape(P, n, b) - t[:, :, None]
    add_gain = torch.where(valid_add, add_gain, torch.full_like(add_gain, -math.inf))
    # deletions: the b smallest v among neighbours (self-loop excluded)
    E = len(G.row)
    if E:
        rk2 = rank[:, G.col]
        key2 = G.row[None, :] * (n + 1) + rk2
        srt2 = torch.argsort(key2, dim=1)
        row2 = G.row[srt2]
        rev = G.start[row2 + 1] - 1 - torch.arange(E, device=dev)[None, :]
        keep = rev < b
        del_j = torch.full((P, n, b), 0, dtype=torch.long, device=dev)
        del_gain = torch.full((P, n, b), -math.inf, dtype=V.dtype, device=dev)
        pk = torch.arange(P, device=dev)[:, None].expand_as(srt2)[keep]
        rr, slot = row2[keep], rev[keep]
        jj = G.col[srt2][keep]
        del_j[pk, rr, slot] = jj
        del_gain[pk, rr, slot] = t[pk, rr] - V[pk, jj]
    else:
        del_j = torch.zeros(P, n, b, dtype=torch.long, device=dev)
        del_gain = torch.full((P, n, b), -math.inf, dtype=V.dtype, device=dev)
    gain = torch.cat([add_gain, del_gain], 2)
    cand = torch.cat([add_j, del_j], 2)
    top = torch.topk(gain, b, dim=2)
    take = top.values > 0
    j = torch.gather(cand, 2, top.indices)
    is_add = top.indices < b
    p_, i_, k_ = take.nonzero(as_tuple=True)
    jj = j[p_, i_, k_]
    a = is_add[p_, i_, k_]
    return (p_[a], i_[a], jj[a]), (p_[~a], i_[~a], jj[~a])


def _build(G, P, adds, dels):
    """Flattened (P*n) transition structure after the given flips, self-loops included."""
    n, dev = G.n, G.dev
    E = len(G.row)
    pe = torch.arange(P, device=dev).repeat_interleave(E)
    re, ce = G.row.repeat(P), G.col.repeat(P)
    if len(dels[0]):
        dk = (dels[0] * n + dels[1]) * n + dels[2]
        ek = (pe * n + re) * n + ce
        keep = ~torch.isin(ek, dk)
        pe, re, ce = pe[keep], re[keep], ce[keep]
    ps = torch.arange(P, device=dev).repeat_interleave(n)
    ss = torch.arange(n, device=dev).repeat(P)
    rows = torch.cat([pe * n + re, ps * n + ss, adds[0] * n + adds[1]])
    cols = torch.cat([pe * n + ce, ps * n + ss, adds[0] * n + adds[2]])
    deg = torch.bincount(rows, minlength=P * n).to(torch.float64)
    return rows, cols, deg


def policy_iteration(G, R, alpha, b, max_iter=200):
    """Maximise (1-alpha) e_t^T (I - alpha P')^-1 r over local-budget-b flips, for every start t and every column
    of R (P x n, float64) at once. Returns V (P x n) with V[p, t] = optimal (I - alpha P')^-1 r evaluated at t,
    and the optimal flips."""
    P, n = R.shape
    empty = (torch.zeros(0, dtype=torch.long, device=G.dev),) * 3
    adds, dels = empty, empty
    V = R.clone()
    prev = None
    for _ in range(max_iter):
        rows, cols, deg = _build(G, P, adds, dels)
        V = _values(n, P, rows, cols, deg, R, alpha, V)
        if b == 0:
            return V, adds, dels
        adds, dels = _select(G, V, R, alpha, b)
        sig = torch.sort(
            torch.cat(
                [
                    ((adds[0] * n + adds[1]) * n + adds[2]) * 2,
                    ((dels[0] * n + dels[1]) * n + dels[2]) * 2 + 1,
                ]
            )
        )[0]
        if prev is not None and len(sig) == len(prev) and torch.equal(sig, prev):
            return V, adds, dels
        prev = sig
    raise RuntimeError("policy iteration did not converge")


def worst_margins(G, H, alpha, b, pairs, chunk_states=4_000_000):
    """For each (y, c) in pairs: W[(y,c)][t] = min over perturbations of the margin pi'_t (H_y - H_c)."""
    H = H.to(torch.float64)
    out = {}
    per = max(1, chunk_states // G.n)
    for s in range(0, len(pairs), per):
        ch = pairs[s : s + per]
        R = torch.stack([H[:, c] - H[:, y] for y, c in ch])  # maximise r = H_c - H_y
        V, _, _ = policy_iteration(G, R, alpha, b)
        for k, (y, c) in enumerate(ch):
            out[(y, c)] = -(1 - alpha) * V[k]
    return out


def certify(G, H, alpha, nodes, budgets, nclass):
    """Predictions on the clean graph (exact solve) and the largest certified budget for each node."""
    Hd = H.to(torch.float64)
    clean = worst_margins(G, Hd, alpha, 0, [(0, c) for c in range(nclass)])
    # clean logits: pi_t H = margin(0,c) gives H_0 - H_c propagated; recover argmax via logits relative to class 0
    rel = torch.stack([-clean[(0, c)] for c in range(nclass)], 1)  # (Pi H)_c - (Pi H)_0
    pred = rel.argmax(1)
    nodes_t = torch.as_tensor(nodes, device=G.dev)
    py = pred[nodes_t]
    ys = sorted(set(py.tolist()))
    rad = torch.zeros(len(nodes), dtype=torch.long, device=G.dev)
    alive = torch.ones(len(nodes), dtype=torch.bool, device=G.dev)
    for b in budgets:
        W = worst_margins(
            G, Hd, alpha, b, [(y, c) for y in ys for c in range(nclass) if c != y]
        )
        ok = torch.ones(len(nodes), dtype=torch.bool, device=G.dev)
        for (y, c), w in W.items():
            sel = py == y
            ok[sel] &= w[nodes_t[sel]] > 0
        alive &= ok
        rad[alive] = b
    return pred, rad, rel


def adv_logits(G, H, alpha, labels_obs, obs, b=1):
    """Worst-case margins (differentiable in H) of observed nodes w.r.t. their labels, as in the official
    get_adv_logits: A[t, c] = pi'_t (H_y - H_c) with the optimal perturbation for pair (y, c); A[t, y] = 0.
    """
    nclass = H.shape[1]
    ys = sorted(set(labels_obs.tolist()))
    pairs = [(y, c) for y in ys for c in range(nclass) if c != y]
    with torch.no_grad():
        Hd = H.detach().to(torch.float64)
        R = torch.stack([Hd[:, c] - Hd[:, y] for y, c in pairs])
        _, adds, dels = policy_iteration(G, R, alpha, b)
        rows, cols, deg = _build(G, len(pairs), adds, dels)
    n = G.n
    w = (1.0 / deg).to(H.dtype)[rows]
    D = torch.stack([H[:, y] - H[:, c] for y, c in pairs]).reshape(-1)
    steps = max(1, math.ceil(math.log(1e-6) / math.log(alpha)))
    Z = D
    for _ in range(steps):
        Z = D + alpha * torch.zeros_like(D).index_add(0, rows, Z[cols] * w)
    Z = ((1 - alpha) * Z).reshape(len(pairs), n)
    A = torch.zeros(len(obs), nclass, device=H.device, dtype=H.dtype)
    pi = {p: k for k, p in enumerate(pairs)}
    for y in ys:
        sel = (labels_obs == y).nonzero().flatten()
        for c in range(nclass):
            if c != y:
                A = A.index_put((sel, torch.full_like(sel, c)), Z[pi[(y, c)], obs[sel]])
    return A


def exact_b1(G, H, alpha, nodes, nclass, row_chunk=2048):
    """Exact worst margin of pi-PPNP under one undirected toggle {u, v}, u != v, anywhere in the graph (PLAN_v8b).

    M = (I - alpha P)^-1. Toggling {u, v} changes rows u, v of P by delta_w with
    delta_w^T z = s (z_o - (P z)_w) / (dh_w + s), s = +1 (insert) or -1 (delete), o the other endpoint. With
    S = [[delta_u^T M e_u, delta_u^T M e_v], [delta_v^T M e_u, delta_v^T M e_v]] and b = (delta_u^T z, delta_v^T z),
    z = M r, Woodbury gives the new margin m + (1-alpha) alpha [x_u, x_v] (I - alpha S)^-1 b, x = row t of M.
    Dense fp64; intended for n <= 8000. Returns pred (all nodes), worst margin per node in `nodes`, and the argmin
    toggle per node.
    """
    n, dev = G.n, G.dev
    Ah = torch.zeros(n, n, dtype=torch.float64, device=dev)
    Ah[G.row, G.col] = 1
    s = 1 - 2 * Ah  # +1 non-edge (insert), -1 edge (delete)
    Ah += torch.eye(n, dtype=torch.float64, device=dev)
    dh = Ah.sum(1)
    P = Ah / dh[:, None]
    M = torch.linalg.inv(torch.eye(n, dtype=torch.float64, device=dev) - alpha * P)
    PM = P @ M
    dM, dPM = M.diagonal().clone(), PM.diagonal().clone()
    del Ah
    Hd = H.to(torch.float64)
    Z = M @ Hd
    PZ = P @ Z
    logits = (1 - alpha) * Z
    pred = logits.argmax(1)
    nodes_t = torch.as_tensor(nodes, device=dev)
    py = pred[nodes_t]
    worst = torch.full((len(nodes),), math.inf, dtype=torch.float64, device=dev)
    arg = torch.zeros(len(nodes), 2, dtype=torch.long, device=dev)
    for y in sorted(set(py.tolist())):
        sel = (py == y).nonzero().flatten()
        for c in range(nclass):
            if c == y:
                continue
            z, pz = Z[:, y] - Z[:, c], PZ[:, y] - PZ[:, c]
            W = torch.empty(n, n, dtype=torch.float64, device=dev)
            for a in range(0, n, row_chunk):
                e = min(n, a + row_chunk)
                sc = s[a:e]
                du = dh[a:e, None] + sc
                dv = dh[None, :] + sc
                S11 = sc * (M[:, a:e].T - dPM[a:e, None]) / du
                S12 = sc * (dM[None, :] - PM[a:e]) / du
                S21 = sc * (dM[a:e, None] - PM[:, a:e].T) / dv
                S22 = sc * (M[a:e] - dPM[None, :]) / dv
                k11, k12, k21, k22 = 1 - alpha * S11, -alpha * S12, -alpha * S21, 1 - alpha * S22
                det = k11 * k22 - k12 * k21
                bu = sc * (z[None, :] - pz[a:e, None]) / du
                bv = sc * (z[a:e, None] - pz[None, :]) / dv
                # first row of (I - alpha S)^-1 b, i.e. the coefficient of x_u
                W[a:e] = (k22 * bu - k12 * bv) / det
            m0 = (1 - alpha) * (M[nodes_t[sel]] @ (Hd[:, y] - Hd[:, c]))
            for q, t in enumerate(nodes_t[sel].tolist()):
                Y = M[t][:, None] * W
                D = Y + Y.T
                D.fill_diagonal_(math.inf)
                k = int(torch.argmin(D))
                w = m0[q] + (1 - alpha) * alpha * D.view(-1)[k]
                i = int(sel[q])
                if w < worst[i]:
                    worst[i] = w
                    arg[i, 0], arg[i, 1] = k // n, k % n
    return pred, worst, arg, logits
