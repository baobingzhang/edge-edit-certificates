"""Scheme D: neighbour-additive GAM with exact local edge-edit certificates (S01 kill test).

logit_i = b + S0_i + mean_{j in N(i)} S1_j,  S0 = psi(z) . W0,  S1 = psi(z) . W1.
Scores are linear in (W0, W1, b), so ridge training is convex. Only N(i) enters logit_i, so only edits
incident to i matter, and the worst case for a fixed (p deletions, q insertions) is found by sorting.
"""

import gam
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def tabulated_psi(Z, theta, K, clip):
    idx = torch.randint(
        0,
        Z.numel(),
        (min(2_000_000, Z.numel()),),
        generator=torch.Generator().manual_seed(0),
    )
    psi, lam = gam.mercer_basis(Z.reshape(-1)[idx.to(Z.device)], theta, K)
    tab_x = torch.linspace(-clip, clip, 8193, device=Z.device)
    tab = torch.cat([psi(tab_x[s : s + 1024]) for s in range(0, 8193, 1024)])
    pos = (Z + clip) / (2 * clip) * 8192
    lo = pos.floor().long().clamp(0, 8191)
    frac = (pos - lo).unsqueeze(-1)
    return tab[lo] * (1 - frac) + tab[lo + 1] * frac, lam


class NeighbourGAM(nn.Module):
    def __init__(self, g, c, K=2, theta=2.0, clip=5.0):
        super().__init__()
        x = g.x
        Z = (
            (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-8)
        ).clamp(-clip, clip)
        Psi, lam = tabulated_psi(Z, theta, K, clip)
        self.n, self.d, self.k, self.c = Psi.size(0), Psi.size(1), Psi.size(2), c
        self.register_buffer("Psi", Psi.reshape(self.n, -1))
        ei = g.edge_index[:, g.edge_index[0] != g.edge_index[1]]
        self.register_buffer("ei", ei)
        deg = torch.zeros(self.n, device=x.device).index_add_(
            0, ei[0], torch.ones(ei.size(1), device=x.device)
        )
        self.register_buffer("deg", deg)
        w = (1 / deg.clamp_min(1))[ei[0]]
        A = torch.sparse_coo_tensor(ei, w, (self.n, self.n)).coalesce()
        self.register_buffer("Mean", torch.sparse.mm(A, self.Psi))
        self.register_buffer("pen", (lam.max() / lam).repeat(self.d).reshape(1, -1, 1))
        self.W0 = nn.Parameter(torch.zeros(self.d * self.k, c))
        self.W1 = nn.Parameter(torch.zeros(self.d * self.k, c))
        self.b = nn.Parameter(torch.zeros(c))

    def forward(self):
        return self.Psi @ self.W0 + self.Mean @ self.W1 + self.b

    def penalty(self):
        return ((self.W0**2 + self.W1**2) * self.pen.squeeze(0)).sum()

    def scores(self):
        return self.Psi @ self.W0 + self.b, self.Psi @ self.W1


def fit(m, tr, y, lam):
    opt = torch.optim.LBFGS(
        [m.W0, m.W1, m.b],
        lr=1.0,
        max_iter=300,
        history_size=20,
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(m()[tr], y[tr]) + lam * m.penalty()
        loss.backward()
        return loss

    opt.step(closure)


def worst_mean(s_nb, s_all_sorted, nb_set, budget, base):
    """Exact min of base + mean(kept + added) over <= budget incident edits.

    s_nb: scores of current neighbours; s_all_sorted: (score, node) ascending over candidate insertions.
    """
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
            den = d - p + q
            mean = (total - pre_del[p] + add[q]) / den if den > 0 else 0.0
            best = min(best, base + mean)
    return best


def neighbour_lists(m):
    src, dst = m.ei.cpu().numpy()
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    return np.searchsorted(src, np.arange(m.n + 1)), dst


def radii(m, nodes, budgets):
    """Exact certified budget per node: the largest b in `budgets` such that no <= b incident edits change the prediction."""
    with torch.no_grad():
        S0, S1 = (t.detach().cpu().numpy() for t in m.scores())
        logits = m().detach().cpu().numpy()
    pred = logits.argmax(1)
    start, dst = neighbour_lists(m)
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
                if worst_mean(s[nb], zip(ss.tolist(), idx.tolist()), nbs, b, S0[i, y] - S0[i, c]) <= 0:
                    ok = False
                    break
            if not ok:
                break
            r = b
        out[t] = r
    return pred[nodes], out


def attribution_scores(m, y):
    """Per-feature scores for class y: ego N x D and neighbour N x D. Contributions sum to the logit minus bias."""
    P = m.Psi.reshape(m.n, m.d, m.k)
    W0, W1 = m.W0[:, y].reshape(m.d, m.k), m.W1[:, y].reshape(m.d, m.k)
    return (P * W0).sum(2), (P * W1).sum(2)


def top1_stable(m, nodes, budgets):
    """Largest b such that the top positive-contribution feature for the predicted class provably stays on top."""
    with torch.no_grad():
        pred = m().argmax(1).cpu().numpy()
    start, dst = neighbour_lists(m)
    out = np.zeros(len(nodes), dtype=int)
    cache = {}
    for t, i in enumerate(nodes):
        y = int(pred[i])
        if y not in cache:
            with torch.no_grad():
                E, F1 = attribution_scores(m, y)
            cache[y] = (E.cpu().numpy(), F1.cpu().numpy())
        E, F1 = cache[y]
        nb = dst[start[i] : start[i + 1]]
        nbs = set(nb.tolist()) | {int(i)}
        contrib = E[i] + (F1[nb].mean(0) if len(nb) else 0.0)
        k = int(contrib.argmax())
        r = 0
        for b in budgets:
            ok = True
            # Cheap sound filter: every neighbour mean lies between the extreme scores of that feature.
            diff_lo = (E[i, k] + min(F1[:, k].min(), 0.0)) - (E[i] + np.maximum(F1.max(0), 0.0))
            for l in np.where(diff_lo <= 0)[0]:
                if l == k:
                    continue
                s = F1[:, k] - F1[:, l]
                idx = np.argsort(s, kind="stable")
                if worst_mean(s[nb], zip(s[idx].tolist(), idx.tolist()), nbs, b, E[i, k] - E[i, l]) <= 0:
                    ok = False
                    break
            if not ok:
                break
            r = b
        out[t] = r
    return out


def worst_active(m, tr, y, B, S1):
    """Exact worst-case edit choice for every (train node, wrong class), returned as tensors of kept/added nodes."""
    S1n = S1.detach().cpu().numpy()
    start, dst = neighbour_lists(m)
    rows = []
    orders = {}
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
            sn = s[nb]
            del_order = nb[np.argsort(-sn, kind="stable")]
            desc = s[del_order]
            pre = np.concatenate([[0.0], np.cumsum(desc)])
            added, add_s = [], [0.0]
            for u in idx:
                if len(added) >= B:
                    break
                if int(u) in nbs:
                    continue
                added.append(int(u))
                add_s.append(add_s[-1] + s[u])
            best, arg = np.inf, (0, 0)
            d = len(nb)
            for p in range(min(B, d) + 1):
                for q in range(min(B - p, len(added)) + 1):
                    den = d - p + q
                    v = (pre[-1] - pre[p] + add_s[q]) / den if den else 0.0
                    if v < best:
                        best, arg = v, (p, q)
            p, q = arg
            rows.append((t, c, np.concatenate([del_order[p:], np.array(added[:q], dtype=np.int64)]).astype(np.int64)))
    return rows


def robust_objective(m, tr_t, y, B, lam, rows=None):
    S0, S1 = m.Psi[tr_t] @ m.W0 + m.b, m.Psi @ m.W1
    tr = tr_t.cpu().numpy()
    if rows is None:
        rows = worst_active(m, tr, y.cpu().numpy(), B, S1)
    idx_row, nodes, weight, cls_t = [], [], [], []
    for r, (t, c, st) in enumerate(rows):
        cls_t.append((t, c))
        if len(st):
            idx_row.append(np.full(len(st), r))
            nodes.append(st)
            weight.append(np.full(len(st), 1.0 / len(st), dtype=np.float32))
    dev = m.Psi.device
    ts = torch.as_tensor([a for a, _ in cls_t], device=dev)
    cs = torch.as_tensor([b for _, b in cls_t], device=dev)
    ys = y[tr_t][ts]
    base = S0[ts, ys] - S0[ts, cs]
    mean = torch.zeros(len(rows), device=dev)
    if idx_row:
        ir = torch.as_tensor(np.concatenate(idx_row), device=dev)
        nd = torch.as_tensor(np.concatenate(nodes), device=dev)
        wt = torch.as_tensor(np.concatenate(weight), device=dev)
        vals = (S1[nd, ys[ir]] - S1[nd, cs[ir]]) * wt
        mean = mean.index_add(0, ir, vals)
    margins = base + mean
    # log(1 + sum_c exp(-m_c)) per train node; convex in the parameters because each m_c is concave.
    pos, seen = [], {}
    for t, _ in cls_t:
        pos.append(seen.get(t, 0))
        seen[t] = pos[-1] + 1
    # Slot c-1 stays 0, supplying the "1 +" inside the log.
    acc = torch.zeros(len(tr), m.c, device=dev).index_put((ts, torch.as_tensor(pos, device=dev)), -margins)
    loss = torch.logsumexp(acc, dim=1).mean()
    return loss + lam * m.penalty(), rows


def robust_fit(m, tr_t, y, lam, B=1, steps=300, lr=0.02, warm=True):
    """Exact subgradient descent on the convex robust objective, keeping the best iterate seen."""
    if warm:
        with torch.no_grad():
            m.W0.zero_(), m.W1.zero_(), m.b.zero_()
        mask = torch.zeros(m.n, dtype=torch.bool, device=m.Psi.device)
        mask[tr_t] = True
        fit(m, mask, y, lam)
    opt = torch.optim.Adam([m.W0, m.W1, m.b], lr=lr)
    best, state, hist = np.inf, None, []
    for it in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr / np.sqrt(1 + it / 20)
        opt.zero_grad()
        obj, _ = robust_objective(m, tr_t, y, B, lam)
        v = obj.item()
        hist.append(v)
        if v < best:
            best, state = v, [t.detach().clone() for t in (m.W0, m.W1, m.b)]
        obj.backward()
        opt.step()
    with torch.no_grad():
        for t, b in zip((m.W0, m.W1, m.b), state):
            t.copy_(b)
    return best, hist


def bruteforce_worst(s, nb, i, budget, base):
    """Enumerate every edit set, for testing worst_mean."""
    import itertools

    n = len(s)
    nb = set(nb)
    cand = [u for u in range(n) if u != i]
    best = np.inf
    for b in range(budget + 1):
        for edits in itertools.combinations(cand, b):
            cur = nb ^ set(edits)
            mean = np.mean([s[u] for u in cur]) if cur else 0.0
            best = min(best, base + mean)
    return best
