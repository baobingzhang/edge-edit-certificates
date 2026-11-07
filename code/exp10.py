"""E10 (PLAN_v8): pi-PPNP with CE / RCE / CEM training, PPR certificates and a b=1 attack upper bound.
usage: exp10.py DATASET [--time]   -> results/s09/DATASET__pippnp.json   (--time: only the timing rule)
"""

import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

import exp
import gam
import ppr

OUT = os.path.join(gam.ROOT, "results", "s09")
ALPHAS = [0.1, 0.5, 0.85]
LOSSES = ["ce", "rce", "cem"]
LIMIT_SEC = 2 * 3600


class MLP(torch.nn.Module):
    def __init__(self, d, c, h=64):
        super().__init__()
        self.l1, self.l2 = torch.nn.Linear(d, h), torch.nn.Linear(h, c)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))


def losses(G, H, Z, alpha, kind, y, tr, va):
    if kind == "ce":
        return F.cross_entropy(Z[tr], y[tr]), F.cross_entropy(Z[va], y[va])
    obs = torch.cat([tr, va])
    A = ppr.adv_logits(G, H, alpha, y[obs], obs, b=1)
    At, Av = A[: len(tr)], A[len(tr) :]
    if kind == "rce":
        return F.cross_entropy(-At, y[tr]), F.cross_entropy(-Av, y[va])

    def cem(Zs, As, ys):  # official cem_loss, margin 1
        return (
            F.cross_entropy(Zs, ys, reduction="none")
            + (torch.clamp(1 - As, min=0).sum(1) - 1)
        ).mean()

    return cem(Z[tr], At, y[tr]), cem(Z[va], Av, y[va])


def train(
    G, x, y, tr, va, alpha, kind, seed, max_epochs=3000, patience=50, epochs_cap=None
):
    """Official training loop (training.py): Adam 1e-2, wd 1e-4, keep the state with lower val loss or higher val acc."""
    torch.manual_seed(seed)
    m = MLP(x.shape[1], int(y.max()) + 1).to(x.device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=1e-4)
    best_loss, best_acc, best_ep, best = math.inf, -math.inf, 0, None
    t0, ep = time.time(), 0
    for ep in range(epochs_cap or max_epochs):
        H = m(x)
        Z = ppr.propagate(G, H, alpha)
        lt, lv = losses(G, H, Z, alpha, kind, y, tr, va)
        opt.zero_grad()
        lt.backward()
        opt.step()
        with torch.no_grad():
            acc_v = float((Z[va].argmax(1) == y[va]).float().mean())
        if lv.item() < best_loss or acc_v > best_acc:
            best_loss, best_acc, best_ep = lv.item(), acc_v, ep
            best = {k: v.detach().clone() for k, v in m.state_dict().items()}
        elif ep >= best_ep + patience:
            break
    m.load_state_dict(best)
    return m, dict(epochs=ep + 1, best_epoch=best_ep, sec=time.time() - t0)


def sample(mask, cap, seed):
    idx = mask.nonzero().flatten().cpu().numpy()
    if len(idx) > cap:
        idx = np.sort(np.random.default_rng(seed).choice(idx, cap, replace=False))
    return idx


def ppr_rows_after_toggle(G, toggles, targets, alpha, tol=1e-10):
    """PPR rows pi'_t of target t after toggling one undirected edge per copy (exact up to tol)."""
    K, n, dev = len(toggles), G.n, G.dev
    E = len(G.row)
    kk = torch.arange(K, device=dev).repeat_interleave(E)
    r, c = G.row.repeat(K), G.col.repeat(K)
    a, b = toggles[:, 0], toggles[:, 1]
    tk = torch.cat([torch.arange(K, device=dev)] * 2)
    tr_, tc_ = torch.cat([a, b]), torch.cat([b, a])
    key = (kk * n + r) * n + c
    tkey = (tk * n + tr_) * n + tc_
    exists = torch.isin(tkey, key)
    keep = ~torch.isin(key, tkey[exists])
    rows = torch.cat(
        [(kk * n + r)[keep], torch.arange(K * n, device=dev), (tk * n + tr_)[~exists]]
    )
    cols = torch.cat(
        [(kk * n + c)[keep], torch.arange(K * n, device=dev), (tk * n + tc_)[~exists]]
    )
    w = (1.0 / torch.bincount(rows, minlength=K * n).double())[rows]
    e = torch.zeros(K * n, dtype=torch.float64, device=dev)
    e[torch.arange(K, device=dev) * n + targets] = 1
    u = e.clone()
    while True:  # u = e + alpha P'^T u
        un = e + alpha * torch.zeros_like(u).index_add_(0, cols, u[rows] * w)
        if (un - u).abs().max() < tol * (1 - alpha):
            u = un
            break
        u = un
    return ((1 - alpha) * u).reshape(K, n)


def attack(G, H, alpha, nodes, pred, nclass, per=10, batch=64):
    """b=1 undirected attack: per node, the top `per` PI-optimal flips by pi_t(i)*gain and the top `per` toggles
    incident to t by gain (over all wrong classes), each re-evaluated exactly. Returns broken flags.
    """
    n, dev = G.n, G.dev
    Hd = H.to(torch.float64)
    ys = sorted(set(pred[nodes].tolist()))
    pairs = [(y, c) for y in ys for c in range(nclass) if c != y]
    R = torch.stack([Hd[:, c] - Hd[:, y] for y, c in pairs])
    V, adds, dels = ppr.policy_iteration(G, R, alpha, 1)
    T = (V - R) / alpha
    nodes_t = torch.as_tensor(nodes, device=dev)
    # clean PPR rows of the targets: u = e + alpha P^T u
    w = 1.0 / (G.deg + 1).double()
    E = torch.zeros(len(nodes), n, dtype=torch.float64, device=dev)
    E[torch.arange(len(nodes)), nodes_t] = 1
    U = E.clone()
    while True:
        agg = torch.zeros_like(U).index_add_(1, G.col, U[:, G.row] * w[G.row]) + U * w
        Un = E + alpha * agg
        if (Un - U).abs().max() < 1e-10 * (1 - alpha):
            U = Un
            break
        U = Un
    PI = (1 - alpha) * U  # rows pi_t
    fp = torch.cat([adds[0], dels[0]])
    fi = torch.cat([adds[1], dels[1]])
    fj = torch.cat([adds[2], dels[2]])
    fg = torch.cat(
        [
            V[adds[0], adds[2]] - T[adds[0], adds[1]],
            T[dels[0], dels[1]] - V[dels[0], dels[2]],
        ]
    )
    pidx = {p: k for k, p in enumerate(pairs)}
    start, col = G.start.cpu().numpy(), G.col.cpu().numpy()
    broken = np.zeros(len(nodes), dtype=bool)
    tried = 0
    for s in range(0, len(nodes), batch):
        cand_t, cand_e, owner = [], [], []
        for k in range(s, min(s + batch, len(nodes))):
            t, y = int(nodes[k]), int(pred[nodes[k]])
            ps = [pidx[(y, c)] for c in range(nclass) if c != y]
            pm = torch.isin(fp, torch.as_tensor(ps, device=dev))
            sc = PI[k, fi[pm]] * fg[pm]
            top = (
                torch.topk(sc, min(per, int(pm.sum()))).indices
                if pm.any()
                else torch.zeros(0, dtype=torch.long, device=dev)
            )
            es = {
                tuple(sorted((int(a), int(b))))
                for a, b in zip(fi[pm][top].tolist(), fj[pm][top].tolist())
                if a != b
            }
            nb = set(col[start[t] : start[t + 1]].tolist())
            gains = []
            for p in ps:
                g_ = V[p] - T[p, t]
                g_ = g_.clone()
                if nb:
                    idx = torch.as_tensor(sorted(nb), device=dev)
                    g_[idx] = -g_[idx]
                g_[t] = -math.inf
                gains.append(g_)
            gmax = torch.stack(gains).max(0).values
            for j in torch.topk(gmax, min(per, n - 1)).indices.tolist():
                es.add(tuple(sorted((t, int(j)))))
            for e in es:
                cand_t.append(t)
                cand_e.append(e)
                owner.append(k)
        if not cand_t:
            continue
        ch = max(1, int(6e6 // (len(G.row) + n)))
        for q in range(0, len(cand_t), ch):
            tg = torch.as_tensor(cand_t[q : q + ch], device=dev)
            tg_e = torch.as_tensor(cand_e[q : q + ch], device=dev)
            rows = ppr_rows_after_toggle(G, tg_e, tg, alpha)
            Z = rows @ Hd
            for z, k in zip(Z, owner[q : q + ch]):
                y = int(pred[nodes[k]])
                zz = z.clone()
                zz[y] = -math.inf
                broken[k] |= bool(z[y] <= zz.max())
                tried += 1
    return broken, tried


def evaluate(ds, G, m, x, y, alpha, mask, cap, seed, budgets, n_attack=0):
    nclass = int(y.max()) + 1
    with torch.no_grad():
        H = m(x)
    nodes = sample(mask, cap, seed)
    pred, rad, rel = ppr.certify(G, H, alpha, nodes, budgets, nclass)
    s = exp.scores(ds, rel.float(), mask, y)
    pr = pred.cpu().numpy()
    ok = pr[nodes] == y.cpu().numpy()[nodes]
    rad = rad.cpu().numpy()
    out = dict(
        acc=s["acc"],
        metric=s["metric"],
        nodes=nodes.tolist(),
        correct=ok.tolist(),
        rad=rad.tolist(),
        deg=G.deg.cpu().numpy()[nodes].astype(int).tolist(),
        cert={str(b): float((ok & (rad >= b)).mean()) for b in budgets},
    )
    if n_attack:
        an = nodes[:n_attack]
        broken, tried = attack(G, H, alpha, an, pr, nclass)
        ub = ok[:n_attack] & ~broken
        out.update(
            attack_nodes=int(len(an)),
            attack_tried=int(tried),
            broken=broken.tolist(),
            ub1=float(ub.mean()),
            cert1_attacked=float((ok[:n_attack] & (rad[:n_attack] >= 1)).mean()),
            unsound=int((broken & (rad[:n_attack] >= 1)).sum()),
        )
    return out


def main():
    ds = sys.argv[1]
    only_time = "--time" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{ds}__pippnp.json")
    if (
        not only_time
        and os.path.exists(path)
        and json.load(open(path)).get("status") == "ok"
    ):
        print(ds, "skip")
        return
    t0 = time.time()
    out = dict(ds=ds, status="running")
    try:
        g, ms = exp.splits(ds)
        G = ppr.Graph(g.edge_index.cuda(), g.num_nodes, "cuda")
        x, y = g.x.cuda().float(), g.y.cuda()
        tr0, va0 = (ms[k][:, 0].nonzero().flatten().cuda() for k in (0, 1))
        # timing rule: CE run gives the epoch count; robust cost per epoch from the first 5 epochs
        timing = {}
        for a in ALPHAS:
            _, info = train(G, x, y, tr0, va0, a, "ce", 0)
            for k in ("rce", "cem"):
                _, ti = train(G, x, y, tr0, va0, a, k, 0, epochs_cap=5)
                timing[f"{a}_{k}"] = dict(
                    per_epoch=ti["sec"] / 5,
                    ce_epochs=info["epochs"],
                    est=ti["sec"] / 5 * info["epochs"],
                )
        robust = all(v["est"] <= LIMIT_SEC for v in timing.values())
        out.update(timing=timing, robust=robust)
        print(
            ds,
            "timing",
            {k: round(v["est"]) for k, v in timing.items()},
            "robust" if robust else "CE only",
            flush=True,
        )
        if only_time:
            return
        trials = []
        for a in ALPHAS:
            for k in LOSSES if robust else ["ce"]:
                m, info = train(G, x, y, tr0, va0, a, k, 0)
                e = evaluate(ds, G, m, x, y, a, ms[1][:, 0], 1000, 0, [1])
                trials.append(
                    dict(
                        alpha=a,
                        loss=k,
                        acc=e["acc"],
                        c1=e["cert"]["1"],
                        val=0.5 * e["acc"] + 0.5 * e["cert"]["1"],
                        **info,
                    )
                )
                print(
                    ds,
                    "val",
                    {
                        kk: (round(v, 4) if isinstance(v, float) else v)
                        for kk, v in trials[-1].items()
                    },
                    flush=True,
                )
        best = max(
            trials, key=lambda t: (t["val"], -LOSSES.index(t["loss"]), t["alpha"])
        )
        out.update(
            trials=trials, cfg=dict(alpha=best["alpha"], loss=best["loss"]), rows=[]
        )
        for s in range(3, ms[2].size(1)):
            tr, va = (ms[k][:, s].nonzero().flatten().cuda() for k in (0, 1))
            m, info = train(G, x, y, tr, va, best["alpha"], best["loss"], s)
            r = evaluate(
                ds,
                G,
                m,
                x,
                y,
                best["alpha"],
                ms[2][:, s],
                1000,
                s,
                exp.BUD,
                n_attack=200,
            )
            r.update(split=s, **info)
            out["rows"].append(r)
            print(
                ds,
                s,
                best,
                f"acc={r['acc']:.3f} c1={r['cert']['1']:.3f} ub1={r['ub1']:.3f} unsound={r['unsound']}",
                flush=True,
            )
        out["status"] = "ok"
    except Exception:
        out.update(status="error", trace=traceback.format_exc())
        print(out["trace"])
    out["sec"] = time.time() - t0
    if not only_time:
        json.dump(out, open(path, "w"))
    print(ds, out["status"], f"{out['sec']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
