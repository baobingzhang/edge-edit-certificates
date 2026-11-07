"""Exact single-edge-edit certificates for a two-layer mean-aggregation GraphSAGE (PLAN_v5, E6).

Toggling {u,v} changes only N(u), N(v), h_u and h_v, so it can change o_i only if u or v lies in N[i] = N(i) + {i}.
Enumerating u in N[i] and every v != u therefore covers every edit that matters: the certificate is exact, not a bound.
"""

import torch


def precompute(m, x, ei):
    """Layer quantities of gam.Conv2(kind="sage") in eval mode. Neighbours of w are the sources of edges into w."""
    n = x.size(0)
    a, b = m.a, m.b
    Z, R = x @ a.lin_l.weight.T, x @ a.lin_r.weight.T
    src, dst = ei
    deg = torch.bincount(dst, minlength=n).float()
    SZ = torch.zeros_like(Z).index_add_(0, dst, Z[src])
    H = torch.relu(SZ / deg.clamp_min(1)[:, None] + a.lin_l.bias + R)
    HS = torch.zeros_like(H).index_add_(0, dst, H[src])
    out = (
        (HS / deg.clamp_min(1)[:, None]) @ b.lin_l.weight.T
        + b.lin_l.bias
        + H @ b.lin_r.weight.T
    )
    order = torch.argsort(dst)
    ptr = torch.zeros(n + 1, dtype=torch.long, device=x.device)
    ptr[1:] = torch.cumsum(torch.bincount(dst, minlength=n), 0)
    return dict(
        n=n,
        Z=Z,
        R=R,
        deg=deg,
        SZ=SZ,
        H=H,
        HS=HS,
        out=out,
        nb=src[order],
        ptr=ptr,
        b1=a.lin_l.bias,
        W2=b.lin_l.weight,
        b2=b.lin_l.bias,
        R2=b.lin_r.weight,
    )


def neighbours(P, w):
    return P["nb"][P["ptr"][w] : P["ptr"][w + 1]]


def toggled_logits(P, i, us, vs):
    """Logits of node i after toggling the single edge {u,v}, for every u in `us` (k) and v in `vs` (m): k x m x C.
    Pairs with u == v are meaningless and must be ignored by the caller."""
    dev = P["Z"].device
    k = len(us)
    us = torch.as_tensor(us, device=dev)
    vs = torch.as_tensor(vs, device=dev)
    mem = torch.zeros(
        k, P["n"], dtype=torch.bool, device=dev
    )  # mem[a, v]: v in N(us[a])
    for a, u in enumerate(us.tolist()):
        mem[a, neighbours(P, u)] = True
    s = torch.where(
        mem[:, vs], -1.0, 1.0
    )  # -1 deletes an existing edge, +1 inserts one
    inN = torch.zeros(P["n"], dtype=torch.bool, device=dev)
    inN[neighbours(P, i)] = True

    def h_new(
        SZw, Zo, degw, Rw
    ):  # new hidden state of w after w gains (+1) or loses (-1) the other endpoint
        d = degw + s
        agg = torch.where(
            d[..., None] > 0,
            (SZw + s[..., None] * Zo) / d.clamp_min(1)[..., None],
            torch.zeros(()),
        )
        return torch.relu(agg + P["b1"] + Rw)

    hu = h_new(
        P["SZ"][us][:, None],
        P["Z"][vs][None],
        P["deg"][us][:, None],
        P["R"][us][:, None],
    )
    hv = h_new(
        P["SZ"][vs][None], P["Z"][us][:, None], P["deg"][vs][None], P["R"][vs][None]
    )
    in_u = inN[us][:, None].float().expand(k, len(vs))
    in_v = inN[vs][None].float().expand(k, len(vs))
    u_is_i = (us == i)[:, None].expand(k, len(vs))
    v_is_i = (vs == i)[None].expand(k, len(vs))
    in_v2 = torch.where(
        u_is_i, 1 - in_v, in_v
    )  # toggling {i,v} flips whether v is a neighbour of i
    in_u2 = torch.where(v_is_i, 1 - in_u, in_u)
    S = (
        P["HS"][i]
        - in_u[..., None] * P["H"][us][:, None]
        - in_v[..., None] * P["H"][vs][None]
        + in_u2[..., None] * hu
        + in_v2[..., None] * hv
    )
    cnt = P["deg"][i] - in_u - in_v + in_u2 + in_v2
    hi = torch.where(
        u_is_i[..., None],
        hu,
        torch.where(v_is_i[..., None], hv, P["H"][i].expand_as(hu)),
    )
    mean = torch.where(
        cnt[..., None] > 0, S / cnt.clamp_min(1)[..., None], torch.zeros(())
    )
    return mean @ P["W2"].T + P["b2"] + hi @ P["R2"].T


@torch.no_grad()
def certify_b1(P, nodes, budget_floats=6e7):
    """For each node: True iff no single edge toggle changes its predicted class."""
    n, Hd = P["n"], P["Z"].size(1)
    allv = torch.arange(n, device=P["Z"].device)
    chunk = max(1, int(budget_floats // (n * Hd)))
    res = []
    for i in nodes:
        i = int(i)
        y = int(P["out"][i].argmax())
        U = torch.unique(
            torch.cat([neighbours(P, i), torch.tensor([i], device=allv.device)])
        ).tolist()
        ok = True
        for c0 in range(0, len(U), chunk):
            us = U[c0 : c0 + chunk]
            o = toggled_logits(P, i, us, allv)
            flip = o.argmax(-1) != y
            flip[torch.arange(len(us)), torch.as_tensor(us, device=allv.device)] = (
                False  # u == v is not an edit
            )
            if bool(flip.any()):
                ok = False
                break
        res.append(ok)
    return res
