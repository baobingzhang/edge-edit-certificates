"""S04 post-review analyses (PLAN_v4). All splits here were already seen; results are labelled supplementary.
usage: exp4.py certmlp|certmlprob|smoothsage|gnam_t3 DATASET"""
import itertools, json, os, random, sys, time, traceback
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import gam, cert, run, smooth, exp_helpers as eh
BUD = [1, 2, 3, 5, 10]
class CertMLP(nn.Module):
    """logit_i = f(x_i) + mean_{N(i)} g(x_j): the exact certificate needs only this form, not additivity."""
    def __init__(self, g, c, hidden=64, drop=0.5, **_):
        super().__init__()
        d = g.x.size(1); self.x = g.x; self.n = g.num_nodes; self.c = c; self.drop = drop
        self.f = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, c)); self.g = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, c))
        ei = g.edge_index[:, g.edge_index[0] != g.edge_index[1]]; self.register_buffer("ei", ei)
        deg = torch.zeros(self.n, device=g.x.device).index_add_(0, ei[0], torch.ones(ei.size(1), device=g.x.device)); self.register_buffer("deg", deg)
        self.register_buffer("A", torch.sparse_coo_tensor(ei, (1 / deg.clamp_min(1))[ei[0]], (self.n, self.n)).coalesce())
    def scores(self): return self.f(self.x), self.g(self.x)
    def forward(self, *_):
        s0, s1 = self.scores(); return s0 + torch.sparse.mm(self.A, s1)
def fit_plain(m, tr, va, y, ds, cfg, epochs=1000, patience=200):
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]); best, st, w = -1, None, 0
    for _ in range(epochs):
        m.train(); opt.zero_grad(); F.cross_entropy(m()[tr], y[tr]).backward(); opt.step(); m.eval()
        with torch.no_grad(): v = gam.metric(ds, m()[va], y[va])
        if v > best: best, w, st = v, 0, {k: t.clone() for k, t in m.state_dict().items()}
        else:
            w += 1
            if w >= patience: break
    m.load_state_dict(st); return best
def robust_margin_loss(m, idx, y, B):
    S0, S1 = m.scores(); rows = cert.worst_active(m, idx.cpu().numpy(), y.cpu().numpy(), B, S1); dev = S0.device
    ts = torch.as_tensor([a for a, _, _ in rows], device=dev); cs = torch.as_tensor([b for _, b, _ in rows], device=dev); ys = y[idx][ts]; ir, nd, wt = [], [], []
    for r, (_, _, st) in enumerate(rows):
        if len(st): ir.append(np.full(len(st), r)); nd.append(st); wt.append(np.full(len(st), 1.0 / len(st), dtype=np.float32))
    mean = torch.zeros(len(rows), device=dev)
    if ir:
        I = torch.as_tensor(np.concatenate(ir), device=dev); N = torch.as_tensor(np.concatenate(nd), device=dev); W = torch.as_tensor(np.concatenate(wt), device=dev); mean = mean.index_add(0, I, (S1[N, ys[I]] - S1[N, cs[I]]) * W)
    mg = S0[idx][ts, ys] - S0[idx][ts, cs] + mean; pos, seen = [], {}
    for t, _, _ in rows: pos.append(seen.get(t, 0)); seen[t] = pos[-1] + 1
    return torch.logsumexp(torch.zeros(len(idx), m.c, device=dev).index_put((ts, torch.as_tensor(pos, device=dev)), -mg), 1).mean()
def fit_robust(m, tr, y, cfg, steps=200, batch=256):
    idx = tr.nonzero().flatten(); opt = torch.optim.Adam(m.parameters(), lr=cfg["rlr"], weight_decay=cfg["wd"]); gen = torch.Generator().manual_seed(0)
    for _ in range(steps):
        sub = idx if len(idx) <= batch else idx[torch.randperm(len(idx), generator=gen)[:batch].to(idx.device)]
        m.train(); opt.zero_grad(); robust_margin_loss(m, sub, y, cfg["B"]).backward(); opt.step()
    m.eval()
def ev(m, mask, y, ds, s):
    rng = np.random.default_rng(s); n = mask.nonzero().flatten().cpu().numpy()
    if len(n) > 2000: n = rng.choice(n, 2000, replace=False)
    m.eval(); p, r = cert.radii(m, n, BUD); ok = p == y[n].cpu().numpy()
    with torch.no_grad():
        S0, S1 = m.scores(); share = float(S1.std() / (S0.std() + S1.std())); met = gam.metric(ds, m()[mask], y[mask])
    return dict(acc=float(ok.mean()), metric=met, cert={str(b): float((ok & (r >= b)).mean()) for b in BUD}, graph_share=share)
GRID = dict(lr=[0.01, 0.001], wd=[5e-4, 1e-5], hidden=[64, 256], drop=[0.2, 0.5])
def smooth_grid(g, c, ms, kind, base):
    deg = g.edge_index.size(1) / g.num_nodes; out = []
    for f, pd in itertools.product([0.25, 0.5, 1.0, 2.0], [0.4, 0.6, 0.8, 0.95]):
        if len(out) >= 8 and False: break
        out.append(dict(base, pa=min(f * deg / g.num_nodes, 0.05), pd=pd, kind=kind))
    random.Random(0).shuffle(out); return out[:8]
def smooth_eval(g, c, ms, s, cfg, N, seed):
    n = g.num_nodes; ei = g.edge_index.cpu().numpy(); und = ei[:, ei[0] < ei[1]]; rng = np.random.default_rng(seed); torch.manual_seed(seed)
    m = gam.Conv2(g.x.size(1), c, kind=cfg["kind"], hidden=cfg["hidden"], drop=cfg["drop"]).cuda(); opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]); tr, va, te = (x[:, s] for x in ms)
    noisy = lambda: torch.as_tensor(smooth.sample_edges(und, n, cfg["pa"], cfg["pd"], rng), device="cuda", dtype=torch.long); best, st, w = -1, None, 0
    for _ in range(600):
        m.train(); opt.zero_grad(); F.cross_entropy(m(g.x, noisy())[tr], g.y[tr]).backward(); opt.step(); m.eval()
        with torch.no_grad(): v = float((m(g.x, noisy())[va].argmax(1) == g.y[va]).float().mean())
        if v > best: best, w, st = v, 0, {k: t.clone() for k, t in m.state_dict().items()}
        else:
            w += 1
            if w >= 150: break
    m.load_state_dict(st); m.eval(); votes = torch.zeros(n, c, dtype=torch.int32, device="cuda"); t0 = time.time()
    with torch.no_grad():
        for _ in range(N): votes[torch.arange(n, device="cuda"), m(g.x, noisy()).argmax(1)] += 1
    sec = time.time() - t0; table = smooth.threshold_table(cfg["pa"], cfg["pd"], BUD); cap = smooth.lower_pA(N, N)
    def e(mask):
        v = votes[mask].cpu().numpy(); yy = g.y[mask].cpu().numpy(); top = v.argmax(1); lo = np.array([smooth.lower_pA(int(a), N) for a in v.max(1)]); ok = top == yy
        return dict(acc=float(ok.mean()), cert={str(b): float((ok & (lo >= table[b])).mean()) for b in BUD}, sample_cap_blocks={str(b): bool(table[b] > cap) for b in BUD})
    return e(va), e(te), sec / max(int(te.sum()), 1)
T3 = dict(Cora=dict(lr=.18, wd=2.4e-4, R=7, theta=2.2, M=100), CiteSeer=dict(lr=.51, wd=3.1e-4, R=5, theta=2.5, M=100), PubMed=dict(lr=.027, wd=4e-4, R=5, theta=3.8, M=100), Cornell=dict(lr=.073, wd=3.7e-4, R=6, theta=2.9, M=100))
def main():
    name, ds = sys.argv[1:3]; path = os.path.join(gam.ROOT, "results/s04", f"{ds}__{name}.json"); os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path): print("skip", path); return
    t0 = time.time()
    try:
        g, ms = eh.splits(ds); g = g.to("cuda"); ms = [x.to("cuda") for x in ms]; c = int(g.y.max()) + 1; y = g.y; tr0, va0 = ms[0][:, 0], ms[1][:, 0]; sp = list(range(3, ms[0].size(1))); out = dict(ds=ds, model=name, label="supplementary, post-review")
        if name == "certmlp":
            grid = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]; trials = []
            for cfg in grid:
                torch.manual_seed(0); m = CertMLP(g, c, **cfg).cuda(); trials.append(dict(cfg=cfg, val=fit_plain(m, tr0, va0, y, ds, cfg))); del m
            cfg = max(trials, key=lambda t: t["val"])["cfg"]; rows = []
            for s in sp:
                torch.manual_seed(s); m = CertMLP(g, c, **cfg).cuda(); fit_plain(m, ms[0][:, s], ms[1][:, s], y, ds, cfg); rows.append(dict(split=s, **ev(m, ms[2][:, s], y, ds, s)))
            out.update(cfg=cfg, rows=rows, params=sum(p.numel() for p in m.parameters()), status="ok")
        elif name == "certmlprob":
            base = json.load(open(os.path.join(gam.ROOT, "results/s04", f"{ds}__certmlp.json")))["cfg"]; trials = []
            for B, rlr in itertools.product([1, 2], [1e-3, 1e-4]):
                cfg = dict(base, B=B, rlr=rlr); torch.manual_seed(0); m = CertMLP(g, c, **cfg).cuda(); fit_plain(m, tr0, va0, y, ds, cfg); fit_robust(m, tr0, y, cfg); e = ev(m, va0, y, ds, 0); trials.append(dict(cfg=cfg, val=.5 * e["acc"] + .5 * e["cert"]["1"])); del m
            cfg = max(trials, key=lambda t: t["val"])["cfg"]; rows = []
            for s in sp:
                torch.manual_seed(s); m = CertMLP(g, c, **cfg).cuda(); fit_plain(m, ms[0][:, s], ms[1][:, s], y, ds, cfg); fit_robust(m, ms[0][:, s], y, cfg); rows.append(dict(split=s, **ev(m, ms[2][:, s], y, ds, s)))
            out.update(cfg=cfg, rows=rows, status="ok")
        elif name == "smoothsage":
            base = json.load(open(os.path.join(gam.ROOT, "results/s02", f"{ds}__sage.json")))["cfg"]; trials = []
            for cfg in smooth_grid(g, c, ms, "sage", base):
                v, _, _ = smooth_eval(g, c, ms, 0, cfg, 500, 0); trials.append(dict(cfg=cfg, val=.5 * v["acc"] + .5 * v["cert"]["1"]))
            cfg = max(trials, key=lambda t: t["val"])["cfg"]; rows = []
            for s in sp:
                _, t, sec = smooth_eval(g, c, ms, s, cfg, 20000, s); rows.append(dict(split=s, metric=t["acc"], sec_per_node=sec, **t))
            out.update(cfg=cfg, rows=rows, status="ok", trials=trials)
        elif name == "gnam_t3":
            cfg = T3[ds]; prev = json.load(open(os.path.join(gam.ROOT, "results/s02", f"{ds}__gnamrff.json"))); torch.manual_seed(0); m = gam.GNAMRFF(gam.laplacian_powers(g, cfg["R"]), c, M=cfg["M"], theta=cfg["theta"]).cuda(); val = run.fit_sgd(ds, m, g, tr0, va0, cfg); out.update(cfg=cfg, val_t3=val, val_s02=prev["best_val"]); del m
            if val > prev["best_val"]:
                rows = []
                for s in sp:
                    torch.manual_seed(s); m = gam.GNAMRFF(gam.laplacian_powers(g, cfg["R"]), c, M=cfg["M"], theta=cfg["theta"]).cuda(); run.fit_sgd(ds, m, g, ms[0][:, s], ms[1][:, s], cfg); m.eval()
                    with torch.no_grad(): rows.append(dict(split=s, **eh.scores(ds, m(g.x, g.edge_index), ms[2][:, s], y)))
                    del m
                out.update(rows=rows)
            out.update(status="ok")
        out["sec"] = time.time() - t0
    except torch.OutOfMemoryError: out = dict(ds=ds, model=name, status="OOM")
    except Exception: out = dict(ds=ds, model=name, status="error", trace=traceback.format_exc())
    conv = lambda o: o.item() if hasattr(o, "item") else str(o)
    json.dump(out, open(path, "w"), indent=1, default=conv); print(ds, name, out["status"], round(out.get("sec", 0)), out.get("cfg"), str(out.get("trace", ""))[-300:], flush=True)
if __name__ == "__main__": main()
