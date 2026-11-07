"""S02 (select on validation of split 0) and S03 (evaluate frozen configs on splits 3-9).
usage: exp.py s02|s03 DATASET MODEL. Output per (stage, dataset, model); finished files are skipped."""
import hashlib, itertools, json, os, random, sys, time, traceback
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv
import gam, cert, run, smooth

BUD = [1, 2, 3, 5, 10]
GRAPH_FREE = {"logreg", "mlp", "egogam"}
SGD = {"logreg": dict(lr=[0.1, 0.01], wd=[1e-3, 5e-4, 1e-5]), **{k: run.GRIDS[k] for k in ("mlp", "gcn", "sage", "gat")},
       "h2gcn": dict(lr=[0.01, 0.001], wd=[5e-4, 1e-5], hidden=[64, 256], drop=[0.2, 0.5])}
LARGE = {"PubMed", "roman-empire", "amazon-ratings", "minesweeper", "tolokers"}


def splits(ds):
    """Dataset-provided splits, or 10 fixed random 60/20/20 splits for Planetoid ("-public" keeps the public one)."""
    name = ds.replace("-public", "")
    g, ms = gam.load(name)
    if ds.endswith("-public") or name.lower() not in ("cora", "citeseer", "pubmed"):
        return g, ms
    rng = np.random.default_rng(20261104)
    tr, va, te = [], [], []
    for _ in range(10):
        p = torch.as_tensor(rng.permutation(g.num_nodes))
        a, b = int(0.6 * g.num_nodes), int(0.8 * g.num_nodes)
        for lst, idx in ((tr, p[:a]), (va, p[a:b]), (te, p[b:])):
            m = torch.zeros(g.num_nodes, dtype=torch.bool); m[idx] = True; lst.append(m)
    return g, [torch.stack(tr, 1), torch.stack(va, 1), torch.stack(te, 1)]


class H2GCN(nn.Module):
    """Ego and exact 1-/2-hop neighbourhoods aggregated separately, two rounds, all rounds concatenated (Zhu et al. 2020)."""
    def __init__(self, g, c, hidden=64, drop=0.5, **_):
        super().__init__()
        n, ei = g.num_nodes, g.edge_index.cpu().numpy()
        import scipy.sparse as sp
        A = sp.csr_matrix((np.ones(ei.shape[1], np.float32), ei), shape=(n, n)); A.setdiag(0); A.eliminate_zeros(); A.data[:] = 1
        I = sp.eye(n, format="csr", dtype=np.float32)
        A2 = (A @ A).astype(bool).astype(np.float32).tocsr(); A2 = (A2 - A2.multiply(A + I)).tocsr(); A2.eliminate_zeros()
        def t(M):
            d = np.asarray(M.sum(1)).ravel(); M = sp.diags(1 / np.maximum(d, 1)).astype(np.float32) @ M; M = M.tocoo()
            return torch.sparse_coo_tensor(np.stack([M.row, M.col]), M.data, (n, n)).coalesce().to(g.x.device)
        self.A1, self.A2, self.drop = t(A), t(A2), drop
        self.emb = nn.Linear(g.x.size(1), hidden); self.out = nn.Linear(hidden * 7, c)
    def forward(self, x, ei):
        h0 = F.relu(self.emb(x))
        r1 = torch.cat([torch.sparse.mm(self.A1, h0), torch.sparse.mm(self.A2, h0)], 1)
        r2 = torch.cat([torch.sparse.mm(self.A1, r1), torch.sparse.mm(self.A2, r1)], 1)
        return self.out(F.dropout(torch.cat([h0, r1, r2], 1), self.drop, self.training))


class LogReg(nn.Module):
    def __init__(self, d, c, **_):
        super().__init__(); self.l = nn.Linear(d, c)
    def forward(self, x, ei):
        return self.l(x)


def sgd_model(name, g, c, cfg):
    if name == "logreg": return LogReg(g.x.size(1), c)
    if name == "h2gcn": return H2GCN(g, c, **cfg)
    if name == "gnamrff": return gam.GNAMRFF(gam.laplacian_powers(g, cfg["R"]), c, M=cfg["M"], theta=cfg["theta"])
    return run.build(name, g, c, cfg)


def gnam_grid(ds):
    base = [dict(lr=0.1, wd=5e-4, R=5, theta=2.5, M=100), dict(lr=0.02, wd=5e-4, R=5, theta=2.5, M=100),
            dict(lr=0.1, wd=5e-4, R=3, theta=1.0, M=100), dict(lr=0.02, wd=1e-4, R=7, theta=4.0, M=100)]
    if ds.replace("-public", "") in LARGE:
        return [dict(lr=0.03, wd=5e-4, R=5, theta=2.3, M=40), dict(lr=0.1, wd=5e-4, R=3, theta=1.0, M=40)]
    return base


def scores(ds, logits, mask, y):
    return dict(metric=gam.metric(ds.replace("-public", ""), logits[mask], y[mask]),
                acc=float((logits[mask].argmax(1) == y[mask]).float().mean()))


def fit_ego(m, tr, y, lam):
    opt = torch.optim.LBFGS([m.W0, m.b], lr=1.0, max_iter=300, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-5, tolerance_change=1e-9)
    def cl():
        opt.zero_grad(); L = F.cross_entropy((m.Psi @ m.W0 + m.b)[tr], y[tr]) + lam * (m.W0 ** 2 * m.pen.squeeze(0)).sum(); L.backward(); return L
    opt.step(cl)


def reset(m):
    with torch.no_grad(): m.W0.zero_(), m.W1.zero_(), m.b.zero_()


def cert_eval(m, mask, y, cap=2000, attr=200, seed=0, fuzz=True):
    rng = np.random.default_rng(seed)
    nodes = mask.nonzero().flatten().cpu().numpy()
    if len(nodes) > cap: nodes = rng.choice(nodes, cap, replace=False)
    t0 = time.time(); pred, rad = cert.radii(m, nodes, BUD); per = (time.time() - t0) / max(len(nodes), 1)
    yn = y[nodes].cpu().numpy(); ok = pred == yn
    deg = m.deg[nodes].cpu().numpy()
    from scipy.stats import spearmanr
    rho = spearmanr(rad, deg).statistic if len(set(rad)) > 1 and len(set(deg)) > 1 else 0.0
    out = dict(cert={str(b): float((ok & (rad >= b)).mean()) for b in BUD}, sec_per_node=per, spearman_deg=float(rho), radius_hist=np.bincount(np.searchsorted(BUD, rad, side="right")).tolist())
    a = nodes if len(nodes) <= attr else rng.choice(nodes, attr, replace=False)
    t0 = time.time(); top = cert.top1_stable(m, a, BUD[:3]); out["top1_stable"] = {str(b): float((top >= b).mean()) for b in BUD[:3]}; out["attr_sec_per_node"] = (time.time() - t0) / max(len(a), 1)
    if fuzz:  # H1: real edits on certified nodes, recomputed through an independent path.
        with torch.no_grad(): S0, S1 = m.scores()
        S0, S1 = S0.cpu(), S1.cpu(); start, dst = cert.neighbour_lists(m); broke = tried = 0
        cand = nodes[rad >= 1][:150]
        for i in cand:
            r = rad[np.where(nodes == i)[0][0]]; y0 = pred[np.where(nodes == i)[0][0]]
            nb = set(dst[start[i]:start[i + 1]].tolist())
            for _ in range(40):
                k = rng.integers(1, r + 1); e = set(rng.choice(m.n, k, replace=False).tolist()) - {int(i)}
                cur = sorted(nb ^ e); tried += 1
                broke += int((S0[i] + (S1[cur].mean(0) if cur else 0)).argmax()) != y0
            for c in range(m.c):  # adversarial direction per wrong class: drop best neighbours, add worst nodes
                if c == y0: continue
                s = (S1[:, y0] - S1[:, c]).numpy(); order = np.argsort(s); add = [u for u in order if u not in nb and u != i][:r]
                drop = sorted(nb, key=lambda u: -s[u])
                for p in range(0, min(r, len(drop)) + 1):
                    cur = (nb - set(drop[:p])) | set(add[: r - p]); cur = sorted(cur); tried += 1
                    broke += int((S0[i] + (S1[cur].mean(0) if cur else 0)).argmax()) != y0
        out["fuzz_tried"], out["fuzz_broken"] = int(tried), int(broke)
    return out


def fit_robust(m, tr_idx, y, lam, B, lr, steps=200, batch=512):
    reset(m)
    mask = torch.zeros(m.n, dtype=torch.bool, device=m.Psi.device); mask[tr_idx] = True
    cert.fit(m, mask, y, lam)
    opt = torch.optim.Adam([m.W0, m.W1, m.b], lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(0)
    avg = None
    for it in range(steps):
        for gr in opt.param_groups: gr["lr"] = lr / np.sqrt(1 + it / 20)
        sub = tr_idx if len(tr_idx) <= batch else tr_idx[torch.randperm(len(tr_idx), generator=gen)[:batch].to(tr_idx.device)]
        opt.zero_grad(); obj, _ = cert.robust_objective(m, sub, y, B, lam); obj.backward(); opt.step()
        if it >= steps // 2:
            cur = [t.detach().clone() for t in (m.W0, m.W1, m.b)]
            k = it - steps // 2
            avg = cur if avg is None else [(a * k + c) / (k + 1) for a, c in zip(avg, cur)]
    with torch.no_grad():
        for t, a in zip((m.W0, m.W1, m.b), avg): t.copy_(a)


class SmoothGCN(nn.Module):
    pass


def smooth_eval(g, c, masks, s, cfg, N, seed):
    """Train a GCN on noisy graphs, then certify the majority vote over N sampled graphs."""
    n = g.num_nodes; ei = g.edge_index.cpu().numpy(); und = ei[:, ei[0] < ei[1]]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    model = gam.Conv2(g.x.size(1), c, kind="gcn", hidden=cfg["hidden"], drop=cfg["drop"]).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    tr, va, te = (m[:, s] for m in masks)
    pa, pd = cfg["pa"], cfg["pd"]
    def noisy(): return torch.as_tensor(smooth.sample_edges(und, n, pa, pd, rng), device="cuda", dtype=torch.long)
    best, st, wait = -1, None, 0
    for _ in range(600):
        model.train(); opt.zero_grad(); F.cross_entropy(model(g.x, noisy())[tr], g.y[tr]).backward(); opt.step()
        model.eval()
        with torch.no_grad(): v = float((model(g.x, noisy())[va].argmax(1) == g.y[va]).float().mean())
        if v > best: best, wait, st = v, 0, {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 150: break
    model.load_state_dict(st); model.eval()
    votes = torch.zeros(n, c, dtype=torch.int32, device="cuda"); t0 = time.time()
    with torch.no_grad():
        for _ in range(N):
            p = model(g.x, noisy()).argmax(1); votes[torch.arange(n, device="cuda"), p] += 1
    sec = time.time() - t0
    table = smooth.threshold_table(pa, pd, BUD)
    def ev(mask):
        v = votes[mask].cpu().numpy(); y = g.y[mask].cpu().numpy(); top = v.argmax(1); k = v.max(1)
        lo = np.array([smooth.lower_pA(int(a), N) for a in k]); ok = top == y
        return dict(acc=float(ok.mean()), cert={str(b): float((ok & (lo >= table[b])).mean()) for b in BUD})
    return ev(va), ev(te), sec / max(int(te.sum()), 1)


def main():
    stage, ds, name = sys.argv[1:4]
    path = os.path.join(gam.ROOT, "results", stage, f"{ds}__{name}.json")
    if os.path.exists(path): print("skip", path); return
    os.makedirs(os.path.dirname(path), exist_ok=True); t0 = time.time()
    try:
        g, masks = splits(ds); g = g.to("cuda"); masks = [m.to("cuda") for m in masks]; c = int(g.y.max()) + 1; y = g.y
        sp_list = [0] if stage == "s02" else list(range(3, masks[0].size(1))) or [0]
        if masks[0].size(1) == 1: sp_list = [0]
        if stage == "s03":
            cfgs = [json.load(open(os.path.join(gam.ROOT, "results", "s02", f"{ds}__{name}.json")))["cfg"]]
        out = dict(ds=ds, model=name, stage=stage)
        if name in ("egogam", "certgam", "certrob"):
            base = None
            if name == "certrob":
                base = json.load(open(os.path.join(gam.ROOT, "results", "s02", f"{ds}__certgam.json")))["cfg"]
            if stage == "s02":
                tr, va = masks[0][:, 0], masks[1][:, 0]; trials = []
                if name == "certrob":
                    m = cert.NeighbourGAM(g, c, K=base["K"], theta=base["theta"]).cuda(); idx = tr.nonzero().flatten()
                    for B, lr in itertools.product([1, 2], [0.02, 0.005]):
                        fit_robust(m, idx, y, base["lam"], B, lr); e = cert_eval(m, va, y, cap=1000, attr=2, fuzz=False)
                        with torch.no_grad(): s = scores(ds, m(), va, y)
                        trials.append(dict(cfg=dict(base, B=B, lr=lr), val=0.5 * s["acc"] + 0.5 * e["cert"]["1"], acc=s["acc"], c1=e["cert"]["1"]))
                else:
                    for K, th in itertools.product([1, 2, 4], [1.0, 2.0, 4.0]):
                        m = cert.NeighbourGAM(g, c, K=K, theta=th).cuda()
                        for lam in [1e-1, 1e-2, 1e-3, 1e-4]:
                            reset(m)
                            fit_ego(m, tr, y, lam) if name == "egogam" else cert.fit(m, tr, y, lam)
                            with torch.no_grad(): s = scores(ds, (m.Psi @ m.W0 + m.b) if name == "egogam" else m(), va, y)
                            trials.append(dict(cfg=dict(K=K, theta=th, lam=lam), val=s["metric"]))
                        del m; torch.cuda.empty_cache()
                best = max(trials, key=lambda t: t["val"]); out.update(cfg=best["cfg"], best_val=best["val"], trials=trials, status="ok")
            else:
                cfg = cfgs[0]; m = cert.NeighbourGAM(g, c, K=cfg["K"], theta=cfg["theta"]).cuda(); rows = []
                for s in sp_list:
                    tr, va, te = (x[:, s] for x in masks); reset(m); t1 = time.time()
                    if name == "egogam": fit_ego(m, tr, y, cfg["lam"])
                    elif name == "certgam": cert.fit(m, tr, y, cfg["lam"])
                    elif cfg.get("solver") == "cutting":
                        import alt_solver; reset(m); cert.fit(m, tr, y, cfg["lam"]); alt_solver.fit_cutting(m, tr.nonzero().flatten(), y, cfg["lam"], cfg["B"])
                    else: fit_robust(m, tr.nonzero().flatten(), y, cfg["lam"], cfg["B"], cfg["lr"])
                    tt = time.time() - t1
                    with torch.no_grad(): r = dict(split=s, train_sec=tt, **scores(ds, (m.Psi @ m.W0 + m.b) if name == "egogam" else m(), te, y))
                    if name != "egogam": r.update(cert_eval(m, te, y, seed=s))
                    if name != "egogam":
                        with torch.no_grad(): S0, S1 = m.scores(); r["graph_share"] = float(S1.std() / (S0.std() + S1.std()))
                    rows.append(r); print(ds, name, s, {k: r[k] for k in ("metric", "acc")}, r.get("cert"), flush=True)
                out.update(cfg=cfg, rows=rows, params=2 * m.d * m.k * c + c if name != "egogam" else m.d * m.k * c + c, status="ok")
        elif name == "smoothgcn":
            deg = g.edge_index.size(1) / g.num_nodes
            if stage == "s02":
                gcn = json.load(open(os.path.join(gam.ROOT, "results", "s02", f"{ds}__gcn.json")))["cfg"]; trials = []
                for f, pd in itertools.product([0.5, 2.0], [0.4, 0.6, 0.8]):
                    cfg = dict(gcn, pa=min(f * deg / g.num_nodes, 0.05), pd=pd)
                    v, _, _ = smooth_eval(g, c, masks, 0, cfg, 500, 0)
                    trials.append(dict(cfg=cfg, val=0.5 * v["acc"] + 0.5 * v["cert"]["1"], acc=v["acc"], c1=v["cert"]["1"]))
                best = max(trials, key=lambda t: t["val"]); out.update(cfg=best["cfg"], best_val=best["val"], trials=trials, status="ok")
            else:
                cfg = cfgs[0]; rows = []
                for s in sp_list:
                    _, t, sec = smooth_eval(g, c, masks, s, cfg, 5000, s)
                    rows.append(dict(split=s, acc=t["acc"], metric=t["acc"], cert=t["cert"], sec_per_node=sec)); print(ds, name, s, rows[-1], flush=True)
                out.update(cfg=cfg, rows=rows, status="ok")
        else:
            if stage == "s02":
                if name == "gnamrff": grid = gnam_grid(ds)
                else:
                    grid = [dict(zip(SGD[name], v)) for v in itertools.product(*SGD[name].values())]
                    random.Random(0).shuffle(grid); grid = grid[:20]
                trials = []
                for cfg in grid:
                    torch.manual_seed(0); m = sgd_model(name, g, c, cfg).cuda()
                    trials.append(dict(cfg=cfg, val=run.fit_sgd(ds.replace("-public", ""), m, g, masks[0][:, 0], masks[1][:, 0], cfg))); del m; torch.cuda.empty_cache()
                best = max(trials, key=lambda t: t["val"]); out.update(cfg=best["cfg"], best_val=best["val"], trials=trials, status="ok")
            else:
                cfg = cfgs[0]; rows = []
                for s in (sp_list if masks[0].size(1) > 1 else range(10)):  # public split: repeat with seeds
                    sp = s if masks[0].size(1) > 1 else 0
                    torch.manual_seed(s); m = sgd_model(name, g, c, cfg).cuda(); t1 = time.time()
                    run.fit_sgd(ds.replace("-public", ""), m, g, masks[0][:, sp], masks[1][:, sp], cfg); tt = time.time() - t1; m.eval()
                    with torch.no_grad(): r = dict(split=s, train_sec=tt, **scores(ds, m(g.x, g.edge_index), masks[2][:, sp], y))
                    rows.append(r); print(ds, name, s, r, flush=True); params = sum(p.numel() for p in m.parameters()); del m; torch.cuda.empty_cache()
                out.update(cfg=cfg, rows=rows, params=params, status="ok")
        out.update(sec=time.time() - t0, peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    except torch.OutOfMemoryError:
        out = dict(ds=ds, model=name, stage=stage, status="OOM", sec=time.time() - t0)
    except Exception:
        out = dict(ds=ds, model=name, stage=stage, status="error", trace=traceback.format_exc(), sec=time.time() - t0)
    json.dump(out, open(path, "w"), indent=1); print(ds, name, stage, out["status"], f"{out['sec']:.0f}s", out.get("cfg"), out.get("trace", "")[-400:], flush=True)

if __name__ == "__main__":
    main()
