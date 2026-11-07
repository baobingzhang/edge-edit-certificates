"""One tuned run: pick a config on the validation metric of split 0, then evaluate on all splits.

Results are written per (phase, dataset, model), so an interrupted queue resumes where it stopped.
usage: run.py PHASE DATASET MODEL [n_splits] [n_configs]
"""

import itertools
import json
import os
import random
import sys
import time
import traceback

import gam
import numpy as np
import torch
import torch.nn.functional as F

GRIDS = {
    "mlp": dict(lr=[0.01, 0.001], wd=[5e-4, 1e-5], hidden=[64, 256], drop=[0.2, 0.5]),
    "gcn": dict(lr=[0.01, 0.001], wd=[5e-4, 1e-5], hidden=[64, 256], drop=[0.2, 0.5]),
    "sage": dict(lr=[0.01, 0.001], wd=[5e-4, 1e-5], hidden=[64, 256], drop=[0.2, 0.5]),
    "gat": dict(lr=[0.01, 0.001], wd=[5e-4, 1e-5], hidden=[64, 256], drop=[0.2, 0.5]),
    "linfilter": dict(lr=[0.1, 0.01], wd=[5e-4, 1e-5], R=[3, 5, 7]),
    "gnamrff": dict(
        lr=[0.1, 0.01], wd=[5e-4, 1e-5], R=[3, 5, 7], theta=[1.0, 2.0, 4.0], M=[100]
    ),
    "hopgam": dict(
        lam=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
        H=[1, 2],
        K=[1, 2, 4, 8],
        theta=[0.5, 1.0, 2.0, 4.0],
    ),
}
EPOCHS, PATIENCE = 1000, 200


def build(name, g, c, cfg):
    d = g.x.size(1)
    if name == "mlp":
        return gam.MLP(d, c, **cfg)
    if name in ("gcn", "sage", "gat"):
        return gam.Conv2(d, c, kind=name, **cfg)
    if name == "linfilter":
        return gam.LinFilter(gam.laplacian_powers(g, cfg["R"]), c)
    if name == "gnamrff":
        return gam.GNAMRFF(
            gam.laplacian_powers(g, cfg["R"]), c, M=cfg["M"], theta=cfg["theta"]
        )
    raise KeyError(name)


def fit_sgd(ds, model, g, tr, va, cfg):
    # G-NAMRFF's validation accuracy recovers only after ~600 epochs; the authors train without early stopping.
    patience = EPOCHS if isinstance(model, gam.GNAMRFF) else PATIENCE
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best, state, wait = -1, None, 0
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(g.x, g.edge_index)[tr], g.y[tr]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = gam.metric(ds, model(g.x, g.edge_index)[va], g.y[va])
        if v > best:
            best, wait = v, 0
            state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    return best


def fit_convex(ds, model, g, tr, va, cfg):
    """L-BFGS on the strictly convex ridge objective. No early stopping: the optimum is the model."""
    opt = torch.optim.LBFGS(
        [model.W, model.bias],
        lr=1.0,
        max_iter=300,
        history_size=20,
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )
    y, lam = g.y[tr], cfg["lam"]

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(model()[tr], y) + lam * model.penalty()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return gam.metric(ds, model()[va], g.y[va])


def one(ds, name, g, c, masks, s, cfg, seed, cache=None):
    torch.manual_seed(seed)
    tr, va, te = (m[:, s % m.size(1)] for m in masks)
    if name == "hopgam":
        key = (cfg["H"], cfg["K"], cfg["theta"])
        if cache is not None and cache.get("key") == key:
            m = cache["m"]
            with torch.no_grad():
                m.W.zero_(), m.bias.zero_()
        else:
            if cache is not None:
                cache.clear()
                torch.cuda.empty_cache()
            m = gam.HopGAM(g, c, H=cfg["H"], K=cfg["K"], theta=cfg["theta"]).cuda()
            if cache is not None:
                cache.update(key=key, m=m)
        val = fit_convex(ds, m, g, tr, va, cfg)
    else:
        m = build(name, g, c, cfg).cuda()
        val = fit_sgd(ds, m, g, tr, va, cfg)
    m.eval()
    with torch.no_grad():
        logits = m(g.x, g.edge_index) if name != "hopgam" else m()
        return val, gam.metric(ds, logits[te], g.y[te])


def main():
    phase, ds, name = sys.argv[1:4]
    n_splits = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    n_cfg = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    path = os.path.join(gam.ROOT, "results", phase, f"{ds}__{name}.json")
    if os.path.exists(path):
        print("skip", path)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.time()
    try:
        g, masks = gam.load(ds)
        g = g.to("cuda")
        masks = [m.to("cuda") for m in masks]
        c = int(g.y.max()) + 1
        grid = [
            dict(zip(GRIDS[name], v)) for v in itertools.product(*GRIDS[name].values())
        ]
        random.Random(0).shuffle(grid)
        grid = grid[:n_cfg]
        cache, best, trials = {}, (-1, None), []
        for cfg in grid:
            v, _ = one(ds, name, g, c, masks, 0, cfg, 0, cache)
            trials.append(dict(cfg=cfg, val=v))
            if v > best[0]:
                best = (v, cfg)
        cfg, vals, tests = best[1], [], []
        for s in range(n_splits):
            v, t = one(ds, name, g, c, masks, s, cfg, s, cache)
            vals.append(v)
            tests.append(t)
        res = dict(
            ds=ds,
            model=name,
            cfg=cfg,
            test=tests,
            val=vals,
            mean=float(np.mean(tests)),
            std=float(np.std(tests)),
            n_cfg=len(grid),
            trials=trials,
            params=None,
            sec=time.time() - t0,
            peak_gb=torch.cuda.max_memory_allocated() / 1e9,
            status="ok",
        )
    except torch.OutOfMemoryError:
        res = dict(ds=ds, model=name, status="OOM", sec=time.time() - t0)
    except Exception:
        res = dict(
            ds=ds,
            model=name,
            status="error",
            trace=traceback.format_exc(),
            sec=time.time() - t0,
        )
    json.dump(res, open(path, "w"), indent=1)
    print(
        ds,
        name,
        res["status"],
        res.get("mean"),
        res.get("std"),
        f"{res['sec']:.0f}s",
        res.get("cfg"),
    )


if __name__ == "__main__":
    main()
