"""A1/A2: how nonlinear are trained G-NAMRFF shape functions, as a function of Pi = rho/theta?

Train G-NAMRFF, replace each learned f_{k,c} by its least-squares line on the filtered inputs, and
compare accuracy. Needs no separately trained control, so its optimisation cannot bias the comparison.
Writes one JSON per dataset; finished (theta, seed) entries are reused.
usage: audit_linear.py DATASET [seeds]
"""

import json
import math
import os
import sys

import gam
import run
import torch


@torch.no_grad()
def linearised(m):
    xt = m.filtered()
    N, D = xt.shape
    C = m.w.size(2)
    full, lin = torch.zeros(N, C, device=xt.device), torch.zeros(N, C, device=xt.device)
    ss_res, ss_tot = 0.0, 0.0
    for s in range(0, D, m.chunk):
        xs, ws = xt[:, s : s + m.chunk], m.w[s : s + m.chunk]
        phi = torch.cos(xs.unsqueeze(-1) * m.a.T + m.b) * math.sqrt(2.0 / m.M)
        f = torch.einsum("nkm,kmc->nkc", phi, ws[:, :-1]) + ws[:, -1].unsqueeze(
            0
        )  # N x k x C
        xm, fm = xs.mean(0, keepdim=True), f.mean(0, keepdim=True)
        dx, df = xs - xm, f - fm
        slope = (dx.unsqueeze(-1) * df).sum(0, keepdim=True) / (
            dx.pow(2).sum(0, keepdim=True).unsqueeze(-1) + 1e-30
        )
        fl = fm + slope * dx.unsqueeze(-1)
        ss_res += (f - fl).pow(2).sum().item()
        ss_tot += df.pow(2).sum().item()
        full += f.sum(1)
        lin += fl.sum(1)
    return full, lin, ss_res / max(ss_tot, 1e-30)


def main():
    ds = sys.argv[1]
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    path = os.path.join(gam.ROOT, "results", f"audit_linear_{ds}.json")
    out = json.load(open(path)) if os.path.exists(path) else []
    done = {(r["theta"], r["seed"]) for r in out}
    g, masks = gam.load(ds)
    g = g.to("cuda")
    tr, va, te = (m[:, 0].to("cuda") for m in masks)
    c = int(g.y.max()) + 1
    # Fixed optimiser settings from the authors' script for Cora.
    cfg = dict(lr=0.1, wd=5e-4)
    P = gam.laplacian_powers(g, 7)
    for theta in [0.01, 0.03, 0.1, 0.3, 1.0, 2.2, 8.0]:
        for seed in range(seeds):
            if (theta, seed) in done:
                continue
            torch.manual_seed(seed)
            m = gam.GNAMRFF(P, c, M=100, theta=theta).cuda()
            run.fit_sgd(ds, m, g, tr, va, cfg)
            m.eval()
            full, lin, frac = linearised(m)
            xt = m.filtered().abs()
            s = xt.reshape(-1)[
                torch.randint(0, xt.numel(), (1_000_000,), device=xt.device)
            ]
            rho = torch.quantile(s, 0.999).item()
            out.append(
                dict(
                    ds=ds,
                    theta=theta,
                    seed=seed,
                    rho=rho,
                    Pi=rho / theta,
                    acc=gam.metric(ds, full[te], g.y[te]),
                    acc_lin=gam.metric(ds, lin[te], g.y[te]),
                    val=gam.metric(ds, full[va], g.y[va]),
                    val_lin=gam.metric(ds, lin[va], g.y[va]),
                    nonlinear_frac=frac,
                    agree=float(
                        (full[te].argmax(1) == lin[te].argmax(1)).float().mean()
                    ),
                    alpha=m.alphas.tolist(),
                )
            )
            json.dump(out, open(path, "w"), indent=1)
            print(
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in out[-1].items()
                    if k != "alpha"
                },
                flush=True,
            )


if __name__ == "__main__":
    main()
