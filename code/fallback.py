"""Composite classifier h_t(G): use the graph model if its exact radius r(G) >= t, otherwise the graph-free model.

Certificate for budget b <= t. Let G' be within b edits of G.
 (a) If e = g(G): whenever h_t(G') uses the graph, r(G') >= t >= b, so g(G') = g(G) = e (G lies within b of G').
     Otherwise it outputs e. Hence h_t(G') = e for every G'.
 (b) If r(G) >= t + b: r(G') >= t, so the graph model stays selected and g(G') = g(G).
"""
import numpy as np

def evaluate(pred_g, rad, pred_e, y, t, budgets):
    use_graph = rad >= t
    out = np.where(use_graph, pred_g, pred_e)
    res = dict(clean=float((out == y).mean()), graph_used=float(use_graph.mean()))
    for b in budgets:
        if b > t:
            continue
        cert = (pred_e == pred_g) | (rad >= t + b)
        # (a) certifies e=g; (b) certifies g. In both cases the certified output is `out`.
        res[f"c{b}"] = float(((out == y) & cert).mean())
    return res
