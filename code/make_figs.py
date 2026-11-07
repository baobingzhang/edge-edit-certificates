"""Figures and supplementary tables mined from results/ (no new experiments). Run after make_paper.py.

Every number drawn or printed here is read from results/; nothing is typed by hand. Macros used by the captions
and the supplementary text go to paper/gen/macros_figs.tex. Missing inputs are listed, never filled in.
"""

import json
import os

import gam
import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

R, P = os.path.join(gam.ROOT, "results"), os.path.join(gam.ROOT, "paper")
G, F = os.path.join(P, "gen"), os.path.join(P, "figs")
ALL = "Cora CiteSeer PubMed Texas Wisconsin Cornell Actor roman-empire amazon-ratings minesweeper tolokers".split()
NAME = {
    "roman-empire": "Roman",
    "amazon-ratings": "Amazon",
    "minesweeper": "Mines.",
    "tolokers": "Tolok.",
}
nm = lambda d: NAME.get(d, d)
missing, macros = [], {}

# ---------------------------------------------------------------- style
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIX", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": BASE,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.45,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "pdf.fonttype": 42,
    }
)
COLW, TEXW = 3.5, 7.16  # IEEEtran column and text width, inches
# Colour follows the model, in the validated categorical order; a marker is the second channel.
STY = {
    "certgam": ("#2a78d6", "o", "CertGAM"),
    "certrob": ("#eb6834", "s", "CertGAM-R"),
    "shrink": ("#1baf7a", "D", "ShrinkGAM-R"),
    "sage2": ("#6250d6", "^", "SAGE (2-layer)"),
    "sageadv": ("#e87ba4", "v", "SAGE-adv"),
    "sageadv4": ("#e87ba4", "v", r"SAGE-adv$\times$4"),
    "ppnp_exact": ("#c98500", "P", r"$\pi$-PPNP, exact"),
    "ppnp_local": ("#c98500", "P", r"$\pi$-PPNP, local"),
    "smoothgcn": ("#008300", "X", "Smooth-GCN"),
    "smoothsage": ("#e34948", "*", "Smooth-SAGE"),
}
DIV = LinearSegmentedColormap.from_list(  # red (below the graph-free reference) - grey - blue (above)
    "div", ["#7a1f1f", "#c23b3a", "#ef9b97", "#f0efec", "#9ec5f4", "#2a78d6", "#104281"]
)


def save(fig, name):
    fig.savefig(os.path.join(F, name))
    plt.close(fig)


# ---------------------------------------------------------------- data
def J(*p):
    f = os.path.join(R, *p)
    return json.load(open(f)) if os.path.exists(f) else None


def okj(*p):
    x = J(*p)
    if not x or x.get("status") != "ok":
        missing.append(f"{'/'.join(p)}: {(x or {}).get('status', 'absent')}")
        return None
    return x


AN, V = J("analysis.json"), J("verdict.json")
T, V5, V7, V8 = V["table"], J("verdict5.json"), J("verdict7.json"), J("verdict8.json")
SEL = {d: AN[d]["selected"] for d in ALL}
REF = {
    d: AN[d]["by_ref"][SEL[d]]["acc"] for d in ALL
}  # validation-selected graph-free reference
REFS = {
    d: {r["split"]: r["acc"] for r in okj("s03", f"{d}__{SEL[d]}.json")["rows"]}
    for d in ALL
}
TRIV = [d for d in ALL if V5["table"][d]["trivial"]]
NT = [d for d in ALL if d not in TRIV]
for d in ALL:  # the reference used here must be the one used in the paper
    assert abs(np.mean(list(REFS[d].values())) - REF[d]) < 1e-9, d


def rows_of(m, d):
    """Per-split rows (list of dicts with 'split', 'acc', 'c1', 'cert') for certified model m on dataset d."""
    if m in ("certgam", "certrob", "smoothgcn"):
        x = okj("s03", f"{d}__{m}.json")
        return x and [
            dict(split=r["split"], acc=r["acc"], c1=r["cert"]["1"], cert=r["cert"], r=r)
            for r in x["rows"]
        ]
    if m == "smoothsage":
        x = okj("s04", f"{d}__smoothsage.json")
        return x and [
            dict(split=r["split"], acc=r["acc"], c1=r["cert"]["1"], cert=r["cert"], r=r)
            for r in x["rows"]
        ]
    if m in ("shrink", "shrink0"):
        x = okj("s07", f"{d}__shrink.json")
        t = "chosen" if m == "shrink" else "kappa0"
        return x and [
            dict(split=r["split"], acc=r["acc"], c1=r["cert"]["1"], cert=r["cert"], r=r)
            for r in x["rows"][t]
        ]
    if m == "sage2":
        x = okj("s05", f"{d}__sage2.json")
        return x and [
            dict(split=r["split"], acc=r["acc"], c1=r["cert1"], r=r)
            for r in x["rows"][x["chosen"]]
        ]
    if m in ("sageadv", "sageadv4"):
        p = (
            ("s06", f"{d}__sageadv.json")
            if m == "sageadv"
            else ("s08", f"{d}__sageadv4.json")
        )
        if not os.path.exists(os.path.join(R, *p)):
            return None  # not run on this dataset (by design)
        x = okj(*p)
        return x and [
            dict(split=r["split"], acc=r["acc"], c1=r["cert1"], r=r) for r in x["rows"]
        ]
    if m == "ppnp_exact":
        if not os.path.exists(os.path.join(R, "s09", f"{d}__pippnp_exact.json")):
            return None  # exact certificate only on graphs with at most 8000 nodes (PLAN_v8b)
        x = okj("s09", f"{d}__pippnp_exact.json")
        return x and [
            dict(split=r["split"], acc=r["acc"], c1=r["cert1"], r=r) for r in x["rows"]
        ]
    if m == "ppnp_local":
        x = okj("s09", f"{d}__pippnp.json")
        return x and [
            dict(split=r["split"], acc=r["acc"], c1=r["cert"]["1"], cert=r["cert"], r=r)
            for r in x["rows"]
        ]
    raise KeyError(m)


MODELS = [
    "certgam",
    "certrob",
    "shrink",
    "sage2",
    "sageadv",
    "sageadv4",
    "ppnp_exact",
    "ppnp_local",
    "smoothgcn",
    "smoothsage",
]
RW = {(m, d): rows_of(m, d) for m in MODELS + ["shrink0"] for d in ALL}
mean = lambda m, d, k="c1": (
    None if not RW[m, d] else float(np.mean([r[k] for r in RW[m, d]]))
)
for m in MODELS:
    for d in ALL:
        if RW[m, d]:
            assert sorted(r["split"] for r in RW[m, d]) == list(range(3, 10)), (m, d)
# cross-checks against the numbers the paper prints
for d in ALL:
    assert (
        abs(mean("certrob", d) - T[d]["rob_cert"]) < 1e-9
        and abs(mean("certgam", d) - T[d]["cert"]) < 1e-9
    ), d
    if RW["sage2", d]:
        assert abs(mean("sage2", d) - V5["table"][d]["cert1"]) < 1e-9, d
    if RW["ppnp_exact", d]:
        assert abs(mean("ppnp_exact", d) - V8["table"][d]["e_cert1"]) < 1e-9, d

# ================================================================ Fig. 1: the gap matrix
GROUPS = [
    ("One-layer (ours)", [("certgam", "P"), ("certrob", "P"), ("shrink", "P")]),
    ("Two-layer SAGE", [("sage2", "P"), ("sageadv", "E"), ("sageadv4", "E")]),
    (r"$\pi$-PPNP [ext.]", [("ppnp_exact", "P"), ("ppnp_local", "P")]),
    ("Smoothing", [("smoothgcn", "P"), ("smoothsage", "S")]),
]
rowsM = [(m, tag) for _, g in GROUPS for m, tag in g]
cols = NT + TRIV
M = np.full((len(rowsM) + 1, len(cols)), np.nan)
for i, (m, _) in enumerate(rowsM):
    for j, d in enumerate(cols):
        v = mean(m, d)
        if v is not None:
            M[i, j] = 100 * (v - REF[d])
M[-1] = np.nanmax(
    M[:-1], axis=0
)  # best row per dataset, chosen on test data: optimistic for the graph
above_any = [cols[j] for j in range(len(NT)) if M[-1, j] > 0]
cells_nt = M[:-1, : len(NT)]
macros.update(
    gapCells=str(int(np.isfinite(cells_nt).sum())),
    gapBelow=str(int((cells_nt < 0).sum())),
    gapAboveAny=", ".join(nm(d) for d in above_any) or "none",
    gapAboveAnyN=str(len(above_any)),
    gapNt=str(len(NT)),
)
fig = plt.figure(figsize=(TEXW, 3.05))
ax = fig.add_axes([0.175, 0.12, 0.665, 0.72])
X = np.ma.masked_invalid(M)
norm = TwoSlopeNorm(vmin=-30, vcenter=0, vmax=10)
im = ax.imshow(X, cmap=DIV, norm=norm, aspect="auto", interpolation="nearest")
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if np.isfinite(M[i, j]):
            v = M[i, j]
            dark = v < -17 or v > 6
            ax.text(
                j,
                i,
                f"{v:+.1f}".replace("-", "\u2212"),
                ha="center",
                va="center",
                fontsize=5.6,
                color="white" if dark else INK,
                fontweight="bold" if v > 0 else "normal",
            )
        else:
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor="#f7f7f5",
                    edgecolor=GRID,
                    hatch="////",
                    lw=0,
                )
            )
            ax.text(j, i, "n/r", ha="center", va="center", fontsize=4.8, color=MUTED)
# white rules between cells and between groups
ax.set_xticks(np.arange(-0.5, M.shape[1]), minor=True)
ax.set_yticks(np.arange(-0.5, M.shape[0]), minor=True)
ax.tick_params(which="minor", length=0)
ax.grid(which="minor", color="white", lw=1.2)
k = 0
for _, g in GROUPS:
    k += len(g)
    ax.axhline(k - 0.5, color="white", lw=3.2)
ax.axvline(len(NT) - 0.5, color="white", lw=4)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels([nm(d) for d in cols], fontsize=6.5)
ax.xaxis.tick_top()
ax.tick_params(axis="both", length=0, pad=2)
tagtxt = {"P": "", "E": r"$^{\dagger}$", "S": r"$^{\ddagger}$"}
ax.set_yticks(range(M.shape[0]))
ax.set_yticklabels(
    [STY[m][2] + tagtxt[t] for m, t in rowsM] + ["Best row (test-selected)"],
    fontsize=6.5,
)
ax.get_yticklabels()[-1].set_fontweight("bold")
# group labels on the far left
k = 0
for lab, g in GROUPS:
    ax.text(
        -3.55,
        k + (len(g) - 1) / 2,
        lab,
        ha="right",
        va="center",
        fontsize=6.3,
        color=INK2,
        rotation=0,
        transform=ax.transData,
        style="italic",
    )
    k += len(g)
# reference accuracies below the matrix
for j, d in enumerate(cols):
    ax.text(
        j,
        M.shape[0] - 0.5 + 0.55,
        f"{100*REF[d]:.1f}",
        ha="center",
        va="top",
        fontsize=5.8,
        color=INK2,
    )
ax.text(
    -0.62,
    M.shape[0] - 0.5 + 0.55,
    "Graph-free ref. (%)",
    ha="right",
    va="top",
    fontsize=6.3,
    color=INK2,
)
ax.text(
    (len(NT) + len(cols) - 1) / 2,
    -1.25,
    "trivial",
    ha="center",
    va="bottom",
    fontsize=5.8,
    color=MUTED,
    style="italic",
)
# right-hand count: datasets below the reference among those that ran (non-trivial)
cax = fig.add_axes([0.905, 0.12, 0.011, 0.72])
cb = fig.colorbar(im, cax=cax, extend="both")
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=5.8, length=1.5)
cb.set_label("Certified acc. at $b=1$ minus graph-free ref. (points)", fontsize=6.2)
for i in range(M.shape[0]):
    v = M[i, : len(NT)]
    n = int(np.isfinite(v).sum())
    ax.text(
        len(cols) - 0.35,
        i,
        f"{int((v < 0).sum())}/{n}",
        ha="left",
        va="center",
        fontsize=5.8,
        color=INK2,
    )
ax.text(
    len(cols) - 0.35,
    -0.95,
    "below",
    ha="left",
    va="bottom",
    fontsize=5.8,
    color=MUTED,
    style="italic",
)
save(fig, "fig_gap.pdf")

# ================================================================ Fig. 2: certified accuracy against budget
B = ["1", "2", "3", "5", "10"]
fig, axes = plt.subplots(3, 4, figsize=(TEXW, 4.45), sharex=True)
for ax, d in zip(axes.ravel(), ALL):
    ax.axhspan(REF[d], 1.0, color="#cde2fb", alpha=0.35, lw=0, zorder=0)
    ax.axhline(REF[d], c=INK, lw=0.8, ls=(0, (2, 1.5)), zorder=1)
    for m in ("certgam", "certrob", "ppnp_local", "smoothgcn"):
        if RW[m, d]:
            c, mk, lab = STY[m]
            ax.plot(
                range(5),
                [np.mean([r["cert"][b] for r in RW[m, d]]) for b in B],
                c=c,
                marker=mk,
                ms=2.8,
                lw=1,
                label=lab,
                ls="--" if m == "smoothgcn" else "-",
                zorder=3,
            )
    for m in ("sage2", "ppnp_exact"):
        if RW[m, d]:
            c, mk, lab = STY[m]
            ax.plot(
                [0],
                [mean(m, d)],
                ls="none",
                marker=mk,
                ms=4.2,
                mfc="white" if m == "sage2" else c,
                mec=c,
                mew=0.9,
                label=lab + " ($b=1$)",
                zorder=4,
            )
    ax.set_title(nm(d) + ("  (trivial)" if d in TRIV else ""), fontsize=7.2, pad=2)
    ax.set_xticks(range(5))
    ax.set_xticklabels(B)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
lax = axes.ravel()[-1]
lax.axis("off")
h = [plt.Line2D([], [], c=INK, lw=0.8, ls=(0, (2, 1.5)))]
l = ["Graph-free ref. (val.-selected)"]
seen = set()
for ax in axes.ravel()[:-1]:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in seen:
            seen.add(ll), h.append(hh), l.append(ll)
lax.legend(h, l, loc="center", fontsize=6.3, handlelength=2.2)
fig.supxlabel(
    "Global edit budget $b$ (edge insertions or deletions)", fontsize=7.2, y=0.01
)
fig.supylabel("Certified accuracy", fontsize=7.2, x=0.01)
fig.tight_layout(h_pad=0.6, w_pad=0.6)
save(fig, "fig_curves.pdf")

# ================================================================ Fig. 3: where the certified gain comes from
share = lambda m, d: float(np.mean([r["r"]["graph_share"] for r in RW[m, d]]))
fig, axs = plt.subplots(2, 1, figsize=(COLW, 4.3))
OFF = {d: (4, -2) for d in ALL}  # label offsets in points, chosen by looking at the rendered figure
OFF.update({"Wisconsin": (-4, -2), "Cornell": (4, -6), "CiteSeer": (4, -5)})
OFF_B = {d: (4, -2) for d in ALL}  # panel (b)
OFF_B.update({"Cornell": (-4, -3), "Actor": (4, 2), "Wisconsin": (4, -5)})
for k, ax in enumerate(axs):
    for d in ALL:
        if k == 0:
            x0, x1 = (
                mean("certgam", d, "acc") - REF[d],
                mean("certrob", d, "acc") - REF[d],
            )
        else:
            x0, x1 = share("certgam", d), share("certrob", d)
        y0, y1 = mean("certgam", d) - REF[d], mean("certrob", d) - REF[d]
        x0, x1, y0, y1 = (
            100 * x0 if k == 0 else x0,
            100 * x1 if k == 0 else x1,
            100 * y0,
            100 * y1,
        )
        faded = d in TRIV
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.35,head_width=0.18",
                lw=0.7,
                color=BASE if faded else MUTED,
                shrinkA=2.5,
                shrinkB=2.5,
            ),
            zorder=2,
        )
        ax.plot(
            x0,
            y0,
            "o",
            c=STY["certgam"][0],
            ms=3.2,
            alpha=0.45 if faded else 1,
            zorder=3,
        )
        ax.plot(
            x1,
            y1,
            "s",
            c=STY["certrob"][0],
            ms=3.2,
            alpha=0.45 if faded else 1,
            zorder=3,
        )
        dx, dy = (OFF if k == 0 else OFF_B)[d]
        ax.annotate(
            nm(d),
            (x0, y0),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.6,
            color=MUTED if faded else INK2,
            ha="left" if dx > 0 else "right",
        )
    ax.axhline(0, c=INK, lw=0.7, ls=(0, (2, 1.5)))
    ax.set_ylabel("Certified acc. ($b=1$) $-$ ref. (points)")
axs[0].axvline(0, c=INK, lw=0.7, ls=(0, (2, 1.5)))
xl, yl = axs[0].get_xlim(), axs[0].get_ylim()
axs[0].add_patch(
    Rectangle((0, 0), xl[1], yl[1], color="#cde2fb", alpha=0.45, lw=0, zorder=0)
)
axs[0].set_xlim(xl), axs[0].set_ylim(yl)
axs[0].set_xlabel("Clean accuracy $-$ ref. (points)")
axs[1].set_xlabel(
    r"Neighbour share of score variation, std$(S^1)/($std$(S^0)+$std$(S^1))$"
)
axs[0].set_title(
    "(a) Clean and certified accuracy relative to the graph-free reference",
    fontsize=7,
    loc="left",
)
axs[1].set_title(
    "(b) Certified training lowers the neighbour share",
    fontsize=7,
    loc="left",
)
h = [
    plt.Line2D([], [], ls="none", marker="o", c=STY["certgam"][0]),
    plt.Line2D([], [], ls="none", marker="s", c=STY["certrob"][0]),
]
axs[0].legend(
    h,
    ["CertGAM (ridge)", "CertGAM-R (certified training)"],
    loc="lower left",
    fontsize=6,
)
fig.tight_layout(h_pad=1.0)
save(fig, "fig_mech.pdf")
sh_down = sum(share("certrob", d) < share("certgam", d) for d in ALL)
up = sum(mean("certrob", d) > mean("certgam", d) for d in ALL)
macros.update(mechShareDown=str(sh_down), mechCertUp=str(up))

# ================================================================ Fig. 4: break-even attack rate
useful = V["verdict"]["H8"]["useful"]
pis = lambda c, ce, g: (
    0.0 if c <= g else (1.0 if ce >= g else min(1.0, (c - g) / (c - ce)))
)
fig, axs = plt.subplots(2, 2, figsize=(COLW, 2.9), sharex=True)
for ax, d in zip(axs.ravel(), useful):
    g = T[d]["gf"]
    pp = np.linspace(0, 1, 101)
    ax.axhline(100 * g, c=INK, lw=0.8, ls=(0, (2, 1.5)), zorder=1)
    for m, c_, ce_, pk, ck in (
        ("certgam", "clean", "cert", "pi", "ci"),
        ("certrob", "rob_clean", "rob_cert", "pi_rob", "ci_rob"),
    ):
        c, mk, lab = STY[m]
        cl, ce = T[d][c_], T[d][ce_]
        p = pis(cl, ce, g)
        assert abs(p - T[d][pk]) < 1e-6, (d, m)
        ax.plot(pp, 100 * ((1 - pp) * cl + pp * ce), c=c, lw=1.1, label=lab, zorder=3)
        lo, hi = AN[d][ck]
        yb = 0.05 if m == "certgam" else 0.12
        ax.plot(
            [lo, hi],
            [yb, yb],
            c=c,
            lw=2.4,
            alpha=0.5,
            solid_capstyle="butt",
            zorder=2,
            transform=ax.get_xaxis_transform(),
        )
        if 0 < p < 1:
            ax.plot(
                [p], [100 * g], marker=mk, c=c, ms=3.6, mec="white", mew=0.5, zorder=4
            )
            ax.annotate(
                rf"$\pi^{{\star}}={p:.2f}$",
                (p, 100 * g),
                xytext=(2, 4 if m == "certrob" else -10),
                textcoords="offset points",
                fontsize=5.6,
                color=c,
            )
        else:
            ax.annotate(
                rf"$\pi^{{\star}}={p:.0f}$",
                (0.98 if p == 1 else 0.02, 100 * g),
                xytext=(0, 4 if m == "certrob" else -9),
                textcoords="offset points",
                fontsize=5.6,
                color=c,
                ha="right" if p == 1 else "left",
            )
    ax.set_title(nm(d) + (" (trivial ref.)" if d in TRIV else ""), fontsize=7, pad=2)
    ax.set_xlim(0, 1)
    lo_y = ax.get_ylim()
    ax.set_ylim(lo_y[0] - 0.22 * (lo_y[1] - lo_y[0]), lo_y[1])
for ax in axs[1]:
    ax.set_xlabel(r"Fraction of nodes attacked, $\pi$")
for ax in axs[:, 0]:
    ax.set_ylabel("Accuracy bound (%)")
h, l = axs[0, 0].get_legend_handles_labels()
h.append(plt.Line2D([], [], c=MUTED, lw=2.4, alpha=0.5))
l.append(r"95% bootstrap interval of $\pi^{\star}$")
fig.legend(h, l, loc="upper center", ncol=3, fontsize=5.8, bbox_to_anchor=(0.5, 1.04), handlelength=1.6)
fig.tight_layout(h_pad=0.5, w_pad=0.6)
save(fig, "fig_breakeven.pdf")

# ================================================================ Fig. 5: certified accuracy by degree, per node
EDGES = [0, 1, 2, 3, 4, 6, 10, 20, 10**9]
BLAB = ["0", "1", "2", "3", "4–5", "6–9", "10–19", "≥20"]


def pernode(m, d):
    """Pooled (deg, certified-correct) over splits 3-9 from the per-node records."""
    if not RW[m, d]:
        return None
    deg, cc = [], []
    for r in RW[m, d]:
        x = r["r"]
        ok_ = np.asarray(x["correct"], bool)
        rob = (
            np.asarray(x["robust"], bool)
            if "robust" in x
            else np.asarray(x["rad"]) >= 1
        )
        assert abs((ok_ & rob).mean() - r["c1"]) < 1e-9, (m, d)
        deg.append(np.asarray(x["deg"])), cc.append(ok_ & rob)
    return np.concatenate(deg), np.concatenate(cc)


DEGM = [
    ("shrink0", "CertGAM-R"),
    ("shrink", None),
    ("sage2", None),
    ("ppnp_exact", None),
]
STY["shrink0"] = (STY["certrob"][0], "s", r"CertGAM-R ($\kappa=0$)")
fig = plt.figure(figsize=(TEXW, 4.9))
gso = fig.add_gridspec(3, 3, hspace=0.32, wspace=0.18)
low_share = {}
for k, d in enumerate(NT):
    gi = gso[k // 3, k % 3].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.1)
    a = fig.add_subplot(gi[0])
    b = fig.add_subplot(gi[1], sharex=a)
    ref_deg = pernode("shrink0", d)
    nb = np.histogram(ref_deg[0], EDGES)[0]
    for m, _ in DEGM:
        if m == "shrink" and V7["table"][d]["kappa"] == 0:
            continue  # kappa=0 selected: identical to the CertGAM-R row
        pn = pernode(m, d)
        if pn is None:
            continue
        dg, cc = pn
        idx = np.digitize(dg, EDGES) - 1
        xs, ys = [], []
        for t in range(len(BLAB)):
            s = idx == t
            if s.sum() >= 30:
                xs.append(t), ys.append(100 * cc[s].mean())
        c, mk, lab = STY[m]
        a.plot(xs, ys, c=c, marker=mk, ms=2.8, lw=1, label=lab)
    a.set_ylim(-3, 103)
    a.set_yticks([0, 50, 100])
    a.set_title(nm(d), fontsize=7, pad=1.5)
    plt.setp(a.get_xticklabels(), visible=False)
    b.bar(range(len(BLAB)), 100 * nb / nb.sum(), color=MUTED, width=0.7, lw=0)
    b.set_ylim(0, max(60, 100 * nb.max() / nb.sum() * 1.15))
    b.set_yticks([0, 50])
    b.grid(axis="x", visible=False)
    b.set_xticks(range(len(BLAB)))
    b.set_xticklabels(BLAB, fontsize=5.6)
    low_share[d] = float(nb[:2].sum() / nb.sum())
    if k % 3 == 0:
        a.set_ylabel("Cert. acc. (%)", fontsize=6.5)
        b.set_ylabel("Nodes (%)", fontsize=6)
h, l = [], []
for ax in fig.axes:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in l:
            h.append(hh), l.append(ll)
fig.legend(
    h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 0.985), fontsize=6.5
)
fig.supxlabel("Degree of the test node (number of neighbours)", fontsize=7.2, y=0.035)
save(fig, "fig_degree.pdf")
macros.update(
    degLowShareMax=f"{100*max(low_share.values()):.0f}",
    degLowShareMaxDs=nm(max(low_share, key=low_share.get)),
)

# ================================================================ Fig. 6: the printed stability constant (restyled; same data as make_paper.py)
el2 = J("audit_lemma.json")[1:]
fig, ax = plt.subplots(figsize=(COLW, 1.95))
ax.axhspan(1, 1e3, color="#fbdedd", alpha=0.5, lw=0, zorder=0)
ax.axhline(1, c=INK, lw=0.7, ls=(0, (2, 1.5)))
ax.plot([r["R"] for r in el2], [r["median_ratio"] for r in el2], marker="o", c=STY["smoothsage"][0], ms=3, lw=1.1, label="Median over edge flips")
ax.plot([r["R"] for r in el2], [r["min_ratio"] for r in el2], marker="s", c=STY["certgam"][0], ms=3, lw=1.1, label="Minimum over edge flips")
ax.set_yscale("log")
ax.set_ylim(0.5, max(r["median_ratio"] for r in el2) * 3)
ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1, 2, 5, 10, 20, 50]))
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.text(el2[0]["R"], 1.15, "above 1: violates the printed constant", ha="left", fontsize=5.8, color="#9b2b2a", va="bottom")
ax.set_xlabel("Filter order $R$ (single monomial)")
ax.set_ylabel("True change / printed constant")
ax.legend(fontsize=6, loc="upper center", ncol=2)
save(fig, "fig_lemma.pdf")

# ================================================================ Supplementary figures
# S-a: the main finding split by split
SPLITM = ["certrob", "shrink", "sage2", "ppnp_exact", "ppnp_local"]
fig, axs = plt.subplots(1, len(SPLITM), figsize=(TEXW, 2.5), sharey=True)
rng = np.random.default_rng(0)  # vertical jitter only, for legibility
split_below = {}
for ax, m in zip(axs, SPLITM):
    c, mk, lab = STY[m]
    nb_, nt_ = 0, 0
    for i, d in enumerate(NT):
        if not RW[m, d]:
            continue
        g_ = np.array([100 * (r["c1"] - REFS[d][r["split"]]) for r in RW[m, d]])
        nb_, nt_ = nb_ + int((g_ < 0).sum()), nt_ + len(g_)
        ax.scatter(g_, i + rng.uniform(-0.18, 0.18, len(g_)), s=7, color=c, marker=mk, lw=0, alpha=0.8, zorder=3)
        ax.plot([g_.mean()] * 2, [i - 0.32, i + 0.32], c=INK, lw=1.1, zorder=4)
    split_below[m] = (nb_, nt_)
    ax.axvline(0, c=INK, lw=0.7, ls=(0, (2, 1.5)))
    ax.set_title(f"{lab}\n{nb_}/{nt_} split-dataset pairs below", fontsize=6.5, pad=3)
    ax.grid(axis="y", visible=False)
axs[0].set_yticks(range(len(NT)))
axs[0].set_yticklabels([nm(d) for d in NT])
axs[0].invert_yaxis()
fig.supxlabel("Certified accuracy at $b=1$ minus graph-free reference, same split (points)", fontsize=7, y=0.0)
fig.tight_layout(w_pad=0.4)
save(fig, "supp_split.pdf")
WEB = ["Texas", "Wisconsin", "Cornell"]
exc = [(m, d) for m in SPLITM for d in NT if RW[m, d] for r in RW[m, d] if r["c1"] >= REFS[d][r["split"]]]
ntest = [len(r["r"]["nodes"]) for d in WEB for r in RW["ppnp_exact", d]]  # all WebKB test nodes are certified
macros.update(
    splitExcN=str(len(exc)),
    splitExcWeb=str(sum(d in WEB for _, d in exc)),
    splitExcOther=", ".join(sorted({nm(d) for _, d in exc if d not in WEB})) or "none",
    webTestMin=str(min(ntest)),
    webTestMax=str(max(ntest)),
)
for m in SPLITM:
    macros[f"splitBelow{m.replace('_', '').replace('2', 'Two')}"] = f"{split_below[m][0]} of {split_below[m][1]}"

# S-b: validation landscapes of pi-PPNP (alpha x loss)
LOSS, ALPHA = ["ce", "rce", "cem"], [0.1, 0.5, 0.85]
panels = [(d, "pippnp_exact", "exact") for d in ALL if os.path.exists(os.path.join(R, "s09", f"{d}__pippnp_exact.json"))]
panels += [(d, "pippnp", "local") for d in ALL if J("s09", f"{d}__pippnp.json")["status"] == "ok"]
nc = 6
nr = int(np.ceil(len(panels) / nc))
fig, axs = plt.subplots(nr, nc, figsize=(TEXW, 1.32 * nr + 0.3))
blues = LinearSegmentedColormap.from_list("b", ["#f0efec", "#9ec5f4", "#2a78d6", "#104281"])
for ax, (d, tag, kind) in zip(axs.ravel(), panels):
    x = J("s09", f"{d}__{tag}.json")
    Mv = np.full((3, 3), np.nan)
    for t in x["trials"]:
        Mv[ALPHA.index(t["alpha"]), LOSS.index(t["loss"])] = 100 * t["val"]
    ax.imshow(np.ma.masked_invalid(Mv), cmap=blues, aspect="auto", vmin=np.nanmin(Mv) - 2, vmax=np.nanmax(Mv) + 1)
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    for a_ in range(3):
        for l_ in range(3):
            v = Mv[a_, l_]
            if np.isfinite(v):
                hi_ = v > np.nanmin(Mv) + 0.75 * (np.nanmax(Mv) - np.nanmin(Mv) + 3)
                ax.text(l_, a_, f"{v:.1f}", ha="center", va="center", fontsize=5.2, color="white" if hi_ else INK)
            else:
                ax.text(l_, a_, "—", ha="center", va="center", fontsize=5.2, color=MUTED)
    ca, cl_ = ALPHA.index(x["cfg"]["alpha"]), LOSS.index(x["cfg"]["loss"])
    ax.add_patch(Rectangle((cl_ - 0.5, ca - 0.5), 1, 1, fill=False, ec=STY["certrob"][0], lw=1.4))
    ax.set_xticks(range(3))
    ax.set_xticklabels(["CE", "RCE", "CEM"], fontsize=5.2)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["0.1", "0.5", "0.85"], fontsize=5.2)
    ax.tick_params(length=0, pad=1)
    ax.set_title(f"{nm(d)} ({kind})", fontsize=6.2, pad=2)
for ax in axs.ravel()[len(panels):]:
    ax.axis("off")
fig.supylabel(r"Teleport $\alpha$", fontsize=7, x=0.0)
fig.tight_layout(h_pad=0.4, w_pad=0.5)
save(fig, "supp_ppnp_select.pdf")
macros["selPpnpPanels"] = str(len(panels))

# S-c: kappa landscapes of ShrinkGAM-R (validation, split 0)
KAP = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
fig, axs = plt.subplots(3, 4, figsize=(TEXW, 3.6), sharex=True)
for ax, d in zip(axs.ravel(), ALL):
    x = J("s07", f"{d}__shrink.json")
    tr = sorted(x["trials"], key=lambda t: t["kappa"])
    assert [t["kappa"] for t in tr] == KAP, d
    xs = range(len(KAP))
    ax.plot(xs, [100 * t["acc"] for t in tr], c=INK2, marker="o", ms=2.4, lw=0.9, label="Validation accuracy")
    ax.plot(xs, [100 * t["c1"] for t in tr], c=STY["shrink"][0], marker="D", ms=2.4, lw=0.9, label="Validation cert. acc. ($b=1$)")
    ax.plot(xs, [100 * t["val"] for t in tr], c=STY["certrob"][0], lw=1.6, alpha=0.9, label="Selection score (mean)")
    ax.axvline(KAP.index(x["kappa"]), c=STY["certrob"][0], lw=0.7, ls=(0, (2, 1.5)))
    ax.set_title(nm(d), fontsize=7, pad=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k:g}" for k in KAP], fontsize=5.6)
    ax.ticklabel_format(axis="y", useOffset=False)
    y0_, y1_ = ax.get_ylim()
    if y1_ - y0_ < 2:  # flat curves (Texas): show a 2-point window instead of float noise
        mid = (y0_ + y1_) / 2
        ax.set_ylim(mid - 1, mid + 1)
axs.ravel()[-1].axis("off")
h, l = axs[0, 0].get_legend_handles_labels()
h.append(plt.Line2D([], [], c=STY["certrob"][0], lw=0.7, ls=(0, (2, 1.5))))
l.append(r"Selected $\kappa$")
axs.ravel()[-1].legend(h, l, loc="center", fontsize=6.3)
fig.supxlabel(r"Pseudo-count $\kappa$", fontsize=7, y=0.01)
fig.supylabel("Percent (validation, split 0)", fontsize=7, x=0.01)
fig.tight_layout(h_pad=0.5, w_pad=0.5)
save(fig, "supp_kappa.pdf")

# S-d: soundness checks and tightness of the pi-PPNP local certificate
V7v = V7["table"]
shr_tried = sum(
    r["fuzz_tried"] for d in ALL if J("s07", f"{d}__shrink.json")
    for t in (("chosen",) if J("s07", f"{d}__shrink.json")["kappa"] == 0 else ("chosen", "kappa0"))
    for r in J("s07", f"{d}__shrink.json")["rows"][t]
)  # kappa=0 rows copy the chosen rows when kappa=0 is selected
shr_broken = sum(V7v[d]["fuzz_broken"] for d in ALL if V7v[d].get("status") == "ok")
pl = [d for d in ALL if RW["ppnp_local", d]]
pp_tried = sum(r["r"]["attack_tried"] for d in pl for r in RW["ppnp_local", d])
pp_uns = sum(r["r"]["unsound"] for d in pl for r in RW["ppnp_local", d])
g10 = J("g10.json")
CHK = [
    ("CertGAM, CertGAM-R: real edits (H1)", V["verdict"]["H1"]["tried"], V["verdict"]["H1"]["broken"]),
    ("ShrinkGAM-R: real edits (E8)", shr_tried, shr_broken),
    (r"$\pi$-PPNP local: single-edit attack (E10)", pp_tried, pp_uns),
    (r"$\pi$-PPNP local: exhaustive edit sets (G10)", g10["undirected_edit_sets"], 0),
]
assert g10["status"] == "PASS"
fig, axs = plt.subplots(1, 2, figsize=(TEXW, 2.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
ax = axs[0]
yy = np.arange(len(CHK))[::-1]
ax.barh(yy, [c_[1] for c_ in CHK], color=STY["certgam"][0], height=0.55, lw=0)
ax.set_xscale("log")
ax.set_yticks(yy)
ax.set_yticklabels([c_[0] for c_ in CHK], fontsize=6.3)
for y_, c_ in zip(yy, CHK):
    ax.text(c_[1] * 1.15, y_, f"{c_[1]:,} checked, {c_[2]} broken", va="center", fontsize=6, color=INK2)
ax.set_xlim(1e3, 3e7)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Recomputed edits or edit sets (log scale)")
ax.set_title("(a) No certificate was ever broken", fontsize=7, loc="left")
ax = axs[1]
for d in pl:
    xs_ = [100 * r["r"]["cert1_attacked"] for r in RW["ppnp_local", d]]
    ys_ = [100 * r["r"]["ub1"] for r in RW["ppnp_local", d]]
    ax.scatter(xs_, ys_, s=9, lw=0.3, ec="white", color=STY["ppnp_local"][0], zorder=3)
lim = [0, 100]
ax.plot(lim, lim, c=INK, lw=0.7, ls=(0, (2, 1.5)), zorder=1)
ax.set_xlim(lim), ax.set_ylim(lim)
gaps_ = [round(r["r"]["attack_nodes"] * (r["r"]["ub1"] - r["r"]["cert1_attacked"])) for d in pl for r in RW["ppnp_local", d]]
ax.set_xlabel(r"Local certificate, cert. acc. at $b=1$ (%)")
ax.set_ylabel("Attack upper bound (%)")
ax.set_title(f"(b) $\\pi$-PPNP: certificate vs. attack, {len(gaps_)} splits", fontsize=7, loc="left")
ax.text(4, 92, f"gap $\\leq$ {max(gaps_)} attacked node per split", fontsize=6, color=INK2)
fig.tight_layout(w_pad=1.2)
save(fig, "supp_sound.pdf")
macros.update(soundShrTried=f"{shr_tried:,}", soundPpnpTried=f"{pp_tried:,}", soundPpnpGapMax=str(max(gaps_)),
              soundPpnpSplits=str(len(gaps_)), soundTotal=f"{sum(c_[1] for c_ in CHK[:3]):,}")

# S-e: time to certify one node
fig, ax = plt.subplots(figsize=(COLW, 2.6))
CM = [("certgam", "CertGAM (exact)"), ("certrob", "CertGAM-R (exact)"), ("ppnp_exact", r"$\pi$-PPNP (exact, Woodbury)"),
      ("smoothgcn", "Smooth-GCN (5000 samples)"), ("smoothsage", "Smooth-SAGE (20,000 samples)")]
spn = {}
for k_, (m, lab) in enumerate(CM):
    c, mk, _ = STY[m]
    for i, d in enumerate(ALL):
        if not RW[m, d]:
            continue
        if m == "ppnp_exact":
            v = np.mean([r["r"]["cert_sec"] / len(r["r"]["nodes"]) for r in RW[m, d]])
        else:
            v = np.mean([r["r"]["sec_per_node"] for r in RW[m, d]])
        spn.setdefault(m, []).append(v)
        ax.scatter(v, i + (k_ - 2) * 0.13, color=c, marker=mk, s=11, lw=0, zorder=3, label=lab if i == 0 or (m == "ppnp_exact" and d == "Cora") else None)
ax.set_xscale("log")
ax.set_yticks(range(len(ALL)))
ax.set_yticklabels([nm(d) for d in ALL])
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel("Seconds per certified node (log scale)")
ax.legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
save(fig, "supp_cost.pdf")
macros.update(costGamMed=f"{1e3*np.median(spn['certgam']):.1f}", costSmoothMed=f"{1e3*np.median(spn['smoothgcn']):.1f}",
              costSageSmoothMed=f"{1e3*np.median(spn['smoothsage']):.0f}")

# S-f: distribution of certified radii
RB = ["0", "1", "2", "3", "5", "10"]
RC = ["#d9d8d2", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
fig, ax = plt.subplots(figsize=(TEXW, 2.5))
yl_ = []
for i, d in enumerate(ALL):
    for j_, m in enumerate(("certgam", "certrob")):
        h_ = np.sum([np.pad(r["r"]["radius_hist"], (0, 6 - len(r["r"]["radius_hist"]))) for r in RW[m, d]], axis=0)
        f_ = h_ / h_.sum()
        y_ = 3 * i + j_
        left = 0
        for k_, v in enumerate(f_):
            ax.barh(y_, 100 * v, left=left, color=RC[k_], height=0.82, lw=0, label=RB[k_] if (i, j_) == (0, 0) else None)
            left += 100 * v
        yl_.append((y_, f"{nm(d)}, {'ridge' if m == 'certgam' else 'cert. tr.'}"))
ax.set_yticks([y for y, _ in yl_])
ax.set_yticklabels([t for _, t in yl_], fontsize=5.4)
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.grid(False)
ax.set_xlabel("Share of evaluated test nodes (%)")
ax.legend(title="Certified radius (largest budget certified)", title_fontsize=6.3, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=6.2)
save(fig, "supp_radius.pdf")

# S-g: lambda ablation (restyled from make_paper.py; same data)
AB = J("s03", "ablation.json")
rob = [v for v in AB.values() if v["factor"] == "rob_lam"]
lams = sorted({v["cfg"]["lam"] for v in rob})
abl = {(v["ds"], v["cfg"]["lam"]): v for v in rob}
dsab = ["Cora", "Texas", "Wisconsin", "Cornell", "Actor", "minesweeper"]
PAL6 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MK6 = "osD^vX"
fig, axs = plt.subplots(1, 2, figsize=(TEXW * 0.8, 2.0))
for d, c, mk in zip(dsab, PAL6, MK6):
    v = [abl[(d, l_)] for l_ in lams if (d, l_) in abl]
    axs[0].semilogx([x["cfg"]["lam"] for x in v], [x["share"] for x in v], marker=mk, c=c, ms=2.8, lw=1, label=nm(d))
    axs[1].semilogx([x["cfg"]["lam"] for x in v], [100 * (x["clean"] - x["c1"]) for x in v], marker=mk, c=c, ms=2.8, lw=1)
axs[0].set_xlabel(r"$\lambda$"), axs[0].set_ylabel("Neighbour share")
axs[1].set_xlabel(r"$\lambda$"), axs[1].set_ylabel(r"Clean $-$ certified at $b=1$ (points)")
axs[0].legend(fontsize=5.8, ncol=2)
fig.tight_layout(w_pad=1.5)
save(fig, "supp_lambda.pdf")

# S-h: case study (restyled from make_paper.py; same data)
C = J("s04", "case.json")
fig, axs = plt.subplots(1, 2, figsize=(TEXW * 0.8, 2.15))
for c_, col in zip(C["curves"], ("#2a78d6", "#eb6834", "#1baf7a")):
    axs[0].plot(c_["z"], c_["ego"], c=col, lw=1.1, label=f"f{c_['feature']} ego")
    axs[0].plot(c_["z"], c_["nbr"], c=col, lw=1.1, ls=(0, (3, 1.5)), label=f"f{c_['feature']} neighbour")
axs[0].set_xlabel("Standardized feature value")
axs[0].set_ylabel("Contribution to class score")
axs[0].legend(fontsize=5.4, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
rr_ = C["node"]["rows"]
yy = list(range(len(rr_)))[::-1]
for y_, r in zip(yy, rr_):
    axs[1].plot([r["lo"], r["hi"]], [y_, y_], c=STY["certgam"][0], lw=2.6, alpha=0.35, solid_capstyle="round")
    axs[1].plot([r["value"]], [y_], "o", c=STY["certgam"][0], ms=3.2)
axs[1].axvline(0, c=BASE, lw=0.6)
axs[1].set_yticks(yy)
axs[1].set_yticklabels([f"f{r['feature']}" for r in rr_], fontsize=6)
axs[1].grid(axis="y", visible=False)
axs[1].set_xlabel("Contribution and its exact range at $b=1$")
fig.tight_layout(w_pad=1.5)
save(fig, "supp_case.pdf")

# ================================================================ Supplementary tables
def stable(fname, caption, label, rows, cols_, first="l"):
    s = ("\\begin{table*}[!t]\n\\centering\\scriptsize\n\\caption{" + caption + "}\\label{" + label + "}\n"
         "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n\\begin{tabular}{" + first
         + "c" * len(cols_) + "}\n\\toprule\n & " + " & ".join(cols_) + " \\\\\n\\midrule\n")
    s += "\n".join(" & ".join(map(str, r)) + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"
    open(os.path.join(G, fname), "w").write(s)


g_ = lambda v: f"{v:g}"
e_ = lambda v: f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e")
FMT = {
    "logreg": lambda c: f"{g_(c['lr'])}/{e_(c['wd'])}",
    "mlp": lambda c: f"{g_(c['lr'])}/{e_(c['wd'])}/{c['hidden']}/{g_(c['drop'])}",
    "egogam": lambda c: f"{c['K']}/{g_(c['theta'])}/{e_(c['lam'])}",
    "gnamrff": lambda c: f"{g_(c['lr'])}/{e_(c['wd'])}/{c['R']}/{g_(c['theta'])}/{c['M']}",
    "certrob": lambda c: f"{c['K']}/{g_(c['theta'])}/{e_(c['lam'])}/{c['B']}/" + ("cut" if c.get("solver") == "cutting" else g_(c["lr"])),
    "smoothgcn": lambda c: f"{e_(c['pa'])}/{g_(c['pd'])}",
}
for m in ("gcn", "sage", "gat", "h2gcn"):
    FMT[m] = FMT["mlp"]
FMT["certgam"] = FMT["egogam"]
LAB = [("logreg", "LogReg", "lr/wd"), ("mlp", "MLP", "lr/wd/h/p"), ("egogam", "Ego-GAM", r"$K/\theta/\lambda$"),
       ("gcn", "GCN", "lr/wd/h/p"), ("sage", "SAGE", "lr/wd/h/p"), ("gat", "GAT", "lr/wd/h/p"),
       ("h2gcn", "H2GCN", "lr/wd/h/p"), ("gnamrff", "G-NAMRFF", r"lr/wd/$R/\theta/M$"),
       ("certgam", "CertGAM", r"$K/\theta/\lambda$"), ("certrob", "CertGAM-R", r"$K/\theta/\lambda/B$/lr"),
       ("smoothgcn", "Smooth-GCN", "$p_+/p_-$")]
rows_ = []
for m, lab, key in LAB:
    r_ = [f"{lab} ({key})"]
    for d in ALL:
        x = J("s03", f"{d}__{m}.json")
        r_.append(FMT[m](x["cfg"]) if x and x.get("status") == "ok" else "--")
    rows_.append(r_)
rows_.append(["Smooth-SAGE ($p_+/p_-$)"] + [FMT["smoothgcn"](J("s04", f"{d}__smoothsage.json")["cfg"]) if RW["smoothsage", d] else "--" for d in ALL])
rows_.append(["SAGE 2-layer (variant)"] + [{"sage": "plain", "sageaug": "aug"}[J("s05", f"{d}__sage2.json")["chosen"]] if RW["sage2", d] else "--" for d in ALL])
rows_.append([r"ShrinkGAM-R ($\kappa$)"] + [g_(J("s07", f"{d}__shrink.json")["kappa"]) if RW["shrink", d] else "--" for d in ALL])
for tag, lab in (("pippnp_exact", "exact"), ("pippnp", "local")):
    rows_.append([rf"$\pi$-PPNP {lab} ($\alpha$/loss)"] + [
        (lambda x: f"{g_(x['cfg']['alpha'])}/{x['cfg']['loss'].upper()}" if x and x.get("status") == "ok" else "--")(J("s09", f"{d}__{tag}.json"))
        for d in ALL])
stable("supp_cfg.tex",
       "Configurations selected on the validation set of split 0 and used on splits 3--9, read from the result files. "
       "lr: learning rate; wd: weight decay; h: hidden units; p: dropout; $K,\\theta,\\lambda$: number of eigenfunctions, RBF "
       "bandwidth and penalty of the shape functions; $B$: training budget of certified training; ``cut'': cutting-plane solver; $R,M$: filter "
       "order, kernel bandwidth and number of random features of G-NAMRFF; $p_+/p_-$: edge insertion/deletion probabilities of smoothing. "
       "``--'': not run or failed (H2GCN exceeded the time limit on tolokers; $\\pi$-PPNP ran out of memory on roman-empire; "
       "the exact $\\pi$-PPNP certificate and SAGE-adv were run only on the graphs listed in the main text).",
       "tab:cfg", rows_, [nm(d) for d in ALL])

TM = [("logreg", "LogReg"), ("mlp", "MLP"), ("egogam", "Ego-GAM"), ("gcn", "GCN"), ("sage", "SAGE"), ("gat", "GAT"),
      ("h2gcn", "H2GCN"), ("gnamrff", "G-NAMRFF"), ("certgam", "CertGAM"), ("certrob", "CertGAM-R"), ("smoothgcn", "Smooth-GCN")]
def tcell(m, d, k):
    x = J("s03", f"{d}__{m}.json")
    if not x or x.get("status") != "ok" or x.get(k) is None:
        return "--"
    v = x[k]
    return (f"{v/60:.1f}" if v >= 60 else f"{v/60:.2f}") if k == "sec" else f"{v:.2f}"
stable("supp_cost.tex",
       "Compute of the confirmatory runs (splits 3--9, one GPU with 8\\,GB): wall-clock minutes for the whole run, including "
       "certification where applicable (top), and peak GPU memory in GB (bottom), as recorded in the result files.",
       "tab:cost",
       [[f"{lab}, min"] + [tcell(m, d, "sec") for d in ALL] for m, lab in TM]
       + [["\\midrule " + f"{TM[0][1]}, GB"] + [tcell(TM[0][0], d, "peak_gb") for d in ALL]]
       + [[f"{lab}, GB"] + [tcell(m, d, "peak_gb") for d in ALL] for m, lab in TM[1:]],
       [nm(d) for d in ALL])
gn = [J("s03", f"{d}__gnamrff.json")["sec"] for d in ALL if J("s03", f"{d}__gnamrff.json").get("status") == "ok"]
cg = [J("s03", f"{d}__certgam.json")["sec"] for d in ALL]
macros.update(costGnamMed=f"{np.median(gn)/60:.0f}", costCertgamMed=f"{np.median(cg)/60:.1f}")

# ---------------------------------------------------------------- write macros
with open(os.path.join(G, "macros_figs.tex"), "w") as f:
    for k_, v_ in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{k_}}}{{{v_}}}\n")
print("macros", macros)
print("missing", sorted(set(missing)))
