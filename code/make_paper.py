"""Generate every table, figure and number in the manuscript from results/. Nothing here is typed by hand.
Missing results are printed as "--" and listed in gen/missing.txt, never filled in."""

import json
import os

import exp
import gam
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, P = os.path.join(gam.ROOT, "results"), os.path.join(gam.ROOT, "paper")
G, F = os.path.join(P, "gen"), os.path.join(P, "figs")
os.makedirs(G, exist_ok=True)
os.makedirs(F, exist_ok=True)
ALL = "Cora CiteSeer PubMed Texas Wisconsin Cornell Actor roman-empire amazon-ratings minesweeper tolokers".split()
NAME = {
    "roman-empire": "Roman",
    "amazon-ratings": "Amazon",
    "minesweeper": "Mines.",
    "tolokers": "Tolok.",
}
MODELS = [
    ("logreg", "LogReg"),
    ("mlp", "MLP"),
    ("egogam", "Ego-GAM"),
    ("gcn", "GCN"),
    ("sage", "SAGE"),
    ("gat", "GAT"),
    ("h2gcn", "H2GCN"),
    ("gnamrff", "G-NAMRFF"),
    ("certgam", "CertGAM"),
    ("certrob", "CertGAM-R"),
    ("smoothgcn", "Smooth-GCN"),
]
missing = []


def load(ds, m, stage="s03"):
    p = os.path.join(R, stage, f"{ds}__{m}.json")
    d = json.load(open(p)) if os.path.exists(p) else None
    if d is None or d.get("status") != "ok":
        missing.append(f"{ds} {m} {stage}: {(d or {}).get('status', 'absent')}")
        return None
    return d


def ms(d, f=lambda r: r["acc"]):
    if d is None:
        return None
    v = [f(r) for r in d["rows"]]
    return float(np.mean(v)), float(np.std(v))


def pct(x):
    return "--" if x is None else f"{100*x[0]:.1f}$\\pm${100*x[1]:.1f}"


macros = {}
a = json.load(open(os.path.join(R, "audit_base.json")))
ratios = [r["ratio"] for r in a["lemma"] if r["R"] >= 2]
macros.update(
    auditMinRatio=f"{min(ratios):.0f}",
    auditMaxRatio=f"{max(ratios):.0f}",
    auditCodeParams=f"{a['params']['Cora']['code_actual']:,}",
    auditPaperParams=f"{a['params']['Cora']['paper_formula']+1:,}",
    auditGcnParams=f"{a['params']['Cora']['gcn16']:,}",
)
el = [r for r in json.load(open(os.path.join(R, "audit_lemma.json"))) if r["R"] >= 2]
macros.update(
    auditEditMinRatio=f"{min(r['min_ratio'] for r in el):.1f}",
    auditEditMedLow=f"{min(r['median_ratio'] for r in el):.0f}",
    auditEditMedHigh=f"{max(r['median_ratio'] for r in el):.0f}",
)
rep = json.load(open(os.path.join(R, "repro", "cora_table3.json")))
macros.update(
    reproMean=f"{100*rep['mean']:.1f}", reproPaper=f"{100*rep['paper_mean']:.2f}"
)
lo = [
    float(l.split(":")[1])
    for l in open(os.path.join(gam.ROOT, "logs", "s04.log"))
    if "Best Test Accuracy" in l
]  # author script, lr 0.018, seeds 0-4
if len(lo) != 5:
    missing.append(f"author lr=0.018 runs: {len(lo)}/5")
macros.update(
    reproLowLr=f"{100*np.mean(lo):.1f}" if lo else "--", reproLowLrN=str(len(lo))
)
al = [
    x
    for x in json.load(open(os.path.join(R, "audit_linear_Cora.json")))
    if x["theta"] >= 1
]
macros.update(
    auditNonlinLow=f"{min(x['nonlinear_frac'] for x in al):.2f}",
    auditNonlinHigh=f"{max(x['nonlinear_frac'] for x in al):.2f}",
)


# Tables
def table(fname, caption, label, rows, cols, resize=True):
    s = (
        "\\begin{table*}[!t]\n\\centering\\scriptsize\n\\caption{"
        + caption
        + "}\\label{"
        + label
        + "}\n\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n\\begin{tabular}{l"
        + "c" * len(cols)
        + "}\n\\toprule\n & "
        + " & ".join(cols)
        + " \\\\\n\\midrule\n"
    )
    s += (
        "\n".join(" & ".join(map(str, r)) + " \\\\" for r in rows)
        + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"
    )
    open(os.path.join(G, fname), "w").write(s)


data = {ds: {m: load(ds, m) for m, _ in MODELS} for ds in ALL}
cols = [NAME.get(d, d) for d in ALL]
table(
    "tab_clean.tex",
    "Test accuracy (\\%, mean$\\pm$std over splits 3--9). AUC-oriented datasets are also reported as accuracy here so that clean and certified columns are comparable.",
    "tab:clean",
    [[lab] + [pct(ms(data[d][m])) for d in ALL] for m, lab in MODELS],
    cols,
)
table(
    "tab_auc.tex",
    "Primary metric where it differs from accuracy (ROC-AUC, \\%).",
    "tab:auc",
    [
        [lab]
        + [
            pct(ms(data[d][m], lambda r: r["metric"]))
            for d in ("minesweeper", "tolokers")
        ]
        for m, lab in MODELS
        if m != "smoothgcn"
    ],
    ["Mines.", "Tolok."],
)
rows = []
for b in ("1", "2", "3", "5"):
    for m, lab in (
        ("certgam", "CertGAM"),
        ("certrob", "CertGAM-R"),
        ("smoothgcn", "Smooth-GCN"),
    ):
        rows.append(
            [f"{lab}, $b={b}$"]
            + [pct(ms(data[d][m], lambda r: r["cert"][b])) for d in ALL]
        )
for m, lab in (
    ("logreg", "LogReg (all $b$)"),
    ("mlp", "MLP (all $b$)"),
    ("egogam", "Ego-GAM (all $b$)"),
):
    rows.append([lab] + [pct(ms(data[d][m])) for d in ALL])
table(
    "tab_cert.tex",
    "Certified accuracy (\\%). Graph-free models are unaffected by edge edits, so their certified accuracy equals clean accuracy at every budget. Smooth-GCN certificates hold with confidence $1-\\alpha$ per node ($\\alpha=0.01$).",
    "tab:cert",
    rows,
    cols,
)
V = json.load(open(os.path.join(R, "verdict.json")))
T = V["table"]
f2 = lambda x: "--" if x is None else f"{x:.2f}"
table(
    "tab_pi.tex",
    "Break-even attack rate $\\pi^\\star$ at $b=1$ against the best graph-free model chosen on validation data. Datasets where the graph adds fewer than 3 points of clean accuracy are marked.",
    "tab:pi",
    [
        ["Graph helps"]
        + [
            (
                ("yes" if T[d]["graph_useful"] else "no")
                if T[d]["graph_useful"] is not None
                else "--"
            )
            for d in ALL
        ],
        ["Best graph-free (\\%)"]
        + ["--" if T[d]["gf"] is None else f"{100*T[d]['gf']:.1f}" for d in ALL],
        ["$\\pi^\\star$ CertGAM"] + [f2(T[d].get("pi")) for d in ALL],
        ["$\\pi^\\star$ CertGAM-R"] + [f2(T[d].get("pi_rob")) for d in ALL],
    ],
    cols,
)
table(
    "tab_attr.tex",
    "Spearman correlation between certified radius and degree, and the neighbour share of score variation, $\\mathrm{std}(S^1)/(\\mathrm{std}(S^0)+\\mathrm{std}(S^1))$.",
    "tab:attr",
    [
        ["Spearman(radius, degree)"]
        + [
            (
                f2(ms(data[d]["certgam"], lambda r: r["spearman_deg"])[0])
                if data[d]["certgam"]
                else "--"
            )
            for d in ALL
        ],
        ["Neighbour share, CertGAM"]
        + [
            (
                f2(ms(data[d]["certgam"], lambda r: r["graph_share"])[0])
                if data[d]["certgam"] and "graph_share" in data[d]["certgam"]["rows"][0]
                else "--"
            )
            for d in ALL
        ],
        ["Neighbour share, CertGAM-R"]
        + [
            (
                f2(ms(data[d]["certrob"], lambda r: r["graph_share"])[0])
                if data[d]["certrob"]
                else "--"
            )
            for d in ALL
        ],
    ],
    cols,
)
# Figures
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    }
)
B = ["1", "2", "3", "5", "10"]
fig, axes = plt.subplots(3, 4, figsize=(7.1, 4.6), sharex=True)
for ax, d in zip(axes.ravel(), ALL):
    for m, lab, c, ls in (
        ("certgam", "CertGAM", "#1f77b4", "-"),
        ("certrob", "CertGAM-R", "#d62728", "-"),
        ("smoothgcn", "Smooth-GCN", "#2ca02c", "--"),
    ):
        if data[d][m]:
            ax.plot(
                range(5),
                [ms(data[d][m], lambda r: r["cert"][b])[0] for b in B],
                ls,
                c=c,
                marker="o",
                ms=2.5,
                lw=1,
                label=lab,
            )
    if T[d]["gf"] is not None:
        ax.axhline(T[d]["gf"], c="k", lw=0.9, ls=":", label="Best graph-free")
    ax.set_title(NAME.get(d, d), fontsize=8)
    ax.set_xticks(range(5))
    ax.set_xticklabels(B)
    ax.set_ylim(0, 1)
axes.ravel()[-1].axis("off")
h, l = axes.ravel()[0].get_legend_handles_labels()
axes.ravel()[-1].legend(h, l, loc="center", frameon=False)
fig.supxlabel("Edit budget $b$", fontsize=8)
fig.supylabel("Certified accuracy", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(F, "cert_curves.pdf"))
plt.close(fig)
fig, ax = plt.subplots(figsize=(3.4, 3))
mk = dict(zip(ALL, "ov^<>sPXD*h"))
for d in ALL:
    t = T[d]
    if t["gf"] is None or t["clean"] is None or t["rob_clean"] is None:
        continue
    ax.plot(
        [t["clean"] - t["gf"], t["rob_clean"] - t["gf"]],
        [t["cert"] - t["gf"], t["rob_cert"] - t["gf"]],
        c=".75",
        lw=0.7,
        zorder=1,
    )
    ax.scatter(
        t["clean"] - t["gf"],
        t["cert"] - t["gf"],
        marker=mk[d],
        c="#1f77b4",
        s=18,
        zorder=2,
    )
    ax.scatter(
        t["rob_clean"] - t["gf"],
        t["rob_cert"] - t["gf"],
        marker=mk[d],
        c="#d62728",
        s=18,
        zorder=2,
        label=NAME.get(d, d),
    )
ax.axhline(0, c="k", lw=0.6)
ax.axvline(0, c="k", lw=0.6)
ax.set_xlabel("Clean accuracy minus best graph-free")
ax.set_ylabel("Certified accuracy ($b=1$) minus best graph-free")
ax.legend(frameon=False, fontsize=6, ncol=2)
fig.savefig(os.path.join(F, "tradeoff.pdf"))
plt.close(fig)
fig, ax = plt.subplots(figsize=(3.4, 2.1))
el2 = json.load(open(os.path.join(R, "audit_lemma.json")))[1:]
ax.plot(
    [r["R"] for r in el2],
    [r["median_ratio"] for r in el2],
    "o-",
    c="#d62728",
    ms=3,
    lw=1,
    label="Median over edge flips",
)
ax.plot(
    [r["R"] for r in el2],
    [r["min_ratio"] for r in el2],
    "s--",
    c="#1f77b4",
    ms=3,
    lw=1,
    label="Minimum over edge flips",
)
ax.axhline(1, c="k", lw=0.7, ls=":")
ax.set_xlabel("Filter order $R$ (single monomial)")
ax.set_ylabel("True change / printed constant")
ax.legend(frameon=False, fontsize=7)
ax.set_yscale("log")
fig.savefig(os.path.join(F, "lemma.pdf"))
plt.close(fig)
SENS = json.load(open(os.path.join(R, "sensitivity.json")))
Vv = V["verdict"]
gnc = {}
for d in ALL:
    p = os.path.join(R, "s03", f"gnam_cert_{d}.json")
    if os.path.exists(p):
        gnc[d] = float(np.mean([r["top1_certified_b1"] for r in json.load(open(p))]))
table(
    "tab_joint.tex",
    "Certified explanations at $b=1$ on at most 200 test nodes per split. ``Top-1'' fixes the class at the clean prediction; ``Joint'' also requires the prediction to be certified. A graph-free model is trivially stable under both, since its attributions ignore the graph.",
    "tab:joint",
    [
        ["Top-1, CertGAM"] + [f"{100*SENS[d]['top1']:.1f}" for d in ALL],
        ["Joint, CertGAM"] + [f"{100*SENS[d]['joint']:.1f}" for d in ALL],
        ["Top-1, CertGAM-R"] + [f"{100*SENS[d]['rob_top1']:.1f}" for d in ALL],
        ["Joint, CertGAM-R"] + [f"{100*SENS[d]['rob_joint']:.1f}" for d in ALL],
    ],
    cols,
)
table(
    "tab_sens.tex",
    "Sensitivity of $\\pi^\\star$ to computing clean and certified accuracy on the same test nodes.",
    "tab:sens",
    [
        ["$\\pi^\\star$ CertGAM"] + [f2(SENS[d]["pi"]) for d in ALL],
        ["$\\pi^\\star$ CertGAM-R"] + [f2(SENS[d]["pi_rob"]) for d in ALL],
    ],
    cols,
)
useful = Vv["H8"]["useful"]
num = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
}
nm = lambda d: {"roman-empire": "roman-empire", "amazon-ratings": "amazon-ratings"}.get(
    d, d
)
syn = json.load(open(os.path.join(R, "s03", "synthetic.json")))
sm = lambda k: float(np.mean([x[k] for x in syn]))
gfm = lambda d: T[d]["gf"]
# Holm adjustment over the pre-specified hypotheses that carry a p-value.
V7p = os.path.join(R, "verdict7.json")  # E8 (PLAN_v7): H14 joins the Holm family
V7 = json.load(open(V7p)) if os.path.exists(V7p) else None
pv = {h: Vv[h]["p"] for h in Vv if Vv[h].get("p") is not None}
if V7:
    pv["H14"] = V7["verdict"]["H14"]["p"]
pk = sorted((p, h) for h, p in pv.items())
holm, run_ = {}, 0.0
for j, (p, h) in enumerate(pk):
    run_ = max(run_, min(1.0, (len(pk) - j) * p))
    holm[h] = run_
macros.update(
    holmTwo=f"{holm['H2']:.3f}", holmThree=f"{holm['H3']:.3f}", holmM=str(len(pk))
)
if V7:
    macros.update(holmFourteen=f"{holm['H14']:.3f}")
macros.update(
    hOneTried=f"{Vv['H1']['tried']:,}",
    hOneBroken=str(Vv["H1"]["broken"]),
    hTwoWins=str(Vv["H2"]["wins"]),
    hTwoP=f"{Vv['H2']['p']:.4f}",
    hThreeGap=f"{100*Vv['H3']['median_gap_gcn']:.1f}",
    hThreeWins=str(Vv["H3"]["het_wins"]),
    hThreeP=f"{Vv['H3']['p']:.3f}",
    hFourCount=str(Vv["H4"]["count"]),
    hFiveCount=str(Vv["H5"]["count"]),
    hSixSpeed=f"{Vv['H6']['median_speedup']:.0f}",
    hSevenC=f"{Vv['H7']['certgam']:.2f}",
    hSevenG=f"{Vv['H7']['gnamrff']:.2f}",
    hEightUseful=str(len(useful)),
    hEightCount=str(Vv["H8"]["count"]),
    hEightList=", ".join(nm(d) for d in useful),
    hNineCount=str(Vv["H9"]["count"]),
    hTenCount=str(Vv["H10"]["count"]),
    synAccC=f"{100*sm('certgam_acc'):.1f}",
    synAccG=f"{100*sm('gnamrff_acc'):.1f}",
)
for d in useful:
    k = d.replace("-", "").replace("empire", "Emp")
    macros[f"piP{k}"] = f"{T[d]['pi']:.2f}"
    macros[f"piR{k}"] = f"{T[d]['pi_rob']:.2f}"
macros.update(
    coraClean=f"{100*T['Cora']['clean']:.1f}",
    coraCert=f"{100*T['Cora']['cert']:.1f}",
    coraRClean=f"{100*T['Cora']['rob_clean']:.1f}",
    coraRCert=f"{100*T['Cora']['rob_cert']:.1f}",
    coraGf=f"{100*T['Cora']['gf']:.1f}",
    coraJoint=f"{100*SENS['Cora']['joint']:.1f}",
    coraTop=f"{100*SENS['Cora']['top1']:.1f}",
    coraRJoint=f"{100*SENS['Cora']['rob_joint']:.1f}",
    romanClean=f"{100*T['roman-empire']['clean']:.1f}",
    romanGcn=f"{100*T['roman-empire']['gcn']:.1f}",
    romanGf=f"{100*T['roman-empire']['gf']:.1f}",
    gnamCertN=str(len(gnc)),
    gnamCertMax=f"{100*max(gnc.values()):.1f}" if gnc else "--",
)
AN = json.load(open(os.path.join(R, "analysis.json")))
S4 = os.path.join(R, "s04")


def L4(d, m):
    p = os.path.join(S4, f"{d}__{m}.json")
    if not os.path.exists(p):
        missing.append(f"{d} {m} s04: absent")
        return None
    x = json.load(open(p))
    return x if x.get("status") == "ok" and x.get("rows") else None


# data table
stats, major = {}, {}
for d in ALL:
    g, _ = gam.load(d)
    te_ = exp.splits(d)[1][2]  # Planetoid: the 10 random splits
    major[d] = float(  # majority-class test accuracy, confirmatory splits 3-9
        np.mean(
            [
                np.bincount(g.y[te_[:, s]].numpy()).max() / int(te_[:, s].sum())
                for s in range(3, te_.size(1))
            ]
        )
    )
    deg = np.bincount(g.edge_index[0].numpy(), minlength=g.num_nodes)
    stats[d] = (
        g.num_nodes,
        g.edge_index.size(1) // 2,
        g.x.size(1),
        int(g.y.max()) + 1,
        float(np.median(deg)),
        100 * float((deg <= 1).mean()),
    )
table(
    "tab_data.tex",
    "Dataset statistics.",
    "tab:data",
    [
        ["Nodes"] + [f"{stats[d][0]:,}" for d in ALL],
        ["Edges"] + [f"{stats[d][1]:,}" for d in ALL],
        ["Features"] + [stats[d][2] for d in ALL],
        ["Classes"] + [stats[d][3] for d in ALL],
        ["Median degree"] + [f"{stats[d][4]:.0f}" for d in ALL],
        ["Degree $\\le1$ (\\%)"] + [f"{stats[d][5]:.0f}" for d in ALL],
    ],
    cols,
)
ci = lambda v: f"[{v[0]:.2f}, {v[1]:.2f}]"
table(
    "tab_pi2.tex",
    "Uncertainty of $\\pi^\\star$ for CertGAM-R at $b=1$: split-level bootstrap interval for the selected reference, and the point value under each graph-free reference. Clean gain is CertGAM minus Ego-GAM, in points.",
    "tab:pi2",
    [["95\\% interval"] + [ci(AN[d]["ci_rob"]) for d in ALL]]
    + [
        [f"$\\pi^\\star$, {lab}"] + [f2(AN[d]["by_ref"][m]["pi_rob"]) for d in ALL]
        for m, lab in (("logreg", "LogReg"), ("mlp", "MLP"), ("egogam", "Ego-GAM"))
    ]
    + [["Clean gain over Ego-GAM"] + [f"{100*AN[d]['ego_gain']:.1f}" for d in ALL]],
    cols,
)
DG = json.load(open(os.path.join(S4, "degree.json")))
bins = ["0-1", "2-3", "4-7", "8-1000000000"]
bl = {"0-1": "$\\le1$", "2-3": "2--3", "4-7": "4--7", "8-1000000000": "$\\ge8$"}
cell = lambda d, m, b, k: (
    f"{100*DG[d][m][b][k]:.0f}" if DG[d][m][b]["n"] >= 30 else "--"
)
table(
    "tab_degree.tex",
    "Certified accuracy at $b=1$ by degree of the test node, in percent, over splits 3--5. Strata with fewer than 30 nodes are omitted. For degree $\\le1$ a single edit can remove every neighbour.",
    "tab:degree",
    [
        [f"CertGAM, degree {bl[b]}"] + [cell(d, "certgam", b, "c1") for d in ALL]
        for b in bins
    ]
    + [
        [f"CertGAM-R, degree {bl[b]}"] + [cell(d, "certrob", b, "c1") for d in ALL]
        for b in bins
    ],
    cols,
)
ml, mr, sg = (
    {d: L4(d, "certmlp") for d in ALL},
    {d: L4(d, "certmlprob") for d in ALL},
    {d: L4(d, "smoothsage") for d in ALL},
)
table(
    "tab_mlp.tex",
    "Supplementary, not pre-specified: one-layer mean-aggregation with MLP shape functions (CertMLP), in percent. These runs used splits already seen and are descriptive.",
    "tab:mlp",
    [
        ["CertMLP clean"] + [pct(ms(ml[d])) for d in ALL],
        ["CertMLP $b=1$"]
        + [pct(ms(ml[d], lambda r: r["cert"]["1"])) if ml[d] else "--" for d in ALL],
        ["CertMLP-R clean"] + [pct(ms(mr[d])) for d in ALL],
        ["CertMLP-R $b=1$"]
        + [pct(ms(mr[d], lambda r: r["cert"]["1"])) if mr[d] else "--" for d in ALL],
    ],
    cols,
)
table(
    "tab_smooth2.tex",
    "Supplementary, not pre-specified: Smooth-SAGE with eight configurations drawn from a 16-point noise grid and 20,000 samples, in percent. With 20,000 samples and $\\alpha=0.01$ the sample size does not cap the budgets shown.",
    "tab:smooth2",
    [["Clean"] + [pct(ms(sg[d])) for d in ALL]]
    + [
        [f"$b={b}$"]
        + [
            (
                (
                    pct(ms(sg[d], lambda r: r["cert"][b]))
                    + (
                        "$^\\dagger$"
                        if sg[d]["rows"][0]["sample_cap_blocks"][b]
                        else ""
                    )
                )
                if sg[d]
                else "--"
            )
            for d in ALL
        ]
        for b in ("1", "2", "3")
    ],
    cols,
)
have = [d for d in ALL if sg[d]]
bgood = [d for d in have if T[d]["rob_cert"] > ms(sg[d], lambda r: r["cert"]["1"])[0]]
from scipy.stats import wilcoxon

pS = (
    float(
        wilcoxon(
            [T[d]["rob_cert"] for d in have],
            [ms(sg[d], lambda r: r["cert"]["1"])[0] for d in have],
        ).pvalue
    )
    if len(have) >= 6
    else float("nan")
)
macros.update(
    smoothNRan=str(len(have)), smoothWins=str(len(bgood)), smoothP=f"{pS:.3f}"
)
mlw = [d for d in ALL if ml[d] and ms(ml[d])[0] > T[d]["clean"] + 0.01]
macros.update(
    mlpCleanBetter=str(len(mlw)),
    mlpNRan=str(sum(1 for d in ALL if ml[d])),
    amzMlp=f"{100*ms(ml['amazon-ratings'])[0]:.1f}" if ml["amazon-ratings"] else "--",
    amzGam=f"{100*T['amazon-ratings']['clean']:.1f}",
    amzGf=f"{100*T['amazon-ratings']['gf']:.1f}",
)
below = sum(AN[d]["rob_cert"] < AN[d]["best_test_ref"] for d in ALL)
gainE = sum(AN[d]["ego_gain"] >= 0.03 for d in ALL)
# N1 (review 2): where the best graph-free reference is no better than predicting the majority class, the comparison is trivial.
triv = [d for d in ALL if AN[d]["best_test_ref"] <= major[d] + 0.005]
nt = [d for d in ALL if d not in triv]
selAcc = lambda d: AN[d]["by_ref"][AN[d]["selected"]]["acc"]
macros.update(
    trivNames=", ".join(NAME.get(d, d) for d in triv) or "none",
    trivMaj=", ".join(f"{100*major[d]:.1f}" for d in triv) or "--",
    ntN=str(len(nt)),
    ntBelowBest=str(sum(AN[d]["rob_cert"] < AN[d]["best_test_ref"] for d in nt)),
    ntBelowSel=str(sum(AN[d]["rob_cert"] < selAcc(d) for d in nt)),
)
macros.update(
    robBelow=str(below),
    egoGain=str(gainE),
    coraCiLo=f"{AN['Cora']['ci_rob'][0]:.2f}",
    coraCiHi=f"{AN['Cora']['ci_rob'][1]:.2f}",
    texasPiLog=f"{AN['Texas']['by_ref']['logreg']['pi_rob']:.2f}",
    texasPiEgo=f"{AN['Texas']['by_ref']['egogam']['pi_rob']:.2f}",
    texasShare=f"{AN['Texas']['share']:.2f}",
    dOneCora=cell("Cora", "certgam", "0-1", "c1"),
    dFourCora=cell("Cora", "certgam", "4-7", "c1"),
)
ALT = json.load(open(os.path.join(R, "s02", "alt1.json")))
rng3 = lambda ds, s: (
    min(x["obj"] for x in ALT[ds][s]),
    max(x["obj"] for x in ALT[ds][s]),
)
macros.update(
    cutTexas=f"{rng3('Texas','cutting')[0]:.4f}--{rng3('Texas','cutting')[1]:.4f}",
    adamTexas=f"{rng3('Texas','adam')[0]:.4f}--{rng3('Texas','adam')[1]:.4f}",
    cutCornell=f"{rng3('Cornell','cutting')[0]:.4f}--{rng3('Cornell','cutting')[1]:.4f}",
    cutActor=f"{rng3('Actor','cutting')[0]:.3f}--{rng3('Actor','cutting')[1]:.3f}",
    adamActor=f"{rng3('Actor','adam')[0]:.3f}--{rng3('Actor','adam')[1]:.3f}",
)
AB = json.load(open(os.path.join(R, "s03", "ablation.json")))
rob = [v for v in AB.values() if v["factor"] == "rob_lam"]
lams = sorted({v["cfg"]["lam"] for v in rob})
dsab = ["Cora", "Texas", "Wisconsin", "Cornell", "Actor", "minesweeper"]
abl = {(v["ds"], v["cfg"]["lam"]): v for v in rob}
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.1))
for d in dsab:
    v = [abl[(d, l)] for l in lams if (d, l) in abl]
    axs[0].semilogx(
        [x["cfg"]["lam"] for x in v],
        [x["share"] for x in v],
        "o-",
        ms=2.5,
        lw=1,
        label=NAME.get(d, d),
    )
    axs[1].semilogx(
        [x["cfg"]["lam"] for x in v],
        [x["clean"] - x["c1"] for x in v],
        "o-",
        ms=2.5,
        lw=1,
    )
axs[0].set_xlabel(r"$\lambda$")
axs[0].set_ylabel("Neighbour share")
axs[1].set_xlabel(r"$\lambda$")
axs[1].set_ylabel(r"clean $-$ cert$_1$")
axs[0].legend(frameon=False, fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(F, "lambda.pdf"))
plt.close(fig)
C = json.load(open(os.path.join(S4, "case.json")))
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.2))
for c_, col in zip(C["curves"], ("#1f77b4", "#d62728", "#2ca02c")):
    axs[0].plot(c_["z"], c_["ego"], c=col, lw=1, label=f"f{c_['feature']} ego")
    axs[0].plot(
        c_["z"], c_["nbr"], c=col, lw=1, ls="--", label=f"f{c_['feature']} neighbour"
    )
axs[0].set_xlabel("Standardized feature value")
axs[0].set_ylabel("Contribution to class score")
axs[0].legend(frameon=False, fontsize=5.5, ncol=3)
rr_ = C["node"]["rows"]
yy = list(range(len(rr_)))[::-1]
axs[1].errorbar(
    [r["value"] for r in rr_],
    yy,
    xerr=[[r["value"] - r["lo"] for r in rr_], [r["hi"] - r["value"] for r in rr_]],
    fmt="o",
    ms=3,
    lw=1,
    c="k",
)
axs[1].set_yticks(yy)
axs[1].set_yticklabels([f"f{r['feature']}" for r in rr_], fontsize=6)
axs[1].set_xlabel("Contribution with exact interval at $b=1$")
fig.tight_layout()
fig.savefig(os.path.join(F, "case.pdf"))
plt.close(fig)
macros.update(caseDeg=str(C["node"]["deg"]))
# E6 (PLAN_v5): two-layer SAGE, exact b=1 certificates; verdicts from results/verdict5.json (code/judge5.py).
V5p = os.path.join(R, "verdict5.json")
if os.path.exists(V5p):
    V5 = json.load(open(V5p))
    S5 = {}
    for d in ALL:
        p = os.path.join(R, "s05", f"{d}__sage2.json")
        S5[d] = json.load(open(p)) if os.path.exists(p) else None
        if not S5[d] or S5[d].get("status") != "ok":
            missing.append(f"{d} sage2 s05: {S5[d]['status'] if S5[d] else 'absent'}")
            S5[d] = None
    A7 = {}
    for (
        d
    ) in (
        ALL
    ):  # E7 (PLAN_v6, exploratory): exact adversarial training; only the six smaller non-trivial graphs
        p7 = os.path.join(R, "s06", f"{d}__sageadv.json")
        A7[d] = json.load(open(p7)) if os.path.exists(p7) else None
        A7[d] = A7[d] if A7[d] and A7[d].get("status") == "ok" else None
    adv = {d: ms(A7[d], lambda r: r["cert1"])[0] for d in ALL if A7[d]}
    advAbove = [d for d in adv if adv[d] > AN[d]["by_ref"][AN[d]["selected"]]["acc"]]
    advWeak = [
        d
        for d in adv
        if S5[d] and adv[d] < np.mean([r["cert1"] for r in S5[d]["rows"]["sageaug"]])
    ]  # adversarial training under-budgeted here
    macros.update(
        advN=str(len(adv)),
        advAboveN=str(len(advAbove)),
        advAbove=", ".join(
            f"{NAME.get(d, d)} ({100*adv[d]:.1f} vs {100*AN[d]['by_ref'][AN[d]['selected']]['acc']:.1f})"
            for d in advAbove
        )
        or "none",
        advVsRob=str(sum(adv[d] > AN[d]["rob_cert"] for d in adv)),
        advWeak=", ".join(NAME.get(d, d) for d in advWeak) or "none",
        advWeakN=str(len(advWeak)),
    )
    A9 = {}
    for d in ALL:  # E9 (PLAN_v7, exploratory): E7 with four times the budget
        p9 = os.path.join(R, "s08", f"{d}__sageadv4.json")
        A9[d] = json.load(open(p9)) if os.path.exists(p9) else None
        A9[d] = A9[d] if A9[d] and A9[d].get("status") == "ok" else None
    adv4 = {d: ms(A9[d], lambda r: r["cert1"])[0] for d in ALL if A9[d]}
    sAug = lambda d: np.mean([r["cert1"] for r in S5[d]["rows"]["sageaug"]])
    macros.update(
        advFourN=num.get(len(adv4), str(len(adv4))),
        advFourBelowN=num.get(sum(adv4[d] <= selAcc(d) for d in adv4), "no"),
        advFour="; ".join(
            f"{NAME.get(d, d)} {100*adv[d]:.1f} to {100*adv4[d]:.1f} (SAGE-aug {100*sAug(d):.1f}, graph-free {100*selAcc(d):.1f})"
            for d in adv4
        )
        or "none",
    )
    rowsv = lambda d, v: dict(rows=S5[d]["rows"][v]) if S5[d] else None
    lab = {"sage": "SAGE", "sageaug": "SAGE-aug"}
    table(
        "tab_sage2.tex",
        "Two-layer mean-aggregation GraphSAGE with exact certificates at $b=1$, in percent, splits 3--9 (hypotheses fixed before the certificates were computed). SAGE-aug is trained on randomly perturbed graphs; the certificate is for the deterministic model. The selected variant is chosen on validation data of split 0. Graph-free is the validation-selected graph-free reference, on all test nodes; the SAGE rows use the certified subsample of at most 1000 test nodes per split. On WebKB the variant choice rests on very few validation nodes. Standard deviations are over splits (population).",
        "tab:sage2",
        [[f"{lab[v]} clean"] + [pct(ms(rowsv(d, v))) for d in ALL] for v in lab]
        + [
            [f"{lab[v]} $b=1$"]
            + [pct(ms(rowsv(d, v), lambda r: r["cert1"])) for d in ALL]
            for v in lab
        ]
        + [
            ["SAGE-adv clean (expl.)"] + [pct(ms(A7[d])) for d in ALL],
            ["SAGE-adv $b=1$ (expl.)"]
            + [pct(ms(A7[d], lambda r: r["cert1"])) for d in ALL],
            ["SAGE-adv$\\times$4 $b=1$ (expl.)"]
            + [pct(ms(A9[d], lambda r: r["cert1"])) for d in ALL],
        ]
        + [
            ["Selected variant"]
            + [lab[S5[d]["chosen"]] if S5[d] else "--" for d in ALL]
        ]
        + [
            ["Graph-free"]
            + [f"{100*AN[d]['by_ref'][AN[d]['selected']]['acc']:.1f}" for d in ALL]
        ]
        + [["CertGAM-R $b=1$"] + [f"{100*AN[d]['rob_cert']:.1f}" for d in ALL]]
        + [
            ["Sel., degree $\\le1$"]
            + [
                (
                    f"{100*V5['table'][d]['c1_low']:.0f}"
                    if V5["table"][d].get("n_low", 0) >= 30
                    else "--"
                )
                for d in ALL
            ]
        ]
        + [
            ["Sel., degree 4--7"]
            + [
                (
                    f"{100*V5['table'][d]['c1_mid']:.0f}"
                    if V5["table"][d].get("n_mid", 0) >= 30
                    else "--"
                )
                for d in ALL
            ]
        ],
        cols,
    )
    vv = V5["verdict"]
    beatGF = [
        NAME.get(d, d)
        for d in ALL
        if V5["table"][d].get("status") == "ok" and not V5["table"][d]["below_ref"]
    ]
    beatRob = sum(
        1
        for d in ALL
        if V5["table"][d].get("status") == "ok"
        and V5["table"][d]["cert1"] > V5["table"][d]["certrob"]
    )
    macros.update(
        hElevenN=str(vv["H11"]["n"]),
        hElevenC=str(vv["H11"]["count"]),
        hElevenPass="passed" if vv["H11"]["passed"] else "did not pass",
        hTwelveN=str(vv["H12"]["n"]),
        hTwelveC=str(vv["H12"]["count"]),
        hTwelvePass="passed" if vv["H12"]["passed"] else "did not pass",
        sageBeatGF=", ".join(beatGF) or "none",
        sageBeatRob=str(beatRob),
        sageNok=str(sum(1 for d in ALL if S5[d])),
    )
# M2 (review 3): Smooth-SAGE against the validation-selected graph-free reference on the non-trivial datasets.
sgc1 = {d: ms(sg[d], lambda r: r["cert"]["1"])[0] for d in nt if sg[d]}
sgAbove = [d for d in sgc1 if sgc1[d] > selAcc(d)]
sgNear = [d for d in sgc1 if d not in sgAbove and selAcc(d) - sgc1[d] <= 0.002]
macros.update(
    sgAbove=", ".join(
        f"{NAME.get(d, d)} ({100*sgc1[d]:.1f} vs {100*selAcc(d):.1f})" for d in sgAbove
    )
    or "none",
    sgAboveN=str(len(sgAbove)),
    sgNear=", ".join(NAME.get(d, d) for d in sgNear) or "none",
)
if os.path.exists(
    V5p
):  # wall time of the two-layer certificate, per split including training
    secs = [
        r["seconds"]
        for d in ALL
        if S5[d]
        for v in S5[d]["rows"]
        for r in S5[d]["rows"][v]
    ]
    macros.update(sageSecMed=f"{np.median(secs):.0f}", sageSecMax=f"{max(secs):.0f}")
# E8 (PLAN_v7): ShrinkGAM-R, pseudo-count kappa in the aggregation denominator; verdicts from code/judge7.py.
if V7:
    T7 = V7["table"]
    S7 = {}
    for d in ALL:
        p = os.path.join(R, "s07", f"{d}__shrink.json")
        S7[d] = json.load(open(p)) if os.path.exists(p) else None
        if not S7[d] or S7[d].get("status") != "ok":
            missing.append(f"{d} shrink s07: {S7[d]['status'] if S7[d] else 'absent'}")
            S7[d] = None
    rw = lambda d, t: dict(rows=S7[d]["rows"][t]) if S7[d] else None
    c1 = lambda r: r["cert"]["1"]
    gl = lambda d, k: (
        "--"
        if T7[d].get(k) is None or T7[d]["n_low" if k == "gain_low" else "n_mid"] < 30
        else f"{100*T7[d][k]:+.1f}"
    )
    table(
        "tab_shrink.tex",
        "ShrinkGAM-R (pseudo-count $\\kappa$ selected on validation data of split 0, CertGAM-R hyperparameters otherwise) against CertGAM-R retrained by the ShrinkGAM code ($\\kappa=0$), in percent, splits 3--9, exact certificates on at most 2000 test nodes per split. Hypotheses H14--H16 were fixed before these runs. Gains are $b=1$ certified accuracy of ShrinkGAM-R minus $\\kappa=0$ within a degree stratum, shown when the stratum has at least 30 nodes. Graph-free is the validation-selected graph-free reference.",
        "tab:shrink",
        [
            ["Selected $\\kappa$"]
            + [f"{S7[d]['kappa']:g}" if S7[d] else "--" for d in ALL]
        ]
        + [["ShrinkGAM-R clean"] + [pct(ms(rw(d, "chosen"))) for d in ALL]]
        + [["$\\kappa=0$ clean"] + [pct(ms(rw(d, "kappa0"))) for d in ALL]]
        + [["ShrinkGAM-R $b=1$"] + [pct(ms(rw(d, "chosen"), c1)) for d in ALL]]
        + [["$\\kappa=0$ $b=1$"] + [pct(ms(rw(d, "kappa0"), c1)) for d in ALL]]
        + [["Graph-free"] + [f"{100*selAcc(d):.1f}" for d in ALL]]
        + [["Gain, degree $\\le1$"] + [gl(d, "gain_low") for d in ALL]]
        + [["Gain, degree 4--7"] + [gl(d, "gain_mid") for d in ALL]],
        cols,
    )
    v7 = V7["verdict"]
    KAPPA7 = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]  # grid of code/exp8.py
    pos = [d for d in ALL if S7[d] and S7[d]["kappa"] > 0]
    up = {d: T7[d]["cert1"] - T7[d]["cert1_k0"] for d in pos}
    macros.update(
        shrN=str(v7["H14"]["n"]),
        hFourteenC=str(v7["H14"]["count"]),
        hFourteenP=f"{v7['H14']['p']:.3f}",
        hFourteenPass="passed" if v7["H14"]["passed"] else "did not pass",
        hFifteenC=str(v7["H15"]["count"]),
        hFifteenN=str(v7["H15"]["n"]),
        hFifteenPass="passed" if v7["H15"]["passed"] else "did not pass",
        hFifteenList=", ".join(
            NAME.get(d, d) for d in nt if T7[d]["cert1"] > T7[d]["ref_sel"]
        )
        or "none",
        hSixteenC=str(v7["H16"]["count"]),
        hSixteenN=str(v7["H16"]["n"]),
        hSixteenPass="passed" if v7["H16"]["passed"] else "did not pass",
        shrKZero=", ".join(NAME.get(d, d) for d in ALL if S7[d] and S7[d]["kappa"] == 0)
        or "none",
        shrPosN=str(len(pos)),
        shrUp=", ".join(
            f"{NAME.get(d, d)} ({100*up[d]:+.1f})" for d in pos if up[d] >= 0.01
        )
        or "none",
        shrDown=", ".join(
            f"{NAME.get(d, d)} ({100*up[d]:+.1f})" for d in pos if up[d] < -0.0005
        )
        or "none",
        shrGap=", ".join(
            f"{NAME.get(d, d)} ({100*(T7[d]['ref_sel']-T7[d]['cert1']):.1f})"
            for d in nt
            if d in pos and T7[d]["cert1"] <= T7[d]["ref_sel"]
        )
        or "none",
        shrKMaxN=num[sum(1 for d in ALL if S7[d] and S7[d]["kappa"] == max(KAPPA7))],
        shrHSixteenZero=num[
            sum(
                1
                for d in ALL
                if S7[d]
                and S7[d]["kappa"] == 0
                and T7[d]["n_low"] >= 30
                and T7[d]["n_mid"] >= 30
            )
        ],
        shrAccMax=f"{100*max(T7[d]['acc_k0']-T7[d]['acc'] for d in pos):.1f}",
        shrReimpl=f"{100*max(abs(T7[d]['cert1_k0']-T7[d]['cert1_s03']) for d in ALL if S7[d]):.1f}",  # review 5
        shrFuzz=str(sum(T7[d]["fuzz_broken"] for d in ALL if S7[d])),
        shrFuzzTried=f"{sum(r['fuzz_tried'] for d in ALL if S7[d] for t in (('chosen',) if S7[d]['kappa'] == 0 else ('chosen', 'kappa0')) for r in S7[d]['rows'][t]):,}",  # kappa=0 rows copy the chosen rows (review 5)
    )
    VS = json.load(open(os.path.join(R, "verify_shrink.json")))
    macros.update(
        shrVerifyN=f"{VS['checked']:,}", shrVerifyRatio=f"{VS['max_ratio']:.6f}"
    )
# E10/E10e (PLAN_v8, v8b): external certified baseline pi-PPNP; verdicts from code/judge8.py.
V8p = os.path.join(R, "verdict8.json")
V8 = json.load(open(V8p)) if os.path.exists(V8p) else None
if V8:
    T8 = V8["table"]
    P8 = {}
    for d in ALL:
        for tag in ("pippnp", "pippnp_exact"):
            p = os.path.join(R, "s09", f"{d}__{tag}.json")
            x = json.load(open(p)) if os.path.exists(p) else None
            if not x or x.get("status") != "ok":
                missing.append(f"{d} {tag} s09: {x['status'] if x else 'absent'}")
                x = None
            P8[d, tag] = x
    cfg8 = lambda x: f"{x['cfg']['alpha']:g}/{x['cfg']['loss'].upper()}" if x else "--"
    strat = lambda d, k, n: f"{100*T8[d][k]:.0f}" if T8[d].get(n, 0) >= 30 else "--"
    table(
        "tab_ppnp.tex",
        "$\\pi$-PPNP~\\cite{bojchevski2019} on our splits, in percent, splits 3--9, at most 1000 test nodes per split. Configuration ($\\alpha$/loss) was chosen on validation data of split 0, separately for each certificate. Exact: minimum margin over all single undirected edge toggles (graphs with at most 8000 nodes). Local: the policy-iteration certificate of~\\cite{bojchevski2019} with one edit per node, sound but much weaker than one global edit. Attack bound: correct nodes of the locally selected model that survived our single-edit attack, on the first 200 test nodes per split, an upper bound on its exact certified accuracy at $b=1$; on Cora the local and exact rows are different models. Degree strata of the exact certificate are shown with at least 30 nodes. Graph-free is the validation-selected graph-free reference. Hypotheses fixed before the runs.",
        "tab:ppnp",
        [["Exact: config"] + [cfg8(P8[d, "pippnp_exact"]) for d in ALL]]
        + [["Exact: clean"] + [pct(ms(P8[d, "pippnp_exact"])) for d in ALL]]
        + [
            ["Exact: $b=1$"]
            + [pct(ms(P8[d, "pippnp_exact"], lambda r: r["cert1"])) for d in ALL]
        ]
        + [["Exact, degree $\\le1$"] + [strat(d, "e_low", "e_n_low") for d in ALL]]
        + [["Exact, degree 4--7"] + [strat(d, "e_mid", "e_n_mid") for d in ALL]]
        + [["Local: config"] + [cfg8(P8[d, "pippnp"]) for d in ALL]]
        + [["Local: clean"] + [pct(ms(P8[d, "pippnp"])) for d in ALL]]
        + [
            ["Local: $b=1$"]
            + [pct(ms(P8[d, "pippnp"], lambda r: r["cert"]["1"])) for d in ALL]
        ]
        + [
            ["Attack bound, $b=1$"]
            + [pct(ms(P8[d, "pippnp"], lambda r: r["ub1"])) for d in ALL]
        ]
        + [["Graph-free"] + [f"{100*selAcc(d):.1f}" for d in ALL]]
        + [["CertGAM-R $b=1$"] + [f"{100*AN[d]['rob_cert']:.1f}" for d in ALL]],
        cols,
    )
    v8 = V8["verdict"]
    ps = lambda h: "passed" if v8[h]["passed"] else "did not pass"
    ex = [d for d in ALL if P8[d, "pippnp_exact"]]
    h20eNo = [
        d
        for d in ex
        if T8[d]["e_n_low"] >= 30
        and T8[d]["e_n_mid"] >= 30
        and T8[d]["e_low"] >= T8[d]["e_mid"]
    ]
    lc = [d for d in ALL if P8[d, "pippnp"]]
    macros.update(
        **{
            f"h{k}{w}": str(v8[h][f])
            for h, k in (
                ("H18", "Eighteen"),
                ("H19", "Nineteen"),
                ("H20", "Twenty"),
                ("H18e", "EighteenE"),
                ("H20e", "TwentyE"),
            )
            for w, f in (("C", "count"), ("N", "n"))
        },
        **{
            f"h{k}Pass": ps(h)
            for h, k in (
                ("H18", "Eighteen"),
                ("H19", "Nineteen"),
                ("H20", "Twenty"),
                ("H18e", "EighteenE"),
                ("H20e", "TwentyE"),
            )
        },
        ppnpExGap=", ".join(
            f"{NAME.get(d, d)} ({100*(selAcc(d)-T8[d]['e_cert1']):.1f})"
            for d in ex
            if d in nt
        ),
        ppnpExAlphaLow=num.get(
            sum(P8[d, "pippnp_exact"]["cfg"]["alpha"] == 0.1 for d in ex), "?"
        ),
        ppnpExN=num.get(len(ex), str(len(ex))),
        ppnpExVsRob=", ".join(
            f"{NAME.get(d, d)} ({100*T8[d]['e_cert1']:.1f} vs {100*AN[d]['rob_cert']:.1f})"
            for d in ex
            if T8[d]["e_cert1"] > AN[d]["rob_cert"]
        )
        or "none",
        ppnpHTwentyENo=", ".join(NAME.get(d, d) for d in h20eNo) or "none",
        ppnpUnsound=str(sum(T8[d]["unsound"] for d in lc)),
        ppnpAttackTried=f"{sum(T8[d]['attack_tried'] for d in lc):,}",
        ppnpLocAbove=", ".join(
            f"{NAME.get(d, d)} ({100*T8[d]['ub1']:.1f} vs {100*selAcc(d):.1f})"
            for d in lc
            if d in nt and T8[d]["ub1"] >= selAcc(d)
        )
        or "none",
        ppnpCeOnly=", ".join(
            NAME.get(d, d) for d in lc if not P8[d, "pippnp"]["robust"]
        )
        or "none",
    )
    tight = max(
        round((r["ub1"] - r["cert1_attacked"]) * r["attack_nodes"])
        for d in lc
        for r in P8[d, "pippnp"]["rows"]
    )
    macros.update(
        ppnpFailed=", ".join(NAME.get(d, d) for d in nt if not P8[d, "pippnp"])
        or "none",
        ppnpTight=num.get(tight, str(tight)),
        ppnpAlphaLowLoc=num.get(
            sum(P8[d, "pippnp"]["cfg"]["alpha"] == 0.1 for d in lc if d in nt), "?"
        ),
        ppnpNtLoc=num.get(sum(1 for d in lc if d in nt), "?"),
        ppnpPubLoc=(
            f"{100*T8['PubMed']['cert_local']['1']:.1f}"
            if P8["PubMed", "pippnp"]
            else "--"
        ),
        ppnpPubUb=f"{100*T8['PubMed']['ub1']:.1f}" if P8["PubMed", "pippnp"] else "--",
        ppnpPubAcc=f"{100*T8['PubMed']['acc']:.1f}" if P8["PubMed", "pippnp"] else "--",
        ppnpPubRef=f"{100*selAcc('PubMed'):.1f}",
        ppnpCleanGap=", ".join(f"{NAME.get(d, d)} {100*(selAcc(d)-T8[d]['e_acc']):.1f} of {100*(selAcc(d)-T8[d]['e_cert1']):.1f}" for d in ex if selAcc(d) - T8[d]["e_acc"] > 0.02),
        ppnpCollapse=", ".join(f"{NAME.get(d, d)} {100*min(t['acc'] for t in P8[d, 'pippnp_exact']['trials'] if t['alpha'] == 0.85 and t['loss'] != 'ce'):.1f} vs {100*[t['acc'] for t in P8[d, 'pippnp_exact']['trials'] if t['alpha'] == 0.85 and t['loss'] == 'ce'][0]:.1f}" for d in ex if d in ("Cora", "CiteSeer")),
        ppnpCollapseN=num.get(sum(1 for d in ex if all(t["acc"] < [u["acc"] for u in P8[d, "pippnp_exact"]["trials"] if u["alpha"] == 0.85 and u["loss"] == "ce"][0] for t in P8[d, "pippnp_exact"]["trials"] if t["alpha"] == 0.85 and t["loss"] != "ce")), "?"),
        ppnpLocVsRob=", ".join(NAME.get(d, d) for d in lc if d in nt and T8[d]["cert_local"]["1"] > AN[d]["rob_cert"]) or "none",
        ppnpPubRob=f"{100*AN['PubMed']['rob_cert']:.1f}",
    )
    G10 = json.load(open(os.path.join(R, "g10.json")))
    G10b = json.load(open(os.path.join(R, "g10b.json")))
    macros.update(
        gTenbTiny=str(G10b["tiny_cases"]), gTenbTexas=f"{G10b['texas_max_diff']:.0e}"
    )
# Hypothesis summary (review 4): thresholds transcribed from PLAN_v2/v3/v5; outcomes read from verdict.json and verdict5.json.
VV = dict(Vv)
if os.path.exists(V5p):
    VV.update(V5["verdict"])
yn = lambda h: (
    ("passed" if VV[h].get("passed", VV[h].get("passed_absolute")) else "not met")
    if h in VV
    else "--"
)
HYP = [
    (
        "H1",
        "Certificates never broken by real edits",
        "0 breaks",
        f"{Vv['H1']['broken']} of {Vv['H1']['tried']:,} edits",
        "",
    ),
    (
        "H2",
        "Certified accuracy above Smooth-GCN, $b=1$",
        "$\\ge7/11$, Wilcoxon $p<0.05$",
        f"{Vv['H2']['wins']}/11, $p={Vv['H2']['p']:.4f}$",
        "reversed by Smooth-SAGE (suppl.)",
    ),
    (
        "H3",
        "Small cost vs GCN; beats G-NAMRFF on heterophily",
        "median gap $\\le5$; $\\ge6/8$, $p<0.05$",
        f"{100*Vv['H3']['median_gap_gcn']:.1f}; {Vv['H3']['het_wins']}/8, $p={Vv['H3']['p']:.3f}$",
        "GCN weak on heterophily",
    ),
    (
        "H4",
        "Certified training raises $b=1$ accuracy",
        "$\\ge6$ datasets, $+3$ points",
        f"{Vv['H4']['count']}/11",
        "exactly at threshold",
    ),
    (
        "H5",
        "Top-1 attribution certified",
        "$\\ge$ half the datasets",
        f"{Vv['H5']['count']}/11",
        "absolute part only",
    ),
    (
        "H6",
        "Deterministic certificate 100$\\times$ faster",
        "median $\\ge100\\times$",
        f"{Vv['H6']['median_speedup']:.0f}$\\times$",
        "",
    ),
    (
        "H7",
        "Synthetic feature recovery",
        "$\\ge0.10$ above G-NAMRFF",
        f"{Vv['H7']['certgam']:.2f} vs {Vv['H7']['gnamrff']:.2f}",
        "data follow our model",
    ),
    (
        "H8",
        "Graph pays under attack, $\\pi^\\star\\ge0.5$",
        "$\\ge$ half of useful datasets",
        f"{Vv['H8']['count']}/{Vv['H8']['n']}",
        "robust only on Cora",
    ),
    (
        "H9",
        "Certified training raises $\\pi^\\star$",
        "$\\ge2/3$ of useful datasets",
        f"{Vv['H9']['count']}/{Vv['H9']['n']}",
        "no test, $n<6$",
    ),
    (
        "H10",
        "Radius correlates with degree",
        "$\\rho>0.2$ on $\\ge2/3$",
        f"{Vv['H10']['count']}/{Vv['H10']['n']}",
        "",
    ),
]
if os.path.exists(V5p):
    HYP += [
        (
            "H11",
            "Two-layer SAGE below graph-free, $b=1$",
            "$\\ge7/9$ non-trivial",
            f"{VV['H11']['count']}/{VV['H11']['n']}",
            "no robust training",
        ),
        (
            "H12",
            "Degree $\\le1$ less certified than 4--7",
            "$\\ge80\\%$ of datasets",
            f"{VV['H12']['count']}/{VV['H12']['n']}",
            "two-layer SAGE",
        ),
    ]
if V7:
    VV.update(V7["verdict"])
    HYP += [
        (
            "H14",
            "Shrinkage raises $b=1$ over $\\kappa=0$",
            "$\\ge7/11$, Wilcoxon $p<0.05$",
            f"{VV['H14']['count']}/11, $p={VV['H14']['p']:.3f}$",
            "$\\kappa=0$ chosen on "
            + str(sum(1 for d in ALL if S7[d] and S7[d]["kappa"] == 0)),
        ),
        (
            "H15",
            "ShrinkGAM-R above graph-free, $b=1$",
            "$\\ge5/9$ non-trivial",
            f"{VV['H15']['count']}/{VV['H15']['n']}",
            "Texas already at $\\kappa=0$",
        ),
        (
            "H16",
            "Gain larger at degree $\\le1$ than 4--7",
            "$\\ge70\\%$ of datasets",
            f"{VV['H16']['count']}/{VV['H16']['n']}",
            "zero gains count as no",
        ),
    ]
if V8:  # PLAN_v8 (H18-H20) and PLAN_v8b (H18e, H20e) thresholds
    VV.update(V8["verdict"])
    fr = lambda h: f"{VV[h]['count']}/{VV[h]['n']}"
    HYP += [
        (
            "H18",
            "$\\pi$-PPNP below graph-free, local cert., $b=1$",
            "$\\ge7/9$ non-trivial",
            fr("H18"),
            "roman-empire failed; pre-specified as uninformative",
        ),
        (
            "H18e",
            "$\\pi$-PPNP below graph-free, exact, $b=1$",
            "$\\ge5/6$ small non-trivial",
            fr("H18e"),
            "added after a smoke test on split 1",
        ),
        (
            "H19",
            "$\\pi$-PPNP attack bound below graph-free",
            "$\\ge5/9$ non-trivial",
            fr("H19"),
            "roman-empire failed; PubMed above",
        ),
        (
            "H20",
            "Degree $\\le1$ less certified (local)",
            "$\\ge70\\%$ of datasets",
            fr("H20"),
            "local certificate",
        ),
        (
            "H20e",
            "Degree $\\le1$ less certified (exact)",
            "$\\ge70\\%$ of datasets",
            fr("H20e"),
            "",
        ),
    ]
s = "\\begin{table*}[!t]\n\\centering\\scriptsize\n\\caption{Pre-specified hypotheses. Thresholds were fixed in dated plans before the corresponding runs; H11--H12, H14--H16 and H18--H20 (with H18e, H20e) were added later, each before the corresponding runs. H13 and all supplementary analyses are descriptive.}\\label{tab:hyp}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lllll l}\n\\toprule\n & Claim & Threshold & Result & Outcome & Caveat \\\\\n\\midrule\n"
s += "".join(f"{h} & {c} & {th} & {r} & {yn(h)} & {n} \\\\\n" for h, c, th, r, n in HYP)
s += "\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"
open(os.path.join(G, "tab_hyp.tex"), "w").write(s)
with open(os.path.join(G, "macros.tex"), "w") as f:
    for k, v in macros.items():
        f.write("\\newcommand{\\" + k + "}{" + v.replace(",", "{,}") + "}\n")
open(os.path.join(G, "missing.txt"), "w").write("\n".join(sorted(set(missing))) + "\n")
print("macros", macros)
print("missing", len(set(missing)))
