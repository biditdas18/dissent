"""Regenerate every figure in the paper and the README from analysis artefacts.

    python src/make_figures.py

Reads  analysis/results.json, analysis/pool_screening.json
Writes figures/fig1_null_and_saturation.{png,pdf}   (paper, Figure 1)
       figures/fig2_protocol.png                    (README illustration)

No number is typed into this file; every plotted value is read from the analysis
artefacts. The one exception is the detectability floor annotation, which is recomputed
here from the exact McNemar expression rather than recalled.
"""
import json, os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANA = os.path.join(ROOT, "analysis")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

NL, CODE, GREY = "#2b6cb0", "#b7791f", "#888888"

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def min_flips_for_significance(alpha=0.05, c=0, cap=60):
    """Smallest b (sham-only harmful flips) with exact two-sided McNemar p < alpha
    when the no-peer arm contributes c discordant cells. Computed, not recalled."""
    for b in range(1, cap + 1):
        n = b + c
        p = min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
        if p < alpha:
            return b
    return None


def fig1(R, S):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7, 2.7))
    P = R["primary"]

    # ---- panel a: the pre-registered contrast -------------------------------
    pts = [(0, 100 * P["excess_HC_nl"], [100 * x for x in P["excess_nl_ci"]], NL),
           (1, 100 * P["excess_HC_code"], [100 * x for x in P["excess_code_ci"]], CODE)]
    axA.axhline(0, color=GREY, lw=0.8, zorder=1)
    for x, y, ci, col in pts:
        axA.errorbar(x, y, yerr=[[y - ci[0]], [ci[1] - y]], fmt="o", ms=4.5,
                     color=col, ecolor=col, elinewidth=1.2, capsize=3.5, zorder=3)
    axA.set_xticks([0, 1])
    axA.set_xticklabels(["natural\nlanguage", "executable\ncode"])
    axA.set_xlim(-0.55, 1.55)
    axA.set_ylim(-1.6, 13.5)
    axA.set_ylabel("excess harmful conformity\n(sham $-$ no peer, pts)")
    axA.text(0.5, 0.94, f"$\\Delta$ = {100*P['delta']:.1f} pts\n"
                        f"95% CI [{100*P['delta_ci'][0]:.1f}, {100*P['delta_ci'][1]:.1f}]",
             transform=axA.transAxes, ha="center", va="top", fontsize=7)
    axA.text(-0.18, 1.02, "a", transform=axA.transAxes, ha="left", va="bottom",
             fontsize=9, fontweight="bold")

    # ---- panel b: why panel a is underpowered -------------------------------
    rows = {(r["pool"], r["medium"]): r for r in S}
    pools = [("gsm8k_main", "GSM8K\n(primary)"),
             ("gsmhard_pilot", "GSM-Hard\n(screen)"),
             ("mmlupro_fixed", "MMLU-Pro\nquant. (screen)")]
    labels = []
    for i, (pool, lab) in enumerate(pools):
        for dx, med, col in [(-0.13, "nl", NL), (0.13, "code", CODE)]:
            d = rows[(pool, med)]
            axB.plot(i + dx, 100 * d["acc"], "o", ms=4.5, color=col, zorder=3)
        labels.append(f"{lab}\n$n$={rows[(pool,'nl')]['n']}, {rows[(pool,'code')]['n']}")
    axB.axhline(100, color=GREY, ls="--", lw=0.8, zorder=1)
    axB.set_xticks(range(len(pools)))
    axB.set_xticklabels(labels)
    axB.set_xlim(-0.45, 2.42)
    axB.set_ylim(0, 118)
    axB.set_ylabel("initial accuracy, before any\npressure condition (%)")
    axB.text(0.03, 0.975, "ceiling: benefit arm empty", transform=axB.transAxes,
             ha="left", va="top", fontsize=6, color=GREY)
    axB.text(1.22, 112, "natural language", fontsize=6, color=NL, ha="left", va="center")
    axB.text(1.30, 80, "executable code", fontsize=6, color=CODE, ha="left", va="center")
    axB.text(-0.18, 1.02, "b", transform=axB.transAxes, ha="left", va="bottom",
             fontsize=9, fontweight="bold")

    fig.tight_layout(w_pad=2.0)
    for ext in ("png", "pdf"):
        # CreationDate=None suppresses the timestamp matplotlib embeds in PDF
        # output. Without it this file differs on every run and `git status` is
        # dirty after reproduce.sh, which contradicts the byte-identical
        # reproduction this repository claims. PNG carries no timestamp, and
        # matplotlib only accepts the metadata key for the PDF backend.
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(os.path.join(FIG, f"fig1_null_and_saturation.{ext}"),
                    metadata=meta)
    plt.close(fig)
    return min_flips_for_significance()


def fig2(R):
    """Protocol schematic: six phases per item, two media x two pressure arms."""
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    ax.set_xlim(0, 110); ax.set_ylim(0, 44); ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec="#444444", fs=6.6, lw=0.7, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                zorder=3, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, col="#666666"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=7, lw=0.7, color=col, zorder=1,
                                     shrinkA=0, shrinkB=0))

    NLT, CDT = "#eaf1f8", "#faf3e4"
    box(1, 17, 13, 9, f"item $i$\n{R['n_items_complete']} GSM8K items\ngold $g$", fc="#f2f2f2")

    # two media lanes
    box(19, 28, 20, 9, "PROSE\nagent answers in\nnatural language", fc=NLT, ec=NL)
    box(19, 6, 20, 9, "CODE\nagent answers with an\nexecutable program", fc=CDT, ec=CODE)
    arrow(14.4, 23, 18.6, 32); arrow(14.4, 20, 18.6, 11)

    # initial answer
    box(43, 28, 16, 9, "initial answer\n$a_0$, graded", fc=NLT, ec=NL)
    box(43, 6, 16, 9, "initial answer\n$a_0$, graded", fc=CDT, ec=CODE)
    arrow(39.4, 32.5, 42.6, 32.5); arrow(39.4, 10.5, 42.6, 10.5)

    # two pressure arms per lane
    for y, ec, fc in [(28, NL, NLT), (6, CODE, CDT)]:
        box(63, y + 6.0, 27, 5.0, "no-peer arm: re-asked alone", fc="white", ec=ec, fs=6.0)
        box(63, y - 2.0, 27, 5.0, "sham arm: 2 peers assert $W \\neq g$", fc=fc, ec=ec, fs=6.0)
        arrow(59.4, y + 4.5, 62.6, y + 8.5); arrow(59.4, y + 4.5, 62.6, y + 0.5)

    box(94, 17, 13, 9, "revised\nanswer $a_1$", fc="#f2f2f2")
    arrow(90.4, 32.5, 93.6, 24.5); arrow(90.4, 10.5, 93.6, 18.5)

    ax.text(55, 41.6, "six phases per item, run in frozen order; medium is the manipulated factor",
            ha="center", va="center", fontsize=6.6, color="#444444")
    ax.text(55, 0.6,
            "harmful conformity: $a_0=g$ and $a_1=W$.   excess HC = HC(sham) $-$ HC(no peer)."
            "   estimand $\\Delta$ = excess HC(code) $-$ excess HC(prose)",
            ha="center", va="center", fontsize=6.6, color="#222222")
    fig.savefig(os.path.join(FIG, "fig2_protocol.png"))
    plt.close(fig)


if __name__ == "__main__":
    R = json.load(open(os.path.join(ANA, "results.json")))
    S = json.load(open(os.path.join(ANA, "pool_screening.json")))
    mf = fig1(R, S)
    fig2(R)
    print(f"wrote figures/fig1_null_and_saturation.{{png,pdf}} and figures/fig2_protocol.png")
    print(f"detectability floor (exact McNemar, c=0, alpha=0.05): b = {mf} sham-only flips")
