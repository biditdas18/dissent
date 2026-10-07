"""Emit LaTeX from analysis/results.json so no number is ever hand-typed.

Writes:
    paper/macros.tex   \newcommand for every reported quantity
    paper/table_main.tex

Every numeric claim in the manuscript must cite a macro defined here. If a macro
is undefined, the build fails loudly rather than printing a stale number.
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANA = os.path.join(ROOT, "analysis")
PAPER = os.path.join(ROOT, "paper")


def pct(x, d=1):
    return "--" if x is None or x != x else f"{100*x:.{d}f}"


def num(x, d=3):
    return "--" if x is None or x != x else f"{x:.{d}f}"


def main():
    with open(os.path.join(ANA, "results.json")) as f:
        R = json.load(f)
    os.makedirs(PAPER, exist_ok=True)
    L = []
    # Names that already exist in LaTeX/IEEEtran. \newcommand on these is a fatal
    # build error, so catch it here where the message is obvious.
    RESERVED = {"Delta", "Gamma", "Theta", "Lambda", "Sigma", "Omega", "Phi", "Psi",
                "Pi", "Xi", "ref", "cite", "title", "author", "date"}
    seen = set()

    def mac(name, val):
        assert name not in RESERVED, f"macro name collides with a LaTeX builtin: {name}"
        assert name not in seen, f"duplicate macro: {name}"
        assert name.isalpha(), f"macro name must be letters only: {name}"
        seen.add(name)
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    mac("NPairs", R["n_pairs_complete"])
    mac("NItems", R["n_items_complete"])
    mac("NAgents", R["n_agents"])
    mac("NStimuli", R["n_stimuli_valid"])
    mac("NExcl", R["n_exclusions"])

    # Item-agent pairs (each observed in BOTH media). n_pairs_complete counts
    # item-agent-medium observations, so the pair count is half of it.
    mac("NPairsAgentItem", R["n_pairs_complete"] // 2)

    # Detectability floor of the realised design, computed not recalled: the
    # smallest number of sham-only harmful flips b (with c = 0 no-pressure-only
    # flips, as observed) at which the exact McNemar two-sided p falls below 0.05.
    # p_two = min(1, 2 * 0.5^b) when all discordant pairs fall in one cell.
    b = 1
    while min(1.0, 2 * 0.5 ** b) >= 0.05:
        b += 1
        assert b < 100
    mac("MinFlips", b)

    p = R["primary"]
    mac("DeltaEst", num(p["delta"]))
    mac("DeltaPct", pct(p["delta"]))
    lo, hi = p["delta_ci"]
    mac("DeltaCI", f"[{num(lo)}, {num(hi)}]")
    mac("DeltaCIPct", f"[{pct(lo)}, {pct(hi)}]")
    # Largest effect the interval still admits, in points. The paper uses this to
    # state what the null does NOT rule out, so it must not be rounded by hand.
    mac("DeltaCIHalfPct", pct(max(abs(lo), abs(hi))))
    mac("ExcessNL", num(p["excess_HC_nl"]))
    mac("ExcessNLPct", pct(p["excess_HC_nl"]))
    mac("ExcessCode", num(p["excess_HC_code"]))
    mac("ExcessCodePct", pct(p["excess_HC_code"]))
    for k, tag in (("excess_nl_ci", "ExcessNLCI"), ("excess_code_ci", "ExcessCodeCI")):
        a, b = p[k]
        mac(tag, f"[{num(a)}, {num(b)}]")
        mac(tag + "Pct", f"[{pct(a)}, {pct(b)}]")

    for m in ("nl", "code"):
        M, tag = R["per_medium"][m], m.upper() if m == "nl" else "Code"
        tag = {"nl": "NL", "code": "Code"}[m]
        mac(f"AccInit{tag}Pct", pct(M["acc_init"]))
        for arm in ("none", "sham"):
            a = {"none": "None", "sham": "Sham"}[arm]
            mac(f"HC{tag}{a}Pct", pct(M[arm]["hc_rate"]))
            mac(f"HC{tag}{a}K", M[arm]["hc_k"])
            mac(f"HC{tag}{a}N", M[arm]["hc_n"])
            mac(f"Flip{tag}{a}Pct", pct(M[arm]["flip_rate"]))
            mac(f"ToW{tag}{a}Pct", pct(M[arm]["to_W_rate_given_init_correct"]))
            mac(f"Benefit{tag}{a}Pct", pct(M[arm]["benefit_rate"]))
            mac(f"Acc{tag}{a}Pct", pct(M[arm]["acc_after"]))
        mc = M["mcnemar"]
        mac(f"McN{tag}B", mc["b_sham_only_harm"])
        mac(f"McN{tag}C", mc["c_none_only_harm"])
        mac(f"McN{tag}P", num(mc["p"], 4))
        if m == "code" and M.get("init_exec_consistency") is not None:
            mac("ExecConsistPct", pct(M["init_exec_consistency"]))

    tot = sum(v for d in R["tokens"].values() for v in d.values())
    mac("TotalTokens", f"{tot:,}")

    # ---- roster audit: refusal-artifact counts recounted from the primary run log,
    # so D3's figures in the paper are generated rather than transcribed. D3 itself
    # records the FIRST-PASS counts at the decision point (74/149); these are the
    # final file totals a reader recounting the log will see.
    rl = os.path.join(ROOT, "data", "runs", "conformity_v1.jsonl")
    if os.path.exists(rl):
        import collections
        per = collections.defaultdict(collections.Counter)
        for line in open(rl):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            per[str(rec.get("model"))][str(rec.get("stop_reason"))] += 1
        dropped = [m for m in per if "sonnet" in m]
        kept = [m for m in per if m not in dropped]
        if dropped:
            d = dropped[0]
            mac("RefusalAgentCalls", sum(per[d].values()))
            mac("RefusalAgentK", per[d].get("refusal", 0))
        mac("RetainedCalls", sum(sum(per[m].values()) for m in kept))
        mac("RetainedRefusals", sum(per[m].get("refusal", 0) for m in kept))

        # Per-medium refusal rate for the removed agent. The removal decision in D3
        # rested on this rate being strongly differential across media, which would
        # bias the primary (medium) contrast through differential missingness.
        # The log has no `medium` field; medium is carried by `phase`/`parse_mode`.
        bymed = collections.defaultdict(collections.Counter)
        if dropped:
            for line in open(rl):
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if str(rec.get("model")) != dropped[0]:
                    continue
                tag = str(rec.get("parse_mode")) + str(rec.get("phase"))
                bymed["Code" if "code" in tag else "NL"][str(rec.get("stop_reason"))] += 1
            for lab, c in bymed.items():
                n = sum(c.values())
                k = c.get("refusal", 0)
                mac(f"RefusalAgent{lab}N", n)
                mac(f"RefusalAgent{lab}K", k)
                mac(f"RefusalAgent{lab}Pct", pct(k / n) if n else "n/a")

    # ---- exclusion breakdown by pre-registered reason, from analysis/exclusions.csv
    ex = os.path.join(ANA, "exclusions.csv")
    if os.path.exists(ex):
        import csv as _csv
        import collections as _c
        reasons = _c.Counter(r["reason"] for r in _csv.DictReader(open(ex)))
        mac("ExclRemovedAgent", reasons.get("agent_removed_D3_refusal_artifact", 0))
        mac("ExclRevisionMissing", reasons.get("revision_missing_or_unparsed", 0))
        # A reviewer read the exclusion total against the observed total and found
        # them incompatible. They have different denominators, so emit both so the
        # paper can say exactly what each counts. The removed-agent rows are
        # pair-level (arm="both"); the revision rows are arm-level, so they exceed
        # the number of distinct pairs they cover.
        import csv as _csv
        _xp = os.path.join(ANA, "exclusions.csv")
        if os.path.exists(_xp):
            _rows = list(_csv.DictReader(open(_xp)))
            _rev = [r for r in _rows if r["reason"] != "agent_removed_D3_refusal_artifact"]
            mac("ExclRevisionPairs", len({(r["qid"], r["agent"], r["medium"]) for r in _rev}))
        mac("CellsDesignRetained", R["n_pairs_total"])

    # ---- item-pool difficulty screening (analysis/pool_screening.json)
    # Initial (pre-pressure) accuracy per pool and medium. This is a property of
    # the items and the roster, measured before any pressure condition, and it
    # bounds what the design can detect: at 100% initial accuracy the beneficial
    # revision arm is empty by construction.
    POOLS = [("gsm8k_main", "GSM", "GSM8K (primary)"),
             ("gsmhard_pilot", "Hard", "GSM-Hard (screen)"),
             ("mmlupro_fixed", "MMLU", "MMLU-Pro quant.\\ (screen)")]
    prow = []
    sp = os.path.join(ANA, "pool_screening.json")
    if os.path.exists(sp):
        with open(sp) as f:
            S = {(d["pool"], d["medium"]): d for d in json.load(f)}
        for key, tag, label in POOLS:
            cells = []
            for m in ("nl", "code"):
                d = S.get((key, m))
                if d is None:
                    cells += ["--", "--"]
                    continue
                mac(f"Pool{tag}{'NL' if m=='nl' else 'Code'}N", d["n"])
                mac(f"Pool{tag}{'NL' if m=='nl' else 'Code'}Pct", pct(d["acc"]))
                # items behind each screen = graded responses / agents; the MMLU-Pro
                # screen is only 2 items, so the paper must never state its rate bare.
                if d["n"] and R.get("n_agents"):
                    mac(f"Pool{tag}{'NL' if m=='nl' else 'Code'}Items", int(d["n"] // R["n_agents"]))
                    mac(f"Pool{tag}{'NL' if m=='nl' else 'Code'}K", int(d["correct"]))
                cells += [str(d["n"]), pct(d["acc"]) + "\\%"]
            prow.append(f"{label} & " + " & ".join(cells) + " \\\\")
        with open(os.path.join(PAPER, "table_pools.tex"), "w") as f:
            f.write("% AUTO-GENERATED by src/make_tables.py -- do not edit.\n")
            f.write("\\begin{tabular}{lrrrr}\n\\toprule\n"
                    "& \\multicolumn{2}{c}{natural language} & "
                    "\\multicolumn{2}{c}{executable code} \\\\\n"
                    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
                    "Item pool & $n$ & initial acc. & $n$ & initial acc. \\\\\n"
                    "\\midrule\n" + "\n".join(prow) + "\n\\bottomrule\n\\end{tabular}\n")

    with open(os.path.join(PAPER, "macros.tex"), "w") as f:
        f.write("% AUTO-GENERATED by src/make_tables.py -- do not edit.\n")
        f.write("\n".join(L) + "\n")

    rows = []
    for m in ("nl", "code"):
        label = {"nl": "Natural language", "code": "Executable code"}[m]
        M = R["per_medium"][m]
        for arm in ("none", "sham"):
            d = M[arm]
            # The benefit column is deliberately not emitted. Every cell of it was
            # "--%": initial accuracy is at ceiling, so there are no initially-wrong
            # pairs and benefit_n == 0 in every condition. Six dashes in a column
            # pushed the table ~26pt past the measure and collided with the
            # neighbouring caption. The fact is stated in the caption instead, where
            # it can say why the arm is empty. Asserted here so that a future run
            # with headroom fails loudly rather than silently losing a column.
            assert d["benefit_n"] == 0, (
                f"benefit arm is non-empty (n={d['benefit_n']}) for {m}/{arm}: "
                "restore the benefit column in table_main and widen the table")
            rows.append(f"{label if arm=='none' else ''} & "
                        f"{'no-pressure' if arm=='none' else 'sham peer'} & "
                        f"{d['hc_n']} & {d['hc_k']} & {pct(d['hc_rate'])}\\% & "
                        f"{pct(d['to_W_rate_given_init_correct'])}\\% \\\\")
        rows.append(f"\\multicolumn{{2}}{{l}}{{\\emph{{excess harmful conformity}}}} & "
                    f"\\multicolumn{{4}}{{l}}{{{pct(M['excess_HC'])}\\% }} \\\\")
        if m == "nl":
            rows.append("\\midrule")
    with open(os.path.join(PAPER, "table_main.tex"), "w") as f:
        f.write("% AUTO-GENERATED by src/make_tables.py -- do not edit.\n")
        # No font or spacing override here on purpose. The seven-column version
        # overran the column by ~26pt and collided with the neighbouring table's
        # caption; dropping the all-dash benefit column above was by itself enough
        # to bring it inside the measure (verified: zero overfull boxes). Shrinking
        # the type instead would have made this table smaller than Table II, since
        # IEEEtran already reduces table bodies inside the float.
        f.write("\\begin{tabular}{llrrrr}\n\\toprule\n"
                "Medium & Condition & $n$ & harmed & HC rate & $\\to W$ \\\\\n"
                "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {len(L)} macros -> paper/macros.tex, paper/table_main.tex")


if __name__ == "__main__":
    main()
