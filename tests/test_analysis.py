"""Synthetic check of the pre-registered estimand. Known inputs -> known Delta."""
import os, sys, json, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import analyze as A

def rec(phase, qid, aid, ans, gold):
    return {"phase": phase, "qid": qid, "agent_id": aid, "attempt": 0, "gold": gold,
            "parsed_answer": ans, "correct": ans == gold, "in_tok": 10, "out_tok": 10,
            "stop_reason": "end_turn", "raw_response": "x", "dataset": "gsm8k", "qtype": "numeric"}

def build(tmp, spec, gold="100", W="50"):
    """spec[(qid,agent,medium)] = (init, none, sham) answers."""
    runs = os.path.join(tmp, "data", "runs"); os.makedirs(runs, exist_ok=True)
    stimd = os.path.join(tmp, "data", "stimuli"); os.makedirs(stimd, exist_ok=True)
    rp = os.path.join(runs, "t.jsonl")
    qids = sorted({k[0] for k in spec})
    with open(rp, "w") as f:
        for (qid, aid, m), (i, n, s) in spec.items():
            f.write(json.dumps(rec(f"init_{m}", qid, aid, i, gold)) + "\n")
            f.write(json.dumps(rec(f"rev_{m}_none", qid, aid, n, gold)) + "\n")
            f.write(json.dumps(rec(f"rev_{m}_sham", qid, aid, s, gold)) + "\n")
    sp = os.path.join(stimd, "sham_gsm8k.jsonl")
    with open(sp, "w") as f:
        for q in qids:
            f.write(json.dumps({"qid": q, "status": "ok", "W": W, "gold": gold,
                                "nl_peers": ["a", "b"], "code_peers": []}) + "\n")
    return rp, os.path.join(runs, "t_exec.jsonl"), sp

tmp = tempfile.mkdtemp()
try:
    # 4 items, 1 agent. All initially CORRECT in both media.
    # nl:   sham harms 3/4, none harms 1/4  -> excess_HC(nl)   = 0.50
    # code: sham harms 1/4, none harms 0/4  -> excess_HC(code) = 0.25
    # Delta = 0.25
    spec = {}
    nl_sham  = ["50", "50", "50", "100"]
    nl_none  = ["50", "100", "100", "100"]
    cd_sham  = ["50", "100", "100", "100"]
    cd_none  = ["100", "100", "100", "100"]
    for i in range(4):
        q = f"q{i}"
        spec[(q, "agent_A", "nl")]   = ("100", nl_none[i], nl_sham[i])
        spec[(q, "agent_A", "code")] = ("100", cd_none[i], cd_sham[i])
    rp, ep, sp = build(tmp, spec)
    pairs, excl, stim = A.build_pairs(rp, ep, sp)
    comp = [r for r in pairs if r["complete"]]
    assert len(comp) == 8, len(comp)
    assert not excl, excl
    for m, exp in (("nl", 0.50), ("code", 0.25)):
        s = A.hc_rate(comp, m, "sham")[0]; n = A.hc_rate(comp, m, "none")[0]
        got = round(s - n, 10)
        assert got == exp, f"excess_HC({m}) = {got}, expected {exp}"
    d, e = A._delta_from(comp)
    assert round(d, 10) == 0.25, d
    boot = A.cluster_bootstrap(comp, n=2000, seed=1)
    lo, hi = boot["delta_ci"]
    assert lo <= 0.25 <= hi, (lo, hi)
    mc = A.mcnemar_exact(comp, "nl")
    assert (mc["b_sham_only_harm"], mc["c_none_only_harm"]) == (2, 0), mc
    print("PASS  excess_HC(nl)=0.50  excess_HC(code)=0.25  Delta=0.25")
    print(f"PASS  bootstrap 95% CI covers truth: [{lo:.3f}, {hi:.3f}]")
    print(f"PASS  McNemar nl b=2 c=0 p={mc['p']:.3f}")
finally:
    shutil.rmtree(tmp)
