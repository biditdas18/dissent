"""Run the conformity experiment (protocol `conformity-v1`).

Phase order (each resumable, append-only):
    1. init_nl                       independent solve, natural language
    2. init_code                     independent solve, executable program
       -> execute all generated programs, persist results
    3. rev_nl_none      / rev_nl_sham
    4. rev_code_none    / rev_code_sham

Usage from the kernel:
    import run_conformity as RC; RC.host = host
    RC.main(limit=None)
"""
import os, sys, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pipeline as P
import conformity as C

host = None  # injected by the caller

RUNS = os.path.join(ROOT, "data", "runs")
# Pool configuration. Overridden by the caller for the mmlupro_quant pool (D5):
# the gsm8k pool is saturated for this roster (239/240 initial answers correct),
# leaving the benefit arm empty and the harm base rate at the floor.
DATASET = "gsm8k"
SAMPLE = os.path.join(ROOT, "data", "samples", "all_questions.jsonl")
STIM = os.path.join(ROOT, "data", "stimuli", "sham_gsm8k.jsonl")

# agent_B (claude-sonnet-4-5) was REMOVED after the first run: via the platform
# inference interface it returned stop_reason="refusal" with output truncated
# mid-sentence on 74/149 calls (61% in the code medium, 36% in prose), while
# agent_A and agent_C refused 0/215. The refused content was ordinary GSM8K
# arithmetic, so this is a response-handling artifact, not a model refusal, and
# the pipeline's retries did not clear it (82 records for 43 items).
# The exclusion was decided on stop_reason patterns alone, before any outcome
# (harmful-conformity) quantity was computed. Logged as D3 in DEVIATIONS.md.
ROSTER = [("agent_A", "claude-haiku-4-5-20251001"),
          ("agent_C", "claude-opus-4-5-20251101")]


def make_backends():
    """Direct Anthropic endpoint when a key is present, else the platform endpoint."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return [P.AnthropicBackend(a, m) for a, m in ROSTER], "anthropic_direct"
    return [P.HostBackend(a, m, host) for a, m in ROSTER], "host_platform"


def main(limit=None, run_id="conformity_v1", batch=25, verbose=True, qids=None):
    """qids: restrict to these question ids. Used by the chunked driver to complete
    ALL phases for a subset of items, so a budget-truncated run still yields
    fully-observed item-agent pairs rather than complete inits with no revisions."""
    os.makedirs(RUNS, exist_ok=True)
    out = os.path.join(RUNS, f"{run_id}.jsonl")
    exec_out = os.path.join(RUNS, f"{run_id}_exec.jsonl")

    qs = [json.loads(l) for l in open(SAMPLE) if json.loads(l)["dataset"] == DATASET]
    stim = C.load_stimuli(STIM)

    # Restrict every phase to items with a VALIDATED sham stimulus. The primary
    # estimand is excess_HC = HC(sham) - HC(none) within medium, which is only
    # defined on items that have both arms; running the none-arm on items lacking
    # a stimulus would make the two arms non-comparable item sets. Items are
    # ordered by the frozen sample, so this is not selection on outcomes.
    qs = [q for q in qs if q["qid"] in stim]
    if qids is not None:
        keep = set(qids)
        qs = [q for q in qs if q["qid"] in keep]
    if limit:
        qs = qs[:limit]
    backends, mode = make_backends()
    if verbose:
        print(f"backend mode: {mode} | questions: {len(qs)} | stimuli ok: {len(stim)}", flush=True)

    # ---- phase 1/2: independent answers in each medium
    for medium, mt in (("nl", C.MAX_TOKENS_NL), ("code", C.MAX_TOKENS_CODE)):
        P.run_phase(qs, backends, f"init_{medium}",
                    lambda q, aid, m=medium: C.build_init(q, m),
                    mt, out, run_id, batch=batch, verbose=verbose)

    # ---- execute the generated programs
    C.execute_programs(out, exec_out, "init_code", verbose=verbose)
    ex = C.load_exec(exec_out)

    # ---- phases 3/4: revision under each pressure condition
    init = P.load_done(out)
    for medium, mt in (("nl", C.MAX_TOKENS_NL), ("code", C.MAX_TOKENS_CODE)):
        for pressure in ("none", "sham"):
            phase = f"rev_{medium}_{pressure}"
            for be in backends:
                def pf(q, aid, m=medium, pr=pressure):
                    r = init[(f"init_{m}", q["qid"], aid)]
                    own_exec = ex.get((f"init_{m}", q["qid"], aid)) if m == "code" else None
                    return C.build_rev(q, m, pr, r["raw_response"], r["parsed_answer"],
                                       own_exec=own_exec, stim=stim.get(q["qid"]))
                elig = [q for q in qs
                        if (f"init_{medium}", q["qid"], be.agent_id) in init
                        and init[(f"init_{medium}", q["qid"], be.agent_id)].get("parsed_answer") is not None
                        and (pressure == "none" or q["qid"] in stim)]
                if verbose:
                    print(f"[{phase}] {be.agent_id}: {len(elig)} eligible of {len(qs)}", flush=True)
                if elig:
                    P.run_phase(elig, [be], phase, pf, mt, out, run_id,
                                batch=batch, verbose=verbose)

    if verbose:
        print("done:", out, flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-id", default="conformity_v1")
    a = ap.parse_args()
    main(limit=a.limit, run_id=a.run_id)
