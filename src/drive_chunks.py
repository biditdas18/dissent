"""Run the six-phase protocol item-chunk by item-chunk instead of phase by phase.

Why: the inference budget may be exhausted before all items finish. Phase-major
execution would then leave every item with initial answers but missing revisions --
zero analysable pairs. Chunk-major execution completes all six phases for a chunk
before starting the next, so a truncated run yields a smaller but FULLY OBSERVED
dataset.

Items are taken in frozen-sample order. Truncation is therefore by budget position,
not by outcome: no item's data is inspected before deciding whether to run it.

Usage (from the python kernel, which supplies `host`):
    import drive_chunks as D; D.host = host; D.main(chunk=4)
"""
import os, json, time, collections

import run_conformity as R
import conformity as C

host = None  # injected by the caller

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "data", "runs")


def token_total(run_id):
    p = os.path.join(RUNS, f"{run_id}.jsonl")
    if not os.path.exists(p):
        return 0
    t = 0
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            t += (r.get("in_tok") or 0) + (r.get("out_tok") or 0)
    return t


def complete_pairs(run_id, roster_agents):
    """Item-agent pairs observed in ALL six phases (the analysable unit)."""
    p = os.path.join(RUNS, f"{run_id}.jsonl")
    if not os.path.exists(p):
        return set()
    seen = collections.defaultdict(set)
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("stop_reason") == "end_turn" and r.get("parsed_answer") is not None \
                    and r["agent_id"] in roster_agents:
                seen[(r["qid"], r["agent_id"])].add(r["phase"])
    need = {"init_nl", "init_code", "rev_nl_none", "rev_nl_sham",
            "rev_code_none", "rev_code_sham"}
    return {k for k, v in seen.items() if need <= v}


def main(chunk=4, run_id="conformity_v1", budget_tokens=None, verbose=True):
    R.host = host
    import pipeline as P
    P.host = host

    # Pool follows run_conformity's configuration, so switching pools requires
    # setting R.DATASET / R.SAMPLE / R.STIM in one place only.
    stim = C.load_stimuli(R.STIM)
    qs = [json.loads(l) for l in open(R.SAMPLE)]
    qids = [q["qid"] for q in qs if q["dataset"] == R.DATASET and q["qid"] in stim]
    agents = {a for a, _ in R.ROSTER}

    spent0 = token_total(run_id)
    if verbose:
        print(f"items available: {len(qids)} | agents: {sorted(agents)} | "
              f"tokens already spent: {spent0:,}", flush=True)

    for i in range(0, len(qids), chunk):
        part = qids[i:i + chunk]
        spent = token_total(run_id)
        if budget_tokens is not None and spent >= budget_tokens:
            print(f"STOP: token budget reached ({spent:,} >= {budget_tokens:,}) "
                  f"after {i} items", flush=True)
            break
        t0 = time.time()
        try:
            R.main(run_id=run_id, batch=len(part) * 2, verbose=False, qids=part)
        except Exception as e:
            print(f"CHUNK {i}-{i+len(part)} FAILED: {type(e).__name__} {repr(e)[:180]}",
                  flush=True)
            print("  stopping; completed chunks are already on disk", flush=True)
            break
        done = complete_pairs(run_id, agents)
        print(f"chunk {i:>3}-{i+len(part):<3} | {time.time()-t0:>5.0f}s | "
              f"tokens {token_total(run_id):>9,} | complete pairs {len(done):>3}",
              flush=True)

    done = complete_pairs(run_id, agents)
    print(f"\nFINAL: {len(done)} fully-observed item-agent pairs across "
          f"{len({q for q, _ in done})} items | tokens {token_total(run_id):,}",
          flush=True)
    return done
