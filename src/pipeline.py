"""
DISSENT — Does initial cross-model disagreement predict when multi-agent debate helps?
Core experiment pipeline. Inference-only, resumable, append-only JSONL persistence.

All design constants in this file are FROZEN BEFORE ANY MODEL CALL (see PROTOCOL below).
"""
import os, json, re, time, hashlib, random, datetime

# ----------------------------------------------------------------------------- PROTOCOL
PROTOCOL_VERSION = "v1"
TEMPERATURE      = 0.0
MAX_TOKENS_R1    = 800
MAX_TOKENS_R2    = 900
# Predetermined tie-break for 3-way complete disagreement (no majority exists):
#   PRIMARY   : group answer = answer of the highest-priority agent in the FIXED roster order
#   SENSITIVITY: ties scored incorrect (reported alongside, never instead)
TIEBREAK_PRIMARY = "roster_priority"
# Deterministic per-question permutation of peer presentation order (position-bias control)
PEER_ORDER_SEED  = 20260914

# ----------------------------------------------------------------------------- PROMPTS
SYS = ("You are a careful expert reasoner. Reason step by step, but keep your reasoning "
       "under 150 words. You MUST end your reply with a line of exactly the form "
       "'FINAL: <answer>' and nothing after it.")

def _fmt_question(q):
    if q["type"] == "numeric":
        return (f"Solve this problem.\n\n{q['question']}\n\n"
                "End with 'FINAL: <number>' giving only the final numeric answer "
                "(digits only, no units, no commas, no currency symbols).")
    opts = "\n".join(f"({k}) {v}" for k, v in q["choices"].items())
    letters = "/".join(q["choices"].keys())
    return (f"Answer this multiple-choice question.\n\n{q['question']}\n\nOptions:\n{opts}\n\n"
            f"End with 'FINAL: <letter>' where <letter> is one of {letters}.")

def build_r1_prompt(q):
    return _fmt_question(q)

def peer_order(qid, n_peers):
    """Deterministic, question-specific permutation of peer presentation order."""
    seed = int(hashlib.sha256(f"{PEER_ORDER_SEED}:{qid}".encode()).hexdigest()[:12], 16)
    order = list(range(n_peers))
    random.Random(seed).shuffle(order)
    return order

def build_r2_prompt(q, own_text, own_ans, peers, show_reasoning=True):
    """peers: list of dicts {answer, text} for the OTHER agents, already in roster order.
    Presentation order is permuted deterministically per question. Peers are anonymised."""
    order = peer_order(q["qid"], len(peers))
    labels = ["Agent X", "Agent Y", "Agent Z"]
    blocks = []
    for slot, pi in enumerate(order):
        p = peers[pi]
        if show_reasoning:
            blocks.append(f"--- {labels[slot]} ---\nReasoning: {p['text'].strip()}\nAnswer: {p['answer']}")
        else:
            blocks.append(f"--- {labels[slot]} ---\nAnswer: {p['answer']}")
    peer_blob = "\n\n".join(blocks)
    return (f"{_fmt_question(q)}\n\n"
            f"=== YOUR PREVIOUS RESPONSE ===\nReasoning: {own_text.strip()}\nAnswer: {own_ans}\n\n"
            f"=== OTHER AGENTS' RESPONSES ===\n{peer_blob}\n\n"
            "=== TASK ===\nCritically evaluate the competing solutions above, including your own. "
            "Identify any error in your own reasoning or in the others'. Do not defer to the majority "
            "simply because it is the majority, and do not change your answer unless you find a concrete "
            "reason to. State your updated reasoning, then give your final answer for this round. "
            "End with 'FINAL: <answer>' in the same format as before.")

# ----------------------------------------------------------------------------- PARSING
_FINAL_MC  = re.compile(r"FINAL\s*[:\-]?\s*\(?\s*([A-J])\s*\)?", re.I)
_FINAL_NUM = re.compile(r"FINAL\s*[:\-]?\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", re.I)
_FB_MC     = re.compile(r"(?:answer\s+is|answer\s*[:=])\s*\(?\s*([A-J])\s*\)?", re.I)
_FB_NUM    = re.compile(r"(?:answer\s+is|answer\s*[:=])\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", re.I)

def parse_answer(text, qtype, valid_letters=None):
    """Return (answer_str_or_None, how). Never guesses; failures are logged, not repaired."""
    if not text:
        return None, "empty"
    if qtype == "mc":
        for rx, how in ((_FINAL_MC, "final"), (_FB_MC, "fallback")):
            m = rx.findall(text)
            if m:
                a = m[-1].upper()
                if valid_letters is None or a in valid_letters:
                    return a, how
        return None, "no_match"
    for rx, how in ((_FINAL_NUM, "final"), (_FB_NUM, "fallback")):
        m = rx.findall(text)
        if m:
            return norm_num(m[-1]), how
    return None, "no_match"

def norm_num(s):
    s = str(s).replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else repr(f)
    except ValueError:
        return s

def is_correct(pred, gold, qtype):
    if pred is None:
        return False
    if qtype == "mc":
        return pred.strip().upper() == str(gold).strip().upper()
    try:
        return abs(float(pred) - float(norm_num(gold))) < 1e-6
    except ValueError:
        return pred.strip() == norm_num(gold)

# ----------------------------------------------------------------------------- AGGREGATION
def aggregate(answers, roster):
    """answers: dict agent_id -> answer (may be None for parse failure).
    Returns (group_answer_primary, group_answer_strict, pattern, disagreement_score, counts).
    Parse failures are kept as a distinct '<PARSE_FAIL>' vote so they cannot silently
    inflate agreement."""
    votes = [answers.get(a) if answers.get(a) is not None else "<PARSE_FAIL>" for a in roster]
    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.values())
    D = 1.0 - top / len(votes)
    if top == len(votes):
        pattern = "unanimous"
    elif top == 1:
        pattern = "complete_disagreement"
    else:
        pattern = "majority_split"
    winners = [v for v, c in counts.items() if c == top]
    if len(winners) == 1:
        primary = strict = winners[0]
    else:
        # no majority: PRIMARY = highest-priority agent in fixed roster order; STRICT = unresolved
        primary = votes[0]
        strict = "<NO_MAJORITY>"
    return primary, strict, pattern, D, counts

# ----------------------------------------------------------------------------- BACKENDS
class Backend:
    """Pluggable LLM backend. Each returns list of dicts: {text, in_tok, out_tok, error}."""
    def __init__(self, agent_id, model, provider):
        self.agent_id, self.model, self.provider = agent_id, model, provider
    def generate(self, prompts, max_tokens):
        raise NotImplementedError

class HostBackend(Backend):
    """Anthropic-family models via the platform in-kernel API (host.llm)."""
    def __init__(self, agent_id, model, host):
        super().__init__(agent_id, model, "anthropic")
        self.host = host
    def generate(self, prompts, max_tokens):
        reqs = [{"prompt": p, "system": SYS, "model": self.model,
                 "max_tokens": max_tokens, "temperature": TEMPERATURE} for p in prompts]
        raw = self.host.llm(reqs, max_concurrency=8)
        out = []
        for r in raw:
            if not isinstance(r, dict) or r.get("error"):
                out.append({"text": None, "in_tok": 0, "out_tok": 0, "stop_reason": "error",
                            "error": str(r.get("error") if isinstance(r, dict) else r)[:300]})
                continue
            u = r.get("usage") or {}
            out.append({"text": r.get("text"),
                        "in_tok": u.get("input_tokens", 0) or 0,
                        "out_tok": u.get("output_tokens", 0) or 0,
                        "stop_reason": r.get("stop_reason"), "error": None})
        return out

class AnthropicBackend(Backend):
    """Anthropic models via the raw provider endpoint (no platform-side classifier)."""
    def __init__(self, agent_id, model):
        super().__init__(agent_id, model, "anthropic")
        import anthropic
        self.cli = anthropic.Anthropic()
    def generate(self, prompts, max_tokens):
        from concurrent.futures import ThreadPoolExecutor
        def one(p):
            try:
                r = self.cli.messages.create(
                    model=self.model, max_tokens=max_tokens, temperature=TEMPERATURE,
                    system=SYS, messages=[{"role": "user", "content": p}])
                txt = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                return {"text": txt, "in_tok": r.usage.input_tokens,
                        "out_tok": r.usage.output_tokens, "stop_reason": r.stop_reason,
                        "error": None}
            except Exception as e:
                return {"text": None, "in_tok": 0, "out_tok": 0, "stop_reason": "error",
                        "error": repr(e)[:300]}
        with ThreadPoolExecutor(max_workers=8) as ex:
            return list(ex.map(one, prompts))

class OpenAIBackend(Backend):
    def __init__(self, agent_id, model):
        super().__init__(agent_id, model, "openai")
        from openai import OpenAI
        self.cli = OpenAI()
    def generate(self, prompts, max_tokens):
        from concurrent.futures import ThreadPoolExecutor
        def one(p):
            try:
                r = self.cli.chat.completions.create(
                    model=self.model, temperature=TEMPERATURE, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": SYS}, {"role": "user", "content": p}])
                return {"text": r.choices[0].message.content,
                        "in_tok": r.usage.prompt_tokens, "out_tok": r.usage.completion_tokens,
                        "stop_reason": r.choices[0].finish_reason, "error": None}
            except Exception as e:
                return {"text": None, "in_tok": 0, "out_tok": 0, "stop_reason": "error",
                        "error": repr(e)[:300]}
        with ThreadPoolExecutor(max_workers=8) as ex:
            return list(ex.map(one, prompts))

class GoogleBackend(Backend):
    def __init__(self, agent_id, model):
        super().__init__(agent_id, model, "google")
        from google import genai
        self.genai = genai
        self.cli = genai.Client()
    def generate(self, prompts, max_tokens):
        from concurrent.futures import ThreadPoolExecutor
        from google.genai import types
        cfg = types.GenerateContentConfig(system_instruction=SYS, temperature=TEMPERATURE,
                                          max_output_tokens=max_tokens)
        def one(p):
            try:
                r = self.cli.models.generate_content(model=self.model, contents=p, config=cfg)
                um = r.usage_metadata
                fr = r.candidates[0].finish_reason if r.candidates else None
                return {"text": r.text, "in_tok": um.prompt_token_count or 0,
                        "out_tok": (um.candidates_token_count or 0),
                        "stop_reason": str(fr), "error": None}
            except Exception as e:
                return {"text": None, "in_tok": 0, "out_tok": 0, "stop_reason": "error",
                        "error": repr(e)[:300]}
        with ThreadPoolExecutor(max_workers=8) as ex:
            return list(ex.map(one, prompts))

# ----------------------------------------------------------------------------- RUNNER
MAX_ATTEMPTS = 3   # bounded retries for infrastructure failures (refusal-truncation, API error)

def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out

def load_done(path):
    """Resumability. Every attempt is persisted; this returns the RESOLVED record per
    (phase,qid,agent) — the first attempt that yielded a parseable answer, else the last
    attempt once MAX_ATTEMPTS is exhausted (terminal infrastructure failure)."""
    by_key = {}
    for r in read_jsonl(path):
        by_key.setdefault((r["phase"], r["qid"], r["agent_id"]), []).append(r)
    done = {}
    for k, rs in by_key.items():
        rs.sort(key=lambda r: r.get("attempt", 0))
        ok = next((r for r in rs if r.get("parsed_answer") is not None), None)
        if ok is not None:
            done[k] = ok
        elif len(rs) >= MAX_ATTEMPTS:
            term = dict(rs[-1]); term["terminal_failure"] = True
            done[k] = term
    return done

def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())

def run_phase(questions, backends, phase, prompt_fn, max_tokens, out_path,
              run_id, batch=25, verbose=True):
    """prompt_fn(q, agent_id) -> str. Skips work already present in out_path."""
    done = load_done(out_path)
    results = dict(done)
    prior = {}
    for r in read_jsonl(out_path):
        k = (r["phase"], r["qid"], r["agent_id"])
        prior[k] = max(prior.get(k, 0), r.get("attempt", 0) + 1)
    for be in backends:
        for attempt_round in range(MAX_ATTEMPTS):
            todo = [q for q in questions
                    if (phase, q["qid"], be.agent_id) not in results
                    and prior.get((phase, q["qid"], be.agent_id), 0) <= attempt_round]
            if not todo:
                continue
            if verbose:
                print(f"[{phase}] {be.agent_id:8s} {be.model:32s} attempt {attempt_round+1} "
                      f"todo={len(todo)}", flush=True)
            for i in range(0, len(todo), batch):
                chunk = todo[i:i + batch]
                prompts = [prompt_fn(q, be.agent_id) for q in chunk]
                t0 = time.time()
                outs = be.generate(prompts, max_tokens)
                dt = (time.time() - t0) / max(len(chunk), 1)
                for q, o in zip(chunk, outs):
                    letters = set(q["choices"].keys()) if q["choices"] else None
                    ans, how = parse_answer(o["text"], q["type"], letters)
                    rec = {"run_id": run_id, "protocol": PROTOCOL_VERSION, "phase": phase,
                           "attempt": attempt_round,
                           "qid": q["qid"], "dataset": q["dataset"], "qtype": q["type"],
                           "agent_id": be.agent_id, "provider": be.provider, "model": be.model,
                           "temperature": TEMPERATURE, "max_tokens": max_tokens,
                           "gold": q["answer"], "raw_response": o["text"],
                           "parsed_answer": ans, "parse_mode": how,
                           "stop_reason": o.get("stop_reason"),
                           "correct": is_correct(ans, q["answer"], q["type"]),
                           "in_tok": o["in_tok"], "out_tok": o["out_tok"],
                           "latency_s_mean_batch": round(dt, 3), "error": o["error"],
                           "ts": datetime.datetime.now(datetime.timezone.utc).isoformat()}
                    append(out_path, rec)
                    if ans is not None:
                        results[(phase, q["qid"], be.agent_id)] = rec
                    elif attempt_round == MAX_ATTEMPTS - 1:
                        term = dict(rec); term["terminal_failure"] = True
                        results[(phase, q["qid"], be.agent_id)] = term
                if verbose:
                    print(f"    {min(i+batch,len(todo))}/{len(todo)}", flush=True)
    return results
