"""Deterministic construction of the frozen 300-question evaluation sample.
Run:  python src/build_sample.py
Sampling is fixed by SEED and performed BEFORE any model call. Re-running reproduces
data/samples/all_questions.jsonl byte-for-byte.
"""
import os, json, random, hashlib, collections
import urllib.request
import pyarrow.parquet as pq

SEED, N_PER = 20260914, 100
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, SAMP = os.path.join(HERE, "data/raw"), os.path.join(HERE, "data/samples")
os.makedirs(RAW, exist_ok=True); os.makedirs(SAMP, exist_ok=True)

SOURCES = {
 "gsm8k_test.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
 "arc_challenge_test.parquet": "https://huggingface.co/api/datasets/allenai/ai2_arc/parquet/ARC-Challenge/test/0.parquet",
 "mmlu_pro_test.parquet": "https://huggingface.co/api/datasets/TIGER-Lab/MMLU-Pro/parquet/default/test/0.parquet",
}
# Source for the difficulty SCREENS (Sec. pools), not part of the frozen 300-item
# sample; kept out of SOURCES so sample_manifest.json stays byte-identical.
SCREEN_SOURCES = {
 "gsmhardv2.jsonl": "https://raw.githubusercontent.com/reasoning-machines/pal/main/datasets/gsmhardv2.jsonl",
}
for name, url in {**SOURCES, **SCREEN_SOURCES}.items():
    dst = os.path.join(RAW, name)
    if not (os.path.exists(dst) and os.path.getsize(dst) > 1000):
        urllib.request.urlretrieve(url, dst)

LET = "ABCDEFGHIJ"
recs = {}

# GSM8K -------------------------------------------------------------- numeric
gsm = [json.loads(l) for l in open(os.path.join(RAW, "gsm8k_test.jsonl"))]
idx = random.Random(SEED).sample(range(len(gsm)), N_PER)
recs["gsm8k"] = [{"qid": f"gsm8k_test_{i}", "dataset": "gsm8k", "type": "numeric",
                  "question": gsm[i]["question"], "choices": None,
                  "answer": gsm[i]["answer"].split("####")[-1].strip().replace(",", "")}
                 for i in sorted(idx)]

# ARC-Challenge ------------------------------------------------------ mc
# NOTE: a minority of ARC items label options 1/2/3/4 rather than A/B/C/D. We normalise
# ALL items to letter labels (1->A, 2->B, ...) so the answer format is uniform across the
# benchmark. This is a presentation fix applied uniformly and before any model call; it
# changes no item's content, option order, or gold answer.
arc = pq.read_table(os.path.join(RAW, "arc_challenge_test.parquet")).to_pylist()
idx = random.Random(SEED).sample(range(len(arc)), N_PER)
out = []
for i in sorted(idx):
    r = arc[i]
    labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
    norm = {LET[j]: texts[j] for j in range(len(texts))}
    gold = LET[labels.index(str(r["answerKey"]))]
    out.append({"qid": f"arc_{r['id']}", "dataset": "arc_challenge", "type": "mc",
                "question": r["question"], "choices": norm, "answer": gold,
                "orig_labels": labels})
recs["arc_challenge"] = out

# MMLU-Pro ----------------------------------------------------------- mc
mp = pq.read_table(os.path.join(RAW, "mmlu_pro_test.parquet")).to_pylist()
idx = random.Random(SEED).sample(range(len(mp)), N_PER)
out = []
for i in sorted(idx):
    r = mp[i]
    opts = [o for o in r["options"] if str(o).strip() and str(o) != "N/A"]
    out.append({"qid": f"mmlupro_{r['question_id']}", "dataset": "mmlu_pro", "type": "mc",
                "question": r["question"], "choices": {LET[j]: opts[j] for j in range(len(opts))},
                "answer": str(r["answer"]), "category": str(r["category"])})
recs["mmlu_pro"] = out

# validate + write ----------------------------------------------------
allq = []
for k, v in recs.items():
    assert len(v) == N_PER, (k, len(v))
    for r in v:
        if r["choices"]:
            assert r["answer"] in r["choices"], (r["qid"], r["answer"])
            assert set(r["choices"]) <= set(LET), r["qid"]
    with open(os.path.join(SAMP, f"{k}.jsonl"), "w") as f:
        for r in v:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    allq += v

with open(os.path.join(SAMP, "all_questions.jsonl"), "w") as f:
    for r in allq:
        f.write(json.dumps(r, sort_keys=True) + "\n")

sha = hashlib.sha256(open(os.path.join(SAMP, "all_questions.jsonl"), "rb").read()).hexdigest()
json.dump({"seed": SEED, "n_per_dataset": N_PER, "total": len(allq), "sources": SOURCES,
           "sampling": "random.Random(SEED).sample(range(N_full), 100) per dataset; indices sorted",
           "arc_label_normalisation": "all option labels mapped to A..J; gold remapped accordingly",
           "qids": {k: [r["qid"] for r in v] for k, v in recs.items()},
           "sha256_all_questions": sha},
          open(os.path.join(SAMP, "sample_manifest.json"), "w"), indent=1)

print("total", len(allq), "sha256", sha)
print("mmlu_pro n_options:", dict(collections.Counter(len(r["choices"]) for r in recs["mmlu_pro"])))
print("arc relabelled items:", sum(1 for r in recs["arc_challenge"] if r["orig_labels"] != list(r["choices"])))
