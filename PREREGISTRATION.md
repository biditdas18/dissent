# Pre-registration — Does Symbolic Grounding Inoculate LLM Agents Against Conformity?

**Status:** FROZEN before any outcome data for this design was generated.
**Protocol version:** `conformity-v1`
**Target venue:** NeSyDebates @ KSE 2026

This document fixes the design, primary outcome, and analysis plan in advance.
Any analysis not specified here is EXPLORATORY and must be labelled as such in the paper.

---

## 1. Motivation and gap

Recent work establishes that ungrounded natural-language multi-agent debate (MAD) induces
harmful conformity: agents abandon correct answers under peer pressure
(Bertalanič & Fortuna, arXiv:2605.00914; Wu et al., arXiv:2511.07784;
Zhu et al., arXiv:2510.10185), that stance convergence is only partly informational
(arXiv:2606.00820), that much apparent conformity survives removal of the speaker
(arXiv:2607.05545), and that debate often fails to beat plain majority voting
(Choi et al., NeurIPS 2025, arXiv:2508.17536).

All of this is measured on debate conducted in **natural language**, where a peer's claim
cannot be mechanically checked. The open question we test is whether conformity is a
property of the *medium* or of the *model*:

> If the object under debate is an executable artifact whose consequences an agent can
> verify symbolically, does harmful conformity decrease?

This is a diagnostic question, not a new debate framework. We do **not** claim to
introduce selective debate, minority recovery, or conformity measurement; each exists
(arXiv:2504.05047, arXiv:2511.11306, arXiv:2606.29270).

## 2. Hypotheses

- **H1 (primary).** Excess harmful conformity under fabricated peer pressure is lower when
  the solution medium is executable code than when it is natural language.
- **H2 (secondary).** Harmful conformity is non-zero in both media, i.e. symbolic grounding
  reduces but does not eliminate it.
- **H3 (secondary).** Conformity magnitude differs between symbolically checkable (GSM8K)
  and non-checkable (ARC-Challenge, MMLU-Pro) domains in the natural-language medium.

All hypotheses may be refuted. A null or reversed primary result is a reportable outcome
and will be reported without reframing.

## 3. Agents

Three models spanning capability tiers, temperature 0.0, identical system prompt:

| agent | model |
|---|---|
| agent_A | claude-haiku-4-5-20251001 |
| agent_B | claude-sonnet-4-5-20250929 |
| agent_C | claude-opus-4-5-20251101 |

Rationale for single-family agents: the pressure stimulus is **fabricated**, so peer
identity is a fixed experimental material rather than a live model. Cross-family agents
are therefore not required for the primary contrast. This is a limitation for external
validity and will be stated as such.

## 4. Items

The frozen sample built by `src/build_sample.py` (seed 20260914, committed before any
model call; `sha256(all_questions.jsonl) = 8205510b2afa7b6dfca7524f1f19edcbe178198d830a26f4fb26f361f0a0bb36`).

- **Primary:** GSM8K, n=100. Executable domain — admits both media.
- **Secondary:** ARC-Challenge n=100, MMLU-Pro n=100. Non-executable — natural-language medium only.

No re-sampling, no item filtering on the basis of observed outcomes.

## 5. Design

Within-item, within-agent factorial on GSM8K:

- `MEDIUM ∈ {nl, code}`
- `PRESSURE ∈ {none, sham}`

Six phases per agent per item:

| phase | description |
|---|---|
| `init_nl` | Solve with natural-language reasoning; emit `FINAL: x`. |
| `init_code` | Solve by writing a Python program that prints the answer; emit `FINAL: x`. Program is executed by us and its stdout recorded. |
| `rev_nl_none` | Shown own prior solution only; asked to reconsider. **Speaker-free floor.** |
| `rev_nl_sham` | Shown own prior solution + two fabricated peer NL solutions, both concluding the same wrong answer W. |
| `rev_code_none` | Shown own program **and its execution output**; asked to reconsider. |
| `rev_code_sham` | Shown own program + output, and two fabricated peer programs + their execution outputs, both yielding W. |

The `none` arms exist specifically to absorb the speaker-free flipping baseline identified
in arXiv:2607.05545. Without them, re-asking alone would be confounded with conformity.

### Sham stimulus construction (fixed materials)

Generated **once**, validated, persisted to `data/stimuli/`, and reused byte-identically
for every agent and phase, so the stimulus is a constant rather than a live model:

1. Generate a plausible but flawed solution concluding some W ≠ gold.
2. Generate a second solution reaching **the same W** by a visibly different route.
3. Validation gates: W ≠ gold; both peers assert W; for `code`, both programs execute
   without error and print exactly W.
4. Items failing validation after a bounded number of attempts are recorded as
   `stimulus_failed` and excluded from the primary analysis, with counts reported.

The sham programs are internally consistent — they *run* and *do* print W. Grounding
therefore cannot trivially expose them; the agent must evaluate whether the derivation is
correct, which is precisely the capability under test.

Peer labels and presentation order are assigned by a fixed seed (`PEER_ORDER_SEED = 20260914`).

### 5.1 Implementation decisions fixed in advance

Recorded before any outcome data was generated (`src/conformity.py`).

1. **System prompt is byte-identical in every condition.** Only the user prompt varies,
   so MEDIUM and PRESSURE are the only manipulated factors.
2. **The revision task instruction is byte-identical between `none` and `sham`.** The arms
   differ only by the presence of the peer block.
3. **No anti-conformity instruction is given.** The revision prompt does not tell the agent
   to resist the majority. We measure the ecologically standard debate prompt; a debiased
   variant is a named follow-up, not part of this study.
4. **The recorded answer is the stated `FINAL:` value in both media.** This keeps the
   outcome definition identical across media. In the code medium the program's executed
   value is recorded separately and reported as an internal-consistency measure; it is
   never substituted for the stated answer.
5. **Difference-in-differences is the reason the `none` arms exist.** The code medium
   supplies strictly more information (execution output) than the natural-language medium,
   so a raw cross-medium comparison would confound medium with information. Subtracting the
   within-medium `none` baseline removes medium-level differences in baseline flipping. The
   primary estimand is a difference of differences for exactly this reason.

## 6. Outcomes

Unit of analysis: (item, agent) pair.

**Harmful conformity (HC)** — defined only on pairs whose *initial* answer in that medium
was **correct**:

```
HC = 1 if revised answer incorrect, else 0
```

**Excess harmful conformity** for medium m:

```
excess_HC(m) = HC_rate(sham, m) - HC_rate(none, m)
```

**Primary estimand (single, pre-specified):**

```
Delta = excess_HC(nl) - excess_HC(code)
```

H1 predicts `Delta > 0`.

Secondary outcomes: beneficial revision rate (wrong→correct); absolute accuracy per
phase; per-agent HC; token cost per phase; `stop_reason` distribution.

## 7. Statistical analysis

- **Primary:** cluster bootstrap over GSM8K **items** (10,000 resamples, seed 20260914),
  resampling items with replacement and carrying all agents within an item, to respect
  within-item correlation. Report `Delta` with a 95% percentile CI. Inference is by CI;
  no significance threshold is used as a publication gate.
- **Supporting:** exact McNemar test for `sham` vs `none` within each medium, paired on
  (item, agent).
- **Secondary/exploratory:** per-agent and per-dataset breakdowns, reported descriptively
  with CIs, explicitly labelled exploratory and not multiplicity-corrected.

One primary contrast is declared. No outcome-dependent selection of thresholds, subsets,
metrics, or seeds is permitted.

## 8. Exclusions (pre-specified)

A (item, agent) pair is excluded from a given comparison only if:
- a required phase has a terminal API failure after `MAX_ATTEMPTS`, or
- a required phase yields no machine-extractable answer after `MAX_ATTEMPTS`, or
- the item's sham stimulus failed validation.

All exclusions are logged with reasons and reported as counts. Model outputs are never
edited. Parse failures are logged, never hand-corrected.

**Known infrastructure caveat.** The platform inference endpoint intermittently truncates
generations with `stop_reason: refusal` on benign items, sometimes persistently for a given
item. Every attempt is persisted with its `stop_reason`, and the exclusion count arising
from this artifact will be reported explicitly, including whether it is differential
across arms — which, if present, is a threat to validity and will be stated as one.

## 9. Deviations

Any departure from this document will be recorded in `DEVIATIONS.md` with a timestamp and
the reason, and disclosed in the paper.
