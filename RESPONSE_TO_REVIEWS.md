# Response to reviews (camera-ready, NeSyDebates @ KSE 2026, paper 262)

Accepted 30 September 2026. Two reviews. Below is every weakness either reviewer
raised, what we changed, and where. Numbers quoted here come from
`analysis/results.json` and `analysis/exclusions.csv` via generated macros; none is
typed by hand.

Where a reviewer's reading of the paper was not what the code does, we say so and
treat it as a writing defect rather than a disagreement, because the submitted
wording permitted the reading.

## R2.1 It is unclear whether agents can actually run code during revision

**The most important comment, and partly a factual misunderstanding our wording
invited.** Execution did happen. `src/exec_sandbox.py` runs every program in a
separate interpreter process, with no network and no third-party libraries, under a
timeout, and reads the last value printed. This was applied to the agent's own
initial program and to both peer programs. The verbatim stdout was then inserted
into the revision prompt, labelled `Execution output of this program`.

Verified by reconstructing a stored revision prompt from the committed logs: the
agent saw its own program's output (`540`, correct) and both peer programs' outputs
(`180`, the fabricated wrong answer), so peer claims in the code medium were backed
by mechanical evidence rather than confidence.

The reviewer is right about one thing, and the paper now states it as a scope limit:
the agent does not call an interpreter itself. It reads outputs produced for it. The
medium supplies verified evidence, not interactive verification.

- Added Section III-C, "What the agent sees in the executable medium".
- Abstract no longer says "a program the agent can run". It now says the program is
  shown with its real execution output and that every program was run in a sandbox.
- Manipulation check rewritten into its two component claims, and now states what it
  does not establish: that agents used the outputs.
- New limitation, "Verified output, not interactive verification".

## R1.3 The prose-versus-code manipulation changes more than checkability

Accepted. The limitation existed in the submission and is now sharper. It names the
two accompanying differences (reasoning process, response format), gives the three
partial defences, and concedes that none is complete. One of them is now empirical
rather than argumentative: initial accuracy was identical across media in this run,
so differential starting difficulty is not an available explanation here. We name the
experiment that would isolate checkability, holding format fixed and varying only
whether execution output is shown, and say we did not run it.

## R2.2 187 cells excluded against 112 observed

The two totals have different denominators, which the submission never said. The 86
removed-agent rows are per pair over the pre-registered three-agent design. The 101
revision rows are per pair-arm and cover 60 distinct pairs. Those 60 plus the 112
observed are exactly the 172 cells of the two-agent design. Now stated in Results,
with `ExclRevisionPairs` and `CellsDesignRetained` emitted as macros so the identity
is machine-checked rather than asserted.

## R1.1 and R2.2 The study is underpowered

Accepted, and it was already the paper's own position. 43 items passed validation and
28 completed all six phases against a pre-registered target of about 100. The design
needs 6 discordant flips for conventional significance and observed 2. We have not
softened this. The paper continues to call the null inconclusive rather than evidence
of absence, and the interval is described as bounding rather than excluding an effect.

## R2.3 The contribution reduces to "the instrument was underpowered on a saturated pool", and the saturation lesson is predictable

Partly accepted, and the Conclusion now concedes it directly: the general warning is
not surprising, and anyone designing a conformity study would accept in the abstract
that an easy benchmark is a poor place to look. We name the three things that are
harder to obtain without running it. A sensitivity number (6 discordant flips) that a
future run can check against its expected event rate before spending inference. The
measured failure of the obvious remedy, since GSM-Hard screened at 100.0% and 90.0%,
so difficulty must be measured on the roster in use rather than inherited from a
dataset's reputation. The released instrument with validated stimuli. The paper
explicitly does not claim the lesson is novel.

## R2.3 The beneficial direction and H3 are untested

Correct, and unchanged from the submission, which already stated both. The beneficial
arm has n=0 by construction at 100.0% initial accuracy. H3 is reported as untested,
not null. We have kept that wording deliberately.

## R2.4 The title overclaims

Accepted. The accepted title is kept verbatim as the first clause, with a subtitle
added: "A Pre-Registered Null and a Benchmark-Saturation Obstacle". A question-form
title alone implies an answer this sample cannot give.

## R1.2 External validity: single model family, one dataset, one revision round

Accepted, unchanged, and already stated as the study's most serious external-validity
threat. All three appear in Limitations, and the first two are the first two items of
the future-work plan.

## Not changed

Nothing was removed. No number moved. The reviewers asked for no new experiments, and
we ran none, so every quantity in the camera-ready is the same quantity as in the
submitted version, recomputed by `reproduce.sh` from the same committed logs.

The paper grew from 5 pages to 6, within the workshop's 8-page limit and within the
host conference's 6-page allowance before per-page charges.
