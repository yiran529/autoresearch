# GreedyLoRE workflow improvements implementation plan

> Execute inline with executing-plans; verify each part and commit it separately.

**Goal:** Preserve exploration freedom while improving evaluation provenance,
bounded timing, evidence handover and reuse of engineering knowledge.

**Architecture:** Keep trusted Attempts as the evaluation source of truth. Add
host-owned timing plans and a derived, version-bound per-node handover. Retain
existing Research/Engineering lifecycle and code inheritance.

**Tech stack:** Python, YAML/JSON, pytest, existing process-level workflow simulator.

**Spec:** User-approved design in this conversation, including U as optional advice.

## Constraints

- Work on the existing feature/greedylore-autoresearch branch.
- U is S*exp(-max(0, loss deltas)), relative to fixed stage-matched N000 evidence.
- U never gates exploration, Parent eligibility or candidate selection.
- New contracts use a 1.05 dense-relative quality gate; historical records stay intact.
- Host controls timing, normally 400 steps and at most 800, with cycle coverage.
- Researcher reads engineering findings before implementation; no automatic code merge.
- Prefer substantive new mechanisms over rank/width/gap tuning as node hypotheses.

## Part 1 — Evaluation policy and advisory utility

Files: grader results/contract/grader/evaluation_context, autoresearch evaluation,
taskdata/eval_contract.yaml, tests/test_utility.py and existing policy tests.

- [x] Add failing tests for the 5% boundary and advisory utility, including missing
  references, inherited quality, and independence from aggregate/selection.
- [x] Calculate U from trusted root Attempts, pin baseline identities and policy
  version; expose unavailable references instead of substituting Dense or Parent.
- [x] Persist components in ScoreBundle metadata/scores and display an advisory view.
- [x] Run focused policy/grader/selection tests and commit.

## Part 2 — Bounded host-owned timing

Files: grader timing policy/efficiency/grader/contract; submission validation and
template; seed timing telemetry plus source manifest; timing tests.

- [x] Add failing tests for 200/400/800 periods, cap overflow, authoritative step
  overrides, reference-window mismatch and actual cycle evidence.
- [x] Accept an optional committed timing declaration; host validates and derives
  the plan, keeping default legacy submissions readable.
- [x] Record requested/actual steps, period and coverage. Exclude unconfirmed
  timing from performance selection without discarding raw results.
- [x] Verify real command construction and synthetic payload validation; commit.

## Part 3 — Structured problem handover

Files: autoresearch nodes/state/analysis/validation/navigation plus new handover
module/schema; CLI; focused lifecycle and evidence tests.

- [x] Add failing tests for REFINE problem links, independent EXPLORE, frozen
  expectations, independent mechanism/practical verdicts and two follow-up limit.
- [x] Add optional backward-compatible research intent, required when parent has
  a new handover; freeze parent handover and engineering references in the Spec.
- [x] Publish host-derived handover from sealed Analysis and engineering refs;
  support retries, historical reports, safe paths and deterministic regeneration.
- [x] Verify lifecycle, legacy recovery, and evidence bindings; commit.

## Part 4 — Concise prompts and process simulation

Files: researcher/engineer/workflow roles, Analyzer prompt/schema, templates,
task README, simulation_support and workflow simulation assertions.

- [ ] Strengthen mechanism-first research instructions and engineering pre-read;
  specify REFINE/EXPLORE intent, verdicts, information gain and advisory-only U.
- [ ] Extend scripted agents to consume handovers and author linked hypotheses;
  exercise the real host pipeline with synthetic measurements.
- [ ] Run normal/fault simulations, prompt-delivery audit, task and framework
  regressions, lint, and independent code review; resolve material findings.
- [ ] Commit and report hashes, verification coverage and remaining limits.
