# Gap-driven AutoResearch Implementation Plan

**Goal:** Apply the approved V5.4 proposal to the local CORAL task, with 600-second diagnostics, long GPU waits, Engineer-authored final analyses and persistent research modes.

**Architecture:** Keep scientific behavior in task prompts/references. Keep assignment identity, attempt limits, evidence envelopes and publication in trusted host code. Reuse current diagnostic isolation, immutable Analysis format and Memory gate.

**Spec:** `/home/wyr/gap_driven_autoresearch/en/PROPOSAL.md`, with the user's approved amendments in this session.

**Tech stack:** Python, pytest, existing CORAL task/runtime infrastructure; no additional dependencies.

## Constraints

- Research has one distinct formal candidate; Engineering has at most two per claim. Same-SHA infrastructure retries are not new candidates.
- Diagnostics have at most three executions per Analysis and 600 seconds per execution. GPU queue wait is separate, up to 3600 seconds.
- Diagnostic worker/client budgets are 4500/4800 seconds. General trusted clients wait at least an hour; formal eval waits also cover configured GPU queues. Existing 86400-second formal GPU waits remain available.
- Research Analysis remains an independent session. Engineer authors its final Analysis; the host seals and publishes it before completing the claim.
- Shared artifacts are host-owned and immutable by version. Old analyses remain readable.
- REFINE/EXPLORE is selected once by the host, persisted and restored; missing old values mean REFINE.
- Every part is verified and committed separately. No GPU training or live model calls are required for workflow tests; pilot scientific effectiveness is reported separately.

## Part 1 — Prompts, references and wait budgets

- [x] Add bounded tests for 600-second diagnostics, GPU queue arguments and long eval polling using simulated time; observe failures.
- [x] Reuse the existing diagnostic runner; update its queue, execution, worker and client limits together.
- [x] Share default formal-eval waiting calculation between submit and wait; account for two GPU acquisition stages. Increase silent-agent defaults and task pending exemption.
- [x] Install Researcher/Analyzer prompts and route references; provide runtime read access through the host-published public notes/toolkit. Preserve existing Engineer completion until Part 2.
- [x] Run diagnostic, analysis, hook, dynamic/runtime mounting and config regressions; lint changed Python; commit.

## Part 2 — Engineer-authored completion

- [x] Test authenticated final-summary submission, all-attempt coverage, two-candidate enforcement, retry/recovery and stale/foreign evidence rejection; observe failures.
- [x] Extend `engineering finish --analysis-file` to send structured text and SHA coverage, not a host file path. Validate ownership and terminal attempts under the host lock.
- [x] Persist the sealed author submission before finalizing; publish idempotently and complete only after verified publication. Never invoke Engineering Analyzer in the finalizer.
- [x] Preserve historical Analysis readability and recover interrupted publication. Keep Engineer alive after its second eval until it submits the summary.
- [x] Update Engineer instructions, runtime reference, shared rules and workflow simulations; run focused plus end-to-end workflow tests; commit.

## Part 3 — Memory

- [x] Update evidence classification and role-specific Summary rules. Preserve the current trigger and JSON/frontmatter contracts.
- [x] Where required, stage only bounded public diagnostic records/results, excluding code snapshots, secrets and unrelated runtime files; test staging boundaries.
- [x] Run Memory and workflow regressions; commit.

## Part 4 — Persistent research modes

- [ ] Test deterministic mode allocation, restoration, legacy REFINE defaults and rejection of model-supplied mode fields; observe failures.
- [ ] Persist the sampled mode when reserving a Node, transmit it through trusted assignment decoding and generated instructions, and record it in the host-created Spec.
- [ ] Read `p_explore` from trusted task config (initial 0.20), never from Agent initialization fields. Preserve a single code Parent and evaluation contract.
- [ ] Run task workflow, dynamic/template and full framework regressions; update delivery notes; commit.

## Validation limits

Workflow simulation uses scripted agents and synthetic results. No scientific improvement, training-quality gain or hardware speedup is claimed without a separate real pilot using permitted data/state and matched diagnostic budgets.

Part 1 validation: 242 passed, 1 namespace-availability skip; core/task changed Python lint clean. Scientific pilot remains unrun: no real training states or model/GPU runs were used.

Part 2 validation: 300 passed, 1 namespace-availability skip, including normal/fault scripted-agent simulations. No Engineering Analyzer is launched by the finalizer. Historical analyses remain unchanged.

Part 3 validation: Memory plus workflow-state regression tests passed (50 total). Diagnostic staging tests cover matching identities, oversized files, binary files, symlinks and foreign SHA exclusion.
