# Gap-driven GreedyLoRE AutoResearch

Task resources are host-published under the run's public `notes/toolkit/`. Each Agent's generated instructions locate that directory under its runtime shared directory (for Codex, `.codex/notes/toolkit/`). It contains workflow rules and route-specific references. It does not contain private evaluation inputs. Research Analysis receives a fixed code/evidence snapshot and only the parent analyses actually referenced by its Spec.

Research Analysis remains an independent session. Choose applicable initial diagnostics, investigate a principal unexplained observation, and report evidence, counterevidence and unknowns without proposing algorithms. Each Analysis has at most three successful diagnostic executions and ten total executions including failures, each up to 600 seconds, with a separate GPU queue budget of one hour per request. Failed diagnostics may be debugged and retried with new request IDs; all retries retain records and count toward the total budget. See `GPU_COORDINATION.md` for host/client wait budgets.

## Node workflow

After its unique formal Research Attempt is terminal with no remaining infrastructure retry, the host retires the Researcher and starts two independent branches: Research Analysis, and Engineering when the authoritative quality gate passes. The host binds each Engineer to a fixed Node and Research SHA; `autoresearch engineering claim Nxxx` confirms that assignment and prepares its Worktree, rather than selecting a target. Required Engineering waits in FIFO order when the configured Engineer capacity is occupied. Researcher and Engineer capacities remain separate.

A Node is an eligible Parent only after Research Analysis is published and Engineering is completed, or explicitly skipped because quality failed or was unavailable in a terminal failure. A malformed/missing quality score on an otherwise normal result is an error, not a skip. Researcher retirement alone does not complete the Node. Children still start from the Parent's Research SHA, not its Engineering commit.

Root N000 follows the same two branches after the host pins S0, creates A0 and submits the fixed baseline Attempt. Dynamic scheduling can start before Root Analysis exists, but does not reserve ordinary Researchers until N000 is fully complete. Root Engineering uses A0 as its quality anchor and ordinary Engineer authorization; the host-only bootstrap exception remains restricted to Root Research. The standalone `bootstrap-root-analysis` command produces baseline evidence only and does not certify full Node completion or automatically start Engineering.

The existing Analysis worker handles only Research Analysis. Short host reconciliation publishes sealed Engineering summaries without waiting for that worker. Analysis retries preserve their fixed evidence/version and diagnostic budget, apply bounded backoff, and do not block later eligible jobs. A validated private output or published immutable Analysis is reused after interruption. Workflow/model process identities are recorded for stop and restart cleanup; Analyzer sessions have a default overall limit of 18000 seconds, including diagnostic queue waits.

## Engineering completion

A claim permits at most two distinct formal Engineering candidates. Same-SHA infrastructure retries are counted separately under the existing retry policy. The second candidate does not retire the Engineer: it must still author and submit the final summary.

Write JSON in authorized scratch outside the tracked Worktree, containing exactly:

- `summary`: non-empty text.
- `findings`: a non-empty array of non-empty strings covering measured results, each candidate, semantic checks and supported attribution.
- `limitations`: a non-empty array of non-empty strings, with uncertainty, missing evidence and check coverage.
- `attempt_shas`: all distinct formal candidate SHAs in this claim, exactly once, including failures and regressions.

Run `autoresearch engineering finish --analysis-file /absolute/scratch/final.json`. The host authenticates ownership, checks terminal Attempts and complete SHA coverage, seals the evidence and persists the submission before returning `finalizing` with the intended Analysis ref. This is not confirmation of completed publication. The host then publishes an immutable version and completes the claim. On interruption, reuse the identical body; conflicting submissions are rejected. No independent Engineering Analyzer is launched. Quality remains inherited (provisional) from the fixed Research SHA.

Historical Analyses remain readable. Node-bound scheduling uses workflow state version 2; version 1 runs are rejected without modification and must continue with their original runtime. No automatic migration or reopening of historical claims is performed. Saved configuration and role snapshots must match the runtime before resuming a version 2 run. Complete dynamic runs require at least one Engineer slot and exactly one Engineering assignment per quality-passing Node; standalone baseline-only tools retain their original behavior.

## Validation

Scripted workflow simulations cover normal operation and injected failures, including same-SHA retry, automatic Node binding, foreign-claim rejection, publication and parallel Research. They use synthetic results, with no model, training or GPU calls. Scientific effectiveness requires a separate real low-rank pilot with matching budgets and legitimate saved training states; passing workflow tests does not establish scientific improvement.


## Memory

The existing Research-driven Memory trigger and two Markdown Summary strings are retained. Summaries distinguish new mechanism evidence, replication, refutation/scope restrictions and open questions, with exact Analysis versions. Research Summary covers algorithm/state/update evidence; Engineering Summary covers measured performance and Engineer-authored attribution. Engineering Analysis is an input, not an independent trigger.

For bounded verification, Memory can read public diagnostic records matching a published Node/mode/version/Research SHA, plus text results. Staging excludes symlinks, scripts, code/state snapshots, binary tensors and unrelated identities, with limits of 1 MB per file, 256 files and 8 MB total. Omitted files must be treated as unavailable, not as negative evidence.


## Host-assigned research modes

The host draws a mode once when reserving a Research assignment, using trusted `grader.args.p_explore` (initially `0.20`). It persists the result in private workflow state, sends it through the authenticated roster and generated Agent instructions, and records it in new Node Specs. Resumes, same-SHA retries and duplicate reconcile/initialize requests retain the original mode even if the configured probability changes. Pre-upgrade assignments missing a mode use `REFINE`; existing immutable Specs are not rewritten.

`REFINE` starts from supported gaps or important unresolved mechanisms in published analyses. `EXPLORE` prioritizes another communication mechanism or a falsifiable cross-mechanism interaction that the current evaluation contract can fairly evaluate. It does not relax the contract or quality gate. Researcher cannot set the mode in `node initialize`, draw it locally, or use multiple code Parents. The host-assigned mode is part of assignment context, not an additional Research initialization field.


## Prompt delivery audit

`tests/test_workflow_simulation.py` runs normal and fault/recovery scenarios through the actual `CodexRuntime` launcher and the actual one-shot Analyzer/Memory launcher. A test-only `codex` executable records argv/stdin, then runs the scripted agents or supplies synthetic JSON. The host's random draws are fixed in the test handler to exercise both REFINE and EXPLORE. No live model or GPU workload is started.

Every run writes `prompt_audit/*.json` alongside its `report.json`. Producer captures include their startup/resume text, generated `AGENTS.md`, exact role content and task resource snapshots. Analyzer/Memory captures include their final stdin, supplied Schema, staged evidence paths and identities. The tests verify mode pinning, resume arguments and instruction-file continuity, role/reference delivery, full code-snapshot hashes against the fixed Git commit, staged evidence hashes against the manifest (including parent Analysis), Memory Analysis hashes against published sources and trigger coverage and absence of independent Engineering Analyzer calls. `prompt_audit/CHECKS.md` is written only after the delivery assertions pass.

The diagnostic-enabled Analyzer has an additional final-input test at the namespace/process boundary, checking the 3 × 600-second execution limit, separate 3600-second queue wait, fixed roots, writable scratch and JSON Schema. That test substitutes namespace/process execution, so it does not establish OS isolation or GPU execution.

These checks establish application input delivery. Live Codex instruction loading, conversation persistence, model compliance and scientific effectiveness require separate live acceptance tests.
