# AutoResearch V1 — Shared Workflow Rules

## Read and write boundaries

- Read project code and published evidence; write directly only in your own Worktree/scratch.
- Use task commands for shared records.
- Do not edit Tree, Spec, Attempts, evaluation logs or historical Artifacts; private evaluation data and credentials are out of scope.

## Research candidates and formal evaluation

- Each Research Node fixes one method hypothesis, one Parent/base_commit and its Spec; candidates may refine that method or meaningful parameters (for example refresh period or rank). A core mechanism or hypothesis change belongs in a child Node handled by a new Agent.
- Smoke tests and ordinary commits are allowed; later candidates may descend from earlier candidates in the same Node. Record each change and rationale in the existing `change_summary` and `selection_rationale` fields.
- Research allows at most 3 successful scored formal runs and 10 total formal runs, across resumes and archived Attempts. Valid scored regressions count as successful regardless of quality passing; failures and permitted same-SHA infrastructure retries count toward the total, and pending runs reserve capacity.
- After the first terminal result, Researcher remains active: within budget continue justified refinement, rather than stopping merely because one result exists. Stop evaluating at either cap; finish earlier with `autoresearch node complete --mode research` when no justified refinement remains and all runs have ended.
- Engineering optimizes the original Node without replacing its Research result.
- Complete `submission.yaml` using the task template before `coral eval`.
- Wait for evaluation and diagnostic results using 1800-second long waits (`yield_time_ms=1800000`) in `write_stdin` and any outer `functions.exec`/`functions.wait`.
- Continue as soon as results return; do not use shorter polling loops or send periodic waiting updates.
- Run and record the applicable checks in `semantic_criteria.md`; fix failed self-checks before submission.
- Engineering must preserve algorithm semantics.
- Do not alter the formal benchmark or include profiling/gradient instrumentation in formal timing.

## Retries and evidence

- Only infrastructure failures may use `autoresearch retry <sha>`: keep the same SHA and code, never create an empty commit to retry.
- Candidate/algorithm failures are evidence.
- Retain failed, regressed, incomparable and missing results honestly; CORAL's `improved` label does not prove quality passes.

## Node lifecycle

- Research freezes after explicit early completion or either formal-run cap, once all runs have ended. The host selects the quality-passing, comparable candidate with valid score and highest efficiency, with deterministic ties; if none qualifies it retains a representative failure and skips Engineering.
- Only after Research freezes does the host retire Researcher, refresh Evaluation and start independent Research Analysis over all Research Attempts and quality-gated Engineering at the selected fixed SHA.
- Root bootstrap remains one fixed baseline Attempt A0; historical completed Nodes are not reopened.
- The host binds one Engineer to each quality-passing Node, subject to the Engineer capacity; pending Engineering cannot be skipped for lack of capacity.
- Research Analysis and Engineering do not wait for each other.
- A Node (including Root N000) enables child expansion only after Research Analysis is published and Engineering is completed or explicitly skipped because quality failed or was unavailable.

## Engineering runs and completion

- Engineering has at most 3 successful formal runs and 10 total formal runs per assignment, across all resumes.
- A completed evaluation with a valid score counts as successful even when performance regresses.
- Failures and permitted same-SHA infrastructure retries count toward the total; pending submissions/retries reserve capacity.
- Stop evaluating when either limit is reached, even if infrastructure retries remain.
- Within the formal-run limits, Engineering continues until no clear, substantial semantics-preserving end-to-end improvement opportunity remains; one improved candidate is not sufficient to stop.
- After a terminal run and once the stopping condition or a formal-run limit is reached, the Engineer authors a final summary covering all claim Attempts and briefly explaining why it stopped, then calls `autoresearch engineering finish --analysis-file /absolute/scratch/final.json`.
- See the Engineer role and `references/engineer_runtime.md` for the JSON fields.
- The host seals and publishes the versioned Analysis before completing the claim.
- No independent Engineering Analyzer is launched.
- Once submitted, do not change the summary or submit another candidate; wait for publication and retirement.

## Evaluation and Analysis records

- Start at shared `notes/tree.md`, then open the assigned/selected Node's `README.md` index for Spec, Evaluation, latest Analyses, candidate SHAs and evidence. Read code at its fixed SHA with `git show <SHA>:<file>`; the current HEAD and historical `source_worktree` are not recovery anchors.
- For diagnostics, `autoresearch refresh-evaluation --node Nxxx` rebuilds Evaluation.
- Read versioned Analysis `.md` files for conclusions and limitations.
- New versions link a same-name `.json` containing the complete fixed evidence record; inspect it only when needed.
- Diagnostic IDs locate immutable records/results under the public diagnostics directory.
- Legacy Analysis `.md` files may still contain JSON.

## Supporting self-check files

- Before each formal Research evaluation (the host may auto-retire at the cap), or before Engineering finish, archive any uncommitted candidate self-check script/log that later Analysis must cite: `autoresearch evidence archive --node Nxxx --request-id self-check-1 --files /absolute/scratch/check.py /absolute/scratch/check.log`.
- This copies small regular files (unique basenames, 32 KiB total per bundle, at most 64 bundles per assignment) into immutable public diagnostics attachments. Use managed diagnostics for larger execution records. Keep the returned public-relative paths; do not cite `/tmp` paths in final Analysis.
- Attachments record the authenticated uploading Agent, an uploader-declared code-context SHA and content hashes. The host checks that the SHA exists, but does not certify that those bytes were executed. These are agent-provided evidence, not host-verified executions or formal scores. Document commands, inputs and limitations in the attached log.
- Research attachments must be archived before each formal evaluation while the assignment is active; Engineering attachments before finish. The host includes them in the role-appropriate frozen evidence manifest. Unneeded scratch need not be preserved.

## Subagents

- Subagents are optional for substantive uncertainty.
- Give each a bounded task and output; they inherit Sandbox, Evaluation and Artifact publication boundaries.
- Handle simple work yourself.

## Evaluation and research handover

- New contracts use a 5% dense-relative final-loss gate on both workloads; read the recorded contract for historical runs. U is optional advice, never a gate for exploration, Parent choice or candidate selection.
- Declare the effective timing cycle in committed `submission.yaml`. The host selects a complete-cycle window, normally 400 and at most 800 measured steps, and records actual event coverage. Unconfirmed cycles or unmatched reference windows cannot establish comparable performance.
- Before implementation, Researcher reads available Engineering Summary and parent Engineering Analysis, then records the applicability/reuse decision in the research intent.
- REFINE links a parent problem; EXPLORE may ask an independent question. Freeze mechanism/practical criteria and the response to parent evidence with `node initialize --intent-file`.
- Independent Research Analysis answers those criteria and supplies bounded incremental findings/follow-up questions. The host derives `handover.json`; its source reports remain authoritative, and child Specs retain the original snapshot.
