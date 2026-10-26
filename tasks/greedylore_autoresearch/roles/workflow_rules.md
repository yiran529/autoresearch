# AutoResearch — Shared Workflow Rules

## Read and write boundaries

- Read project code and published evidence; write directly only in your own Worktree/scratch.
- Use task commands for shared records.
- Do not edit Tree, Spec, Attempts, evaluation logs or historical Artifacts; private evaluation data and credentials are out of scope.

## Research development and evaluation
- Research uses the host-enabled staged protocol. Before choosing a method, acquire `autoresearch node planning acquire --wait`; under the token read the current registry, choose one completed code Parent and judge overlap. Register through `autoresearch research register` with `origin`, first `design` and planning controls. Implement after successful registration releases the lock; renew before expiry, reacquire/reread after loss. Existing registered Nodes resume their saved stage without selecting again.
- `origin` fixes Parent/base_commit, question, core hypothesis, evidence/selection rationale and response to parent/Engineering evidence. The host generates Spec and compatibility handover fields; do not author a separate research intent. Save each design before training; append revisions without overwriting earlier designs, predictions, observations or decisions. Core mechanism/hypothesis changes require a new Node; normal actions of a fixed adaptive policy do not.
- Normally use two, at most three 130M candidate trajectories. Failed/stopped/restarted trajectories count, pending reserves capacity; exact same-SHA continuation stays in its original trajectory and restores complete state. Fixed steps/tokens/schedule and all budgets persist across pauses/restarts. Only host-classified infrastructure failures permit limited same-version recovery. Never bypass the staged entrypoint through `coral eval`, old initialization/completion or independent GPU work.
- Researcher observes and diagnoses within its own context. Use managed `research diagnostic` only for a paused/completed trajectory and a decision-relevant uncertainty, within cumulative 5-success/15-total execution limits, at most 600 seconds each. Diagnostic branches must not contaminate candidate state or supply free training. The host holds four GPUs through development and 60M finalization; work borrows them serially.
- Use managed `research timing` when efficiency evidence can change a decision, before training, while paused or after completion. Timing pins the design's SHA and uses the existing fixed protocol; a changed method needs an appended design within the three-design budget. Reuse matching results by default; intentional repeats require `repeat_reason`. The latest completed matching result is used, not the fastest. Its separate host budget defaults to 5 completed/15 total executions, including repeats/retries, across resumes; diagnostics retain their own limits.
- Prepare and commit `submission.yaml` with the existing fields and run applicable `semantic_criteria.md` checks before training; fix self-check failures. Declare the full effective `timing.period_steps`, or null for nonperiodic methods (`requested_steps` alone is optional), and actual maintenance-event telemetry. No profiling/gradient instrumentation in formal timing; missing rare events or unmatched windows cannot establish long-run comparable efficiency. U is advisory, never a selection/quality gate.
- Record interpretation/evidence and `continue/revise/stop` through `research decision`; the host appends observations. A decision does not execute the next action. Use `research finish` once all work is terminal, nominating one completed 130M iteration (or null); see `references/research_commands.md` and `templates/research_origin.json` for JSON/CLI examples.
- Training completion does not automatically run timing. Before freezing, the host reuses matching timing or supplies missing timing within the same budget, then checks eligibility. Failed timing or exhausted capacity starts no transfer; inspect evidence before revising or finishing with null.
- The host freezes the entire record and selected SHA/policy before at most one 60M transfer, only after complete eligible 130M evidence. No candidate switches, tuning or reopening from 60M feedback; only bounded infrastructure recovery. No eligible candidate means explicit skip with failure evidence. Reused 60M feedback is transfer evidence, not independent campaign held-out.
- Wait for managed evaluation/diagnostic results using 1800-second long waits (`yield_time_ms=1800000`) in `write_stdin` and outer `functions.exec`/`functions.wait`. Continue on return; no short polling, periodic log reads or waiting updates.

## Retries and evidence
- Transport timeouts reuse the original request ID and payload; new executions use new IDs. Inspect failures, fix diagnostic causes and preserve failed records; do not blindly repeat or create empty commits for retries.
- Keep failed, regressed, incomparable and unavailable evidence. `improved` alone does not establish quality passing; local mechanism success does not establish training/efficiency benefit. Do not rewrite predictions after results.

## Node lifecycle
- The host admits at most two unfinished Nodes, retaining the slot through Research, published Research Analysis and completed/legitimately skipped Engineering. Waiting Nodes resume existing code/session/budgets before new Nodes are created.
- After frozen 60M terminates or is explicitly skipped, the host releases the Research reservation, retires Researcher, refreshes Evaluation and starts one independent Research Analysis over all attempts and quality-gated Engineering at the selected fixed SHA. Researcher writes no separate final report.
- Root bootstrap remains one fixed baseline Attempt A0; completed historical Nodes are not reopened. Research Analysis and Engineering proceed independently. Engineering pending for capacity cannot be skipped for that reason, and preserves algorithm semantics without replacing the Research result.
- Child expansion requires published Research Analysis and completed Engineering, or explicit Engineering skip because quality failed or was unavailable. Complete Node termination triggers host cleanup of large dynamic artifacts, retaining small evidence and reports.

## Engineering runs and completion

- Engineering reads the claim's design source: the frozen selected iteration in `research.json` and matching Research SHA, or Root's fixed baseline Spec/A0. Required handover must be present and consistent. Preserve adaptive policy decisions, triggers and state semantics.
- Commit candidates and measure only through `autoresearch engineering timing --sha <full-HEAD-SHA> --request-id <id>`. The command waits and uses the existing timing contract. Same-SHA infrastructure recovery adds `--retry` with a new ID; a lost reply reuses the original ID/SHA/flags. Direct formal submission cannot bypass this entrypoint.
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

- Before managed Research experiments, or before Engineering finish, archive any uncommitted candidate self-check script/log that later Analysis must cite: `autoresearch evidence archive --node Nxxx --request-id self-check-1 --files /absolute/scratch/check.py /absolute/scratch/check.log`.
- This copies small regular files (unique basenames, 32 KiB total per bundle, at most 64 bundles per assignment) into immutable public diagnostics attachments. Use managed diagnostics for larger execution records. Keep the returned public-relative paths; do not cite `/tmp` paths in final Analysis.
- Attachments record the authenticated uploading Agent, an uploader-declared code-context SHA and content hashes. The host checks that the SHA exists, but does not certify that those bytes were executed. These are agent-provided evidence, not host-verified executions or formal scores. Document commands, inputs and limitations in the attached log.
- Research attachments must be archived before managed experiments while the assignment is active; Engineering attachments before finish. The host includes them in the role-appropriate frozen evidence manifest. Unneeded scratch need not be preserved.

## Subagents

- Subagents are optional for substantive uncertainty.
- Give each a bounded task and output; they inherit Sandbox, Evaluation and Artifact publication boundaries.
- Handle simple work yourself.

## Evaluation and research handover

- New contracts use a 5% dense-relative final-loss gate on both workloads; read the recorded contract for historical runs. U is optional advice, never a gate for exploration, Parent choice or candidate selection.
- New methods declare `timing.period_steps` in committed `submission.yaml`: a full fixed cycle, or null if none. Only `requested_steps` is optional; legacy submissions may omit `timing`. The host measures complete fixed cycles or a default window for nonperiodic methods, normally 400 and at most 800 steps. Invalid event evidence or unmatched reference windows cannot establish comparable performance.
- Before implementation, Researcher reads available Engineering Summary and parent Engineering Analysis, then records applicability/reuse decisions or report-unavailability reasons in `origin.engineering_review`.
- A starting point may be an observation/problem, a desired property under a resource constraint, or a mechanism hypothesis; it need not come from handover or establish an existing defect. REFINE builds on the Parent method/mechanism; EXPLORE uses a substantively different mechanism. Cite an addressed parent question, otherwise use null and explain the connection in parent_response. Register origin and initial design once; the host derives compatibility criteria for existing Spec/handover consumers. The immutable first design and appended iteration designs supply the actual predictions and tests.
- Independent Research Analysis answers those criteria and supplies bounded incremental findings/follow-up questions. The host derives `handover.json`; its source reports remain authoritative, and child Specs retain the original snapshot.
