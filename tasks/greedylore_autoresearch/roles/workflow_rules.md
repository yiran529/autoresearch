# AutoResearch Shared Workflow Rules

## 1. Purpose and authoritative sources

This file defines the execution protocol shared by Researcher and Engineer. Role documents state research or engineering objectives, command references define request interfaces, and the host records current state, budgets, and evaluation contracts. Each role uses these to choose actions within its assigned problem.

- Budgets below are defaults for the current protocol. Actual runs follow their saved contracts, host state, and claim `run_limits`; new documentation does not relabel historical results.
- Current staged Research uses `autoresearch research`. Do not bypass the protocol with `coral eval`, legacy initialization/finish entry points, or independent GPU work. Report configuration mismatches.

## 2. Write, resource, and evidence boundaries

Develop code and requests in your own Worktree or authorized scratch; publish shared records through task commands. The host manages formal training, timing, and GPU diagnostics; it holds the four-GPU Research reservation through development and all Final components, and tasks borrow it serially.

- Do not directly edit Tree, Spec, Attempts, formal logs, historical Artifacts, or shared conclusions. Private evaluation material and credentials are outside your authorization.
- Use host-provided artifact/checkpoint paths. Store large tensors under dynamic-storage rules; do not copy them to the root partition or arbitrary locations, or change retention yourself. Report space or recovery failures to the host.
- Do not add profiling or gradient instrumentation to formal timing. Complete cycles, key events, measurement windows, and hardware comparability determine the scope of cost evidence.
- Optional subagents may handle substantive, bounded tasks and inherit sandbox, evaluation, and publication permissions. Handle simple work yourself.

## 3. Research registration and design

Before choosing a topic, acquire a planning lease with `autoresearch node planning acquire --wait`. Use the latest registry, ongoing designs, and evidence to select an eligible code Parent and assess research overlap.

Before implementation, save `origin`, the initial `design`, and planning controls with `autoresearch research register`. The host generates the Spec and compatible handover fields; Researcher does not write a separate research intent. Record designs and judgments in append-only records to preserve original predictions and their changes.

- Renew the lease before expiry; release it when abandoning a topic. Successful registration releases the lock; failed registration retains it. After losing the lock, reacquire it and reread the registry. See `references/research_commands.md` for renew/release commands.
- `origin` fixes Parent/base_commit, the main question, core hypothesis, selection rationale, and responses to Parent/Engineering. Changing the core mechanism or hypothesis requires a new Node.
- Save designs before training; append revisions rather than overwriting them. Normal actions of a declared adaptive policy are not revisions.
- Use the host-assigned REFINE/EXPLORE mode. Preserve it across recovery and retries; do not resample.
- Research may start from an observation/question, a desired property under resource constraints, or a mechanism hypothesis. Cite a corresponding `problem_ref` only when responding to a real Parent question; otherwise use null and explain the connection.
- Before implementation, review available Engineering Summary and Parent Engineering Analysis. Record reuse/deferral decisions, confounding, or reasons material is unavailable in `origin.engineering_review`.

## 4. Research development and budgets

Train, diagnose, and measure through `research train`, `research diagnostic`, and `research timing`. Save decisions to continue, revise, or stop through `research decision`. Use development timing as needed, before training, after pausing, or after completion.

| Item | Default limits and counting |
|---|---|
| Development training trajectories | Normally at most 2, hard limit 3; before executing a third, explain an evidence-supported correction or an important unknown that affects selection |
| Designs | At most 3, including the initial design and designs screened only by timing |
| Development timing | 5 completed / 15 total; repeats and retries after failure accumulate under the protocol |
| Research diagnostics | 10 successful / 20 total, at most 600 seconds each; failures, timeouts, and interruptions count |
| Infrastructure recovery | Allowed only after host classification, default limit 2, subject to the corresponding counts |

- Failures, voluntary stops, and rewind/restart consume trajectories; pending work reserves capacity. Pause/resume does not reset steps, tokens, schedule, or consumed capacity.
- Exact continuation at the same iteration/SHA belongs to the original trajectory and requires full restoration of model, optimizer, compressor/EF, schedule, data position, and RNG. Train a new design from scratch with matched initialization and data order, initializing method state according to the design.
- Continuing a Parent checkpoint with a different method supports only local evidence and cannot replace from-scratch training quality. Call it exact continuation only when the original data position and necessary state are restored.
- Reusing a training prefix is valid only when method behavior and necessary state match.
- The first accepted training or timing request binds the design SHA; code changes require an appended design. `decision` does not execute subsequent actions; `design` does not modify code or start training.
- Diagnostics apply only to paused/completed trajectories. Diagnostic branches must not contaminate candidates or provide extra candidate training.
- Timing reuses matching host-validated results by default. Supply `repeat_reason` for deliberate remeasurement and use the latest completed matching result, rather than selecting the fastest. Completed but incomparable timing still counts as completed.
- Commit `submission.yaml` with `mode: research`, Spec references, and actual event telemetry. Newly submitted timing must declare the complete valid `period_steps` (null for nonperiodic methods); `requested_steps` is optional supplementary information. Handle legacy runs under their saved contracts.
- Run applicable correctness checks before training and fix failures first. A new Research algorithm need not be semantically equivalent to its Parent.
- Preserve failures, regressions, incomparable evidence, and unavailable evidence; do not rewrite original predictions. `improved`, local mechanism success, advisory U, and expected future Engineering gains do not establish practical success.

## 5. Freezing, Final evaluation, and Node lifecycle

Use `research finish` to nominate one completed iteration that passes development validity/quality/resource checks, or null. The host first freezes the selected SHA, design, compression policy, and workload mapping, then runs fresh mandatory Final timing, the fixed language-transfer task, and ResNet18/CIFAR100.

- Development timing is not a nomination prerequisite. Final timing has an independent budget, does not reuse development measurements, and adds no threshold for the size of the speedup. Cycle/event and hardware validity checks still apply.
- Bring all development work to a terminal state first. Use `decision` to explicitly stop planned/paused/infra-failed trajectories. If no candidate was executed, provide a representative code SHA as required by the interface.
- Define the candidate's compression mapping in `method_policy.py` through `configure_compression(args, workload)` and the shared hook, covering matrix and convolutional gradients. Do not change fixed training/evaluation settings such as data, budget, optimizer, seed, or precision.
- After freezing, do not switch candidates, tune against Final feedback, or reopen development. Recovery is allowed only for host-classified infrastructure failures, by default at most 2 per component. Preserve other completed components; reuse within a component is only for recovery of the same frozen evaluation. Vision retries start from scratch.
- If no eligible candidate exists, explicitly skip and preserve negative results. Failed components or components that exhaust retries retain their terminal states; other mandatory components still require terminal states or explicit skips.
- The host retains the Research reservation until all Final components are terminal or explicitly skipped, then releases it, retires Researcher, refreshes Evaluation, and independently runs Research Analysis over the complete record and Engineering subject to the quality gate.
- Root has only one fixed baseline Attempt A0. Do not reopen historical completed Nodes. Researcher does not claim another Node or write a separate final report.
- Research Analysis and Engineering proceed independently. Waiting for Engineering capacity is not a valid reason to skip.
- Parent eligibility requires published Research Analysis and Engineering either completed or explicitly skipped for quality failure/unavailability. The host allows at most 2 unfinished Nodes; waiting agents first resume existing work and budgets.
- Once the entire Node is finished and no task depends on large artifacts, the host cleans up dynamic artifacts while retaining small evidence and reports. Researcher retirement does not permit early cleanup.

## 6. Engineering execution and completion

On first entry, confirm the binding and Research anchor with `autoresearch engineering claim Nxxx`. Follow the Engineer role for semantics-preserving optimization and use `autoresearch engineering timing --sha <full-HEAD-SHA> --request-id <id>` for formal timing.

- The frozen `selected_iteration` and matching Research SHA anchor a developed Node; Root uses Spec/A0. Required handover must exist and be consistent. Preserve adaptive decisions, triggers, state, and random semantics.
- Set `mode: engineering` in `submission.yaml`. `base_commit` and `quality_anchor` always remain the fixed Research SHA; later candidates may build on a preceding Engineering Commit.
- Defaults are 4 successful / 10 total. Regressions with valid scores count as successful; failures and allowed retries count toward total; pending work reserves capacity. Recovery/rewind does not reset counts. Stop when either limit is reached.
- Current claim `run_limits` take precedence over old launch budgets. Within budget, continue until no well-supported, material, semantics-preserving end-to-end improvement opportunities remain. One improvement alone is insufficient reason to stop.
- After at least one formal Attempt is terminal and a stopping condition or limit is met, Engineer writes an Analysis covering all claim Attempts and submits it with `autoresearch engineering finish --analysis-file /absolute/scratch/final.json`.
- Before submission, wait for diagnostics to complete and archive necessary self-check evidence. See Engineer for output fields. The claim completes only after host sealing and versioned publication; there is no independent Engineering Analyzer.
- `finalizing` does not mean publication is complete. After submission, do not change the summary or add candidates; reuse the same content after interruption.

## 7. Waiting, retries, and evidence archiving

Execution commands wait for published results by default. After a connection interruption, Research reconnects with `research wait --node Nxxx --request-id <original-ID>`, retrieves state with `research status`, and requests recovery of interrupted host work with `research recover`.

- Use blocking waits supported by the current tools and continue when the task returns. Do not write separate watchers, poll at short intervals, or read logs periodically to wait.
- For a lost reply, reuse the same ID and payload; use a new ID for a new execution. Inspect the failure reason first, retain failure records, and do not blindly rerun or retry through empty commits.
- Engineering infrastructure retries at the same SHA use `--retry` and a new ID; lost replies reuse the original ID/SHA/flags. Research/Final recovery uses the same frozen version and does not rerun completed work.

Archive small self-check scripts and logs needed for later citations with `autoresearch evidence archive --node Nxxx --request-id <id> --files <paths...>`. Research archives before managed experiments while its assignment is active; Engineering archives before finish.

- Each bundle contains small regular files with unique basenames, totaling at most 32 KiB; each assignment allows at most 64 bundles. Preserve larger execution evidence through managed diagnostics.
- Save the returned public-relative paths. Logs must state actual commands, inputs, and scope. Temporary or retired Worktrees are not evidence anchors.
- The host records the authenticated uploader, their declared code-context SHA, and content hashes. Hashes verify content and the host confirms that the SHA exists, but does not certify that those bytes were executed. Attachments enter the role's frozen manifest as agent-provided evidence.

## 8. Evaluation and evidence navigation

The Node README links Spec, Evaluation, Analysis versions, candidate SHAs, and evidence. Fixed code can be inspected with `git show <SHA>:<file>`; `autoresearch refresh-evaluation --node Nxxx` rebuilds Evaluation needed for diagnostics. Analysis Markdown is for reading; the matching JSON holds the complete frozen record for inspection as needed.

- Current HEAD and historical `source_worktree` are not recovery anchors. Diagnostic IDs refer to immutable records; old Analysis `.md` files may contain JSON.
- Language tasks require final loss relative to the same-contract dense reference to be ≤ 1.05. Vision permits at most a 1.0 percentage-point drop by default, configurable by the operator. Use `references/evaluation_tasks.md` and saved contracts for fixed tasks and current contract versions.
- New contracts require matching dense/Root evidence; historical seeds are not new references. Do not invent hashes. U is advisory only and does not decide exploration, Parent selection, or nomination.
- Timing uses a complete cycle or the default nonperiodic window, normally 400 and at most 800 steps. Invalid events, missing rare events, or unmatched reference windows limit conclusions about long-term comparability.
- Final results reused by later Nodes are known transfer/robustness evidence and cannot be called independent campaign held-out evidence.
- Independent Research Analysis answers initial and iterative predictions and provides bounded incremental findings and questions. The host derives `handover.json`; source reports remain authoritative and child Specs retain the original snapshot.
