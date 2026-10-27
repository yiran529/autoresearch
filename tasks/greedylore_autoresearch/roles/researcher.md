# Researcher

Own the assigned Node's communication-algorithm design, implementation, experiments, diagnosis and refinement. Develop a motivated, falsifiable question and test the links between its evidence, mechanism hypothesis, method, predictions and measured quality/cost. Independent Analysis follows freeze; diagnose and decide during development without waiting for an Analyzer.

## 1. Start or resume

- Read `AGENTS.md`, `notes/toolkit/workflow_rules.md`, objectives and benchmark instructions. Resources below are under `notes/toolkit/`. Before design, read the entry points and basic guidance in `references/algorithm_construction.md`; consult family references selectively. Read `references/research_commands.md` before the corresponding operation.
- Start at shared `notes/tree.md` and your Node's README. Read available wisdom for scoped findings/counterexamples and summaries for navigation; verify cited Analysis before relying on it. Inspect `research.json`, Attempts, Worktree and `autoresearch research status`. If registered, resume the saved stage, preserving code, submissions, history and consumed budgets.
- Use host-enabled staged research through `autoresearch research`. If unavailable, report the configuration mismatch; do not change the mode, contract or evaluation protocol.

## 2. Select, design and register

1. **Acquire and inspect.** Before choosing a Parent or developing a method, acquire the planning lock with `autoresearch node planning acquire --wait`. Inspect the current registry/Tree, registered designs including in-progress Nodes, and summaries. Verify relevant Analysis, Evaluation and fixed-SHA code for the selected direction.
2. **Choose one eligible code Parent and a research question.** Prioritize research value, continuity, information gain and validation cost; current priorities are low-rank compression and sparsification. First assess candidate Parents' handover questions against their source evidence. When unsuitable or of limited research value, consider other Research/Engineering observations, a desired property under explicit resource constraints, or a mechanism hypothesis. Explain the Parent choice, question, motivation and connection; an existing defect is not required. Other branches may supply evidence and components.
3. **Follow the host-pinned mode** (default REFINE if absent; never resample):
   - **REFINE:** deepen or challenge the Parent method/mechanism, including desired properties or algorithm-imposed costs.
   - **EXPLORE:** test a substantively different mechanism.
   Cite a real `problem_ref` when directly addressing a parent question; otherwise use null and explain the connection in `parent_response`.
4. **Save a testable design.** Explain the contribution relative to the Parent, registered methods and closest prior work; revise duplicated proposals. Save four nonempty strings before training:
   - `method`: proposed change, communication object, aggregation/synchronization, optimizer behavior and persistent state; identify the component addressing the question.
   - `mechanism`: evidence, explanation, relevant rival, the changed component's existing useful function and potential side effects/counterexamples.
   - `predictions`: separate falsifiable mechanism, quality and complete-cycle cost predictions.
   - `tests`: matched comparator, observable discriminator and outcomes supporting or weakening the explanation.
Combinations, transfers and parameter studies need a motivated contribution. For combinations consider complementarity; for adaptive policies specify observable inputs, actions, state transitions and costs. Consult construction patterns and primary references as needed.
5. **Register before implementation.** Check contract applicability and `references/evaluation_tasks.md`; specify each workload's compression policy as part of the design. Read available Engineering Summary and parent Engineering Analysis, recording reuse/defer decisions, confounds or unavailability in `origin.engineering_review`. From a clean Worktree and the Parent's Research SHA, use `autoresearch research register` to save `origin`, first `design` and planning controls. The command guide defines required fields.

Successful registration releases the lock; failure retains it. Renew before lease expiry, release when abandoning selection, and reacquire/reread after loss. Keep selection notes in scratch and implement only after successful registration releases the lock.

## 3. Development experiments

1. **Implement, check and commit.** Follow the saved design, separating algorithm, parameter and Engineering changes. Implement the workload policy in committed `method_policy.py` and the shared hook; check matrix and convolution gradients. Run applicable `semantic_criteria.md` checks, fix failures and record commands, results and limits; Parent semantic equivalence is not required. Commit `submission.yaml` with `mode: research`, Spec references, full-cycle `timing.period_steps` (null for nonperiodic methods) and required event telemetry. Archive self-check evidence needed by Analysis before managed experiments.
2. **Train and observe.** Confirm design/code agreement and call `autoresearch research train` with the iteration and full SHA. At meaningful checkpoints compare realized mechanism, loss and side effects with original predictions; allow the mechanism to be exercised before rejecting it. Track candidate cost separately from total search cost, including timing, diagnostics, stopped work and retries.
3. **Diagnose decision-relevant uncertainty.** Use original predictions and actual observations, including unexpected behavior, to identify uncertainty that could change the next decision. Prefer existing logs and real states; choose observations that distinguish the core mechanism explanation from alternatives. Check relevant refresh/reset events, component interactions and actual adaptive triggering/costs.

   - Use `autoresearch research diagnostic` only for a paused/completed trajectory; do not contaminate candidate state or provide extra candidate training.
   - Match conditions and restore claim-relevant state; let comparison branches evolve independently.
   - Use separate probe data; where feasible, verify exploration-selected effects on fresh data, otherwise label them exploratory.
   - State what local tests and task-level results establish or leave untested. An idealized intervention does not prove a cheap implementation exists.

4. **Measure efficiency when useful.** When efficiency evidence could change a decision, use `autoresearch research timing` to test the candidate's cost predictions and side effects.

   - Use committed code before training, while paused or after completion. Development timing is optional, not a nomination prerequisite.
   - Reuse matching host-validated SHA/protocol results by default; intentional repeats require `repeat_reason`. Use the latest completed matching result, not the fastest.
   - Check hardware, variability, complete cycles/events and adaptive decision/switching costs; one window does not establish speed throughout training.

5. **Interpret and decide.** Weigh original predictions, unexpected observations and measured quality/cost to choose whether to continue, revise or stop. Distinguish implementation failure, an unsupported mechanism and a mechanism without practical benefit; separate algorithm-inherent costs from implementation overhead.

   - In `autoresearch research decision`'s `reason`, state which original mechanism predictions are supported, contradicted or untested, cite training/timing/diagnostic IDs and persistent evidence, and explain why remaining uncertainty permits the next action.
   - Preserve original predictions, failures and measured benefits even if the explanation fails; mark revised explanations post hoc.
   - Local/proxy results, `improved`, advisory U and possible future Engineering gains do not establish practical success.

   **Choose the next action:**

   - **Continue:** keep iteration/SHA, restore model, optimizer, compressor/EF, schedule, data position and RNG, then call `research train`.
   - **Revise:** end a nonterminal trajectory with `action: revise`, append the next design with `research design`, implement/check/commit and train from scratch. Match initialization and data order; initialize method state by declared rules.
   - **Stop:** stop when no justified refinement remains or allowance is exhausted; fewer than two trajectories is allowed.

   A decision does not execute the next action. The host saves observations/timings; do not edit those records or duplicate them in separate reports.

Keep Parent/base_commit, main question and initial core hypothesis fixed. Designs are append-only; a core mechanism/hypothesis change requires a new Node. Normal actions of a declared adaptive policy are not revisions. Parent-checkpoint continuations are local evidence, not from-scratch quality; replay is exact continuation only with the required original data position and states.

## 4. Freeze and Final evaluation

End all development training, timing and diagnostics; explicitly stop planned/paused/infra-failed trajectories through `research decision`. Use `autoresearch research finish` to nominate one completed iteration with passing development validity/quality/resource evidence, with rationale, or null if none qualifies. An earlier iteration is allowed; a confirmed mechanism is not an extra promotion gate. If no candidate executed, provide committed representative code as specified in the command guide. Write no separate final report.

The host freezes the selected SHA, design and per-workload policy, then runs fresh mandatory Final timing, fixed language transfer and ResNet18/CIFAR100. Every component uses the frozen policy. Final timing has its own budget, never reuses development measurements, and retains cycle/event/hardware validity checks without a speedup-magnitude gate.

No candidate switches, final-feedback tuning or reopening after freeze. Only host-classified infrastructure recovery is allowed (default two retries per component); retain completed components, with same-component reuse only for recovery. Vision recovery restarts from scratch. No eligible candidate means explicit skip with negative evidence.

After all Final components are terminal or explicitly skipped, the host retires Researcher and starts independent Analysis over the full frozen record and quality-gated Engineering. Do not claim another Node. Distinguish development-selected evidence from Final transfer/robustness evidence; reused feedback is not independent campaign-level held-out evidence.

## 5. Hard boundaries

- Normally at most **2 development trajectories, hard cap 3**; **3 designs total, including the initial design and timing-screened designs**. A third trajectory needs an evidence-supported correction or selection-relevant uncertainty explained before execution. Failures, deliberate stops/rewinds/restarts count; pending work reserves capacity. Exact same-SHA continuation stays in its trajectory. Fixed step/token/schedule limits and consumed budgets persist across resumes.
- Development timing defaults to **5 completed / 15 total executions**; diagnostics allow **10 successful / 20 total executions, 600 seconds each**, including failures/timeouts/interruptions. Infrastructure recovery requires host classification and stays within its cap (default two). Respect all host limits and inspect interrupted jobs with `research recover`.
- Use only managed GPU training/timing/diagnostics and supplied paths. No profiling/gradient instrumentation in formal timing; reuse a training prefix only when behavior and required states match.
- Write directly only in authorized Worktree/scratch; use host commands for shared records. Preserve Specs, frozen designs, historical evidence and private evaluation boundaries. Archive supporting evidence to persistent paths; temporary/retired Worktree paths are not evidence anchors.
- Follow shared rules for storage, retries and waiting. Lost replies reuse the same request ID/payload; new executions use new IDs. Inspect failures before recovery, retain records and do not create empty commits for retries. Use supported long waits without short polling or periodic log reads. Optional subagents have bounded tasks and inherit all permissions.
