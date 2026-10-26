# Research command guide
Operational reference for the staged Researcher role. Consult relevant actions as needed; this is not an additional Researcher report.
## 1. Protocol, planning and registration
Staged mode requires host `grader.args.staged_research: true`; do not enable it yourself. Acquire `autoresearch node planning acquire --wait` before selection. Keep the token. If the configured protocol is unavailable, report it instead of switching workflows. Renew/release with `autoresearch node planning renew --token <token>` / `autoresearch node planning release --token <token>`. Successful registration releases the lock; failure retains it. Reacquire/reread after loss or expiry.
Staged actions use `autoresearch research <action> --node Nxxx --file /absolute/scratch/data.json --request-id <id>`. The file is a request prepared in scratch, not the shared `research.json`; braces/optional fields in the table below describe its structure, not executable JSON. Use a distinct short ID for each new operation.
| Action | Request JSON fields | Meaning |
| --- | --- | --- |
| `register` | `origin`, `design`; optional `planning_token`, `overlap_review` | Save origin and first design from a clean Worktree |
| `design` | `method`, `mechanism`, `predictions`, `tests` | Append next design after previous trajectory ends |
| `train` | `iteration`, `sha`; optional `pause_after_steps` | Start or continue that iteration at its fixed full SHA |
| `timing` | `iteration`, `sha`; optional `repeat_reason` | Independent fixed-protocol timing; default reuse for matching evidence |
| `diagnostic` | `iteration`, `argv`, `timeout`, `scripts_dir` | Managed diagnostic for a paused/completed trajectory |
| `decision` | `iteration`, `action`, `reason` | Append `continue/revise/stop` and evidence-based judgment |
| `finish` | `selected_iteration`, `reason`; optional `sha` | Freeze selection (or null) and conditionally run 60M |
| `status` | No file | Read host state/results |
| `recover` | No file | Trigger interrupted-job recovery/publication replay |
`origin` is the fixed Node starting point: Parent, question/hypothesis, evidence and selection rationale, delta/risks/falsification criteria, parent problem response and Engineering review. Each field below is required; `problem_ref` is null when the starting property/mechanism has no directly corresponding parent handover question, in either mode; explain the motivation and Parent connection in `parent_response`. Non-null references must identify a real parent question. The host assigns Node/mode/base SHA/evidence metadata; do not supply those host fields. The first `design` provides method, explanation, predictions and tests once; do not repeat it in a separate intent object. Planning token/overlap review are top-level registration controls, not research content. The host creates the initial Spec and compatibility handover fields automatically.
Registration example (replace example text and token with actual values):
```json
{
  "origin": {
    "parent_node": "N000",
    "problem_ref": "N000:current",
    "question": "One falsifiable research question",
    "hypothesis": "The initial core hypothesis connecting the design to that question",
    "selection_rationale": "Cite supporting evidence and explain why this question matters",
    "delta": "Substantive method difference from the Parent and closest methods",
    "risk": "Relevant side effects and confounds",
    "falsification_criteria": "Results that would weaken the core hypothesis",
    "parent_response": "Explain continuation, challenge or independent exploration of parent evidence",
    "engineering_review": "Applicable Engineering evidence and reuse/defer decisions, or unavailable reports"
  },
  "design": {
    "method": "Method, communication and persistent state",
    "mechanism": "Explanation, evidence and relevant rival",
    "predictions": "Separate mechanism, quality and complete-cycle cost predictions",
    "tests": "Matched comparator, observable discriminator and falsification conditions"
  },
  "planning_token": "TOKEN_RETURNED_BY_PLANNING",
  "overlap_review": "Methods inspected and substantive differences"
}
```
Follow the host-assigned mode: REFINE builds on the Parent method/mechanism; EXPLORE uses a substantively different mechanism. Handover is navigation, not a required source of ideas; property/mechanism starts are valid in both modes. The host freezes the initial design hash before publication, so recovery cannot substitute a different design.
## 2. Optional timing at any development stage
After implementing, checking and committing a design, write `/tmp/timing1.json`:
```json
{"iteration": 1, "sha": "FULL_COMMIT_SHA"}
```
```bash
autoresearch research timing --node N001 --file /tmp/timing1.json --request-id n001-timing1
```
The request queues a separate timing experiment; read `research status` after the normal long wait. Timing can run before training, while paused, or after completion. It neither launches training nor modifies its state/checkpoint. The first admitted timing or training pins the design's SHA; a changed candidate needs an appended design, including timing-only screening within the existing three-design limit.
The fixed experiment uses 130M AdamW BF16/FP32, the existing repetitions and complete-cycle policy; it does not time a restored training checkpoint or measure quality. Results and evidence links appear in `research.json` under the iteration's `timings[]` and in job status. Use measured conditions and relevant state/event coverage when interpreting them.
The same SHA and host timing context reuse the latest completed measurement by default, even with a new request ID; reuse consumes no execution slot. To deliberately repeat, include a nonempty `repeat_reason` with a new ID. All repeats remain visible; final scoring uses the latest completed matching measurement, not the fastest. Failed measurements retain their evidence. Infrastructure recovery is limited by the host (default 2 retries per SHA); fix candidate failures through a new design/SHA.
The independent host limits default to 5 completed and 15 total timing executions per Node (`research_timing_success_limit`, `research_timing_total_limit`), across all iterations/resumes, separate from diagnostics. A completed ineligible timing still counts as completed; failures count toward total. Missing final timing uses this same budget, so reserve capacity. If final timing is missing, `finish` supplies it before checking eligibility/freezing; failure or budget exhaustion returns evidence without starting 60M. Inspect status and decide whether to revise or finish with null.
## 3. Example: pause, inspect, continue, finish
Assume N001 is registered and the first design implemented, checked and committed. Replace `FULL_COMMIT_SHA` with the actual 40-character commit SHA. Filenames and request IDs are examples.
1. Write `/tmp/train1.json`:
```json
{"iteration": 1, "sha": "FULL_COMMIT_SHA", "pause_after_steps": 1000}
```
Submit:
```bash
autoresearch research train --node N001 --file /tmp/train1.json --request-id n001-train1
```
The endpoint is absolute optimizer update 1000, not 1000 additional steps. Submission returns queued state; inspect status after the appropriate long wait:
```bash
autoresearch research status --node N001
```
2. When paused, inspect existing evidence. If useful, write a diagnostic request, then submit through `research diagnostic`:
```json
{"iteration": 1, "argv": ["python", "experiments/probe.py", "--output", "results/probe.json"], "timeout": 600, "scripts_dir": "/absolute/authorized/scratch/probe-scripts"}
```
Write that request to `/tmp/diagnostic1.json` and submit:
```bash
autoresearch research diagnostic --node N001 --file /tmp/diagnostic1.json --request-id n001-probe1
```
Use a real script directory inside your Worktree or `/tmp/<your-agent-id>/`, and the host-provided interpreter. Scripts are staged as `experiments/`; commands start at the diagnostic workspace root. Write persistent results to `results/`. The host supplies `CORAL_DIAGNOSTIC_CODE` and permitted checkpoint paths; do not substitute arbitrary host paths. Inspect returned records/stdout/stderr and cite persistent evidence. Record your conclusion in `/tmp/decision1.json`:
```json
{"iteration": 1, "action": "continue", "reason": "Assessment against predictions with persistent run/diagnostic references"}
```
```bash
autoresearch research decision --node N001 --file /tmp/decision1.json --request-id n001-decision1
```
3. `decision` only records judgment: it does not resume training. Write `/tmp/continue1.json` with the same iteration/SHA and omit the pause endpoint to finish the fixed trajectory:
```json
{"iteration": 1, "sha": "FULL_COMMIT_SHA"}
```
```bash
autoresearch research train --node N001 --file /tmp/continue1.json --request-id n001-continue1
```
A later pause can instead use a larger absolute endpoint. Unchanged continuation stays in iteration 1.
4. To revise instead, record `/tmp/revise1.json` and submit it (already terminal trajectories need no extra stop):
```json
{"iteration": 1, "action": "revise", "reason": "Cite the observation and explain the within-hypothesis correction"}
```
```bash
autoresearch research decision --node N001 --file /tmp/revise1.json --request-id n001-revise1
```
Write `/tmp/design2.json`:
```json
{"method": "Second method and difference from iteration 1", "mechanism": "Evidence, explanation and rival", "predictions": "Updated mechanism/quality/cost predictions", "tests": "Matched control, discriminator and falsification"}
```
```bash
autoresearch research design --node N001 --file /tmp/design2.json --request-id n001-design2
```
Implement, check and commit; then write `/tmp/train2.json` with that new SHA and start from scratch:
```json
{"iteration": 2, "sha": "NEW_FULL_COMMIT_SHA", "pause_after_steps": 1000}
```
```bash
autoresearch research train --node N001 --file /tmp/train2.json --request-id n001-train2
```
`design` only records the design; it does not change code or train. Normal actions of a fixed adaptive policy are not revisions. A core hypothesis change requires another Node.
5. Once all work is terminal and iteration 1 has completed passing 130M quality, write `/tmp/finish.json`:
```json
{"selected_iteration": 1, "reason": "Selection rationale against frozen criteria"}
```
```bash
autoresearch research finish --node N001 --file /tmp/finish.json --request-id n001-finish
```
The host reuses matching timing or supplies it within budget, checks eligibility, then freezes and runs 60M for the selected candidate. Training completion itself does not launch timing. Use null if none qualifies; if no candidate ever ran, also provide a committed representative `sha` as unexecuted evidence. Do not switch candidates after freeze.
## 4. State, recovery and evidence
The host appends execution observations and `timings[]`, and keeps diagnostics; there is no observation-write action. Put interpretation, evidence links and continue/revise/stop reasoning in `decision.reason`. `design` never edits code or runs training; `finish` never starts another 130M search.
After a communication timeout reuse the same request ID and unchanged payload; it retrieves/replays the original operation. A deliberate new execution, including pause continuation or a corrected diagnostic, uses a new ID and consumes its applicable budget. Inspect status and use `recover` for interrupted jobs before resubmission. Only host-classified infrastructure failure allows same-iteration/SHA `train` recovery; 60M recovery uses `finish` with the same frozen selection. Default infrastructure retry cap is 2; publication replay must not rerun completed work. Diagnose failures and preserve records rather than blindly repeating them.
Use `autoresearch evidence archive --node Nxxx --request-id self-check-1 --files /absolute/scratch/check.py /absolute/scratch/check.log` before managed experiments when later Analysis needs those self-checks. Use the returned public-relative references. Uploads are agent-provided evidence, not certified executions; record actual commands/inputs/limits. Do not cite temporary paths. Commit existing submission fields; the first admitted `research train` or `research timing` binds iteration/design to SHA, so do not invent a submission iteration field.
## 5. Host-managed resources
Training, timing and diagnostics serially borrow the four-GPU reservation held through staged development and 60M finalization. Do not submit off-path GPU jobs or reacquire a group yourself. Diagnostic and retry limits remain those in the role/host contract.
The host controls `research_checkpoint_interval_steps` (default 1000 optimizer updates), with saves at requested pauses and training end. Periodic saving alone does not pause training. Use host-provided checkpoint/artifact locations under `dynamic_artifact_root` (default `/dev/shm/wyr_tmp`), organized by run/Node. Do not copy large tensors to the root filesystem, choose arbitrary host paths or alter retention; report space/recovery failures rather than falling back. Small records/logs persist; large artifacts are cleaned by the host after the whole Node finishes. Temporary memory storage is not a promise of tensor recovery after a machine restart.
