# Role Priority GPU Queue Implementation Plan

> **For agentic workers:** Use executing-plans inline, with test-driven-development and verification-before-completion. Keep the existing feature branch as requested.

**Goal:** Analyzer = Engineer > Researcher; FIFO within each class. A waiting Researcher must admit a later Engineer without restarting its evaluation or repeating quality. Running GPU stages are non-preemptive.

**Architecture:** Register live GPU requests in the existing host/UID lock domain. Plan feasible grants in role/FIFO order under allocator.lock, respecting individual sharing rules and per-run active-stage limits. Opt-in bounded waiting-worker slots and a continuously scanning daemon admit later engineering submissions while earlier research workers await GPUs; reserve one worker admission slot for high-priority requests.

**Spec:** User-approved fixed-role, non-preemptive priority in this conversation; existing `2026-10-17-greedylore-autoresearch-gpu-coordination-design.md` remains the basis for ownership, cleanup, timing isolation and evidence requirements.

## Constraints and interfaces

- No duration estimates or automatic aging. Strict priority can postpone Research indefinitely under sustained high-priority load.
- Keep `parallel.max_workers: 1` as the run's active GPU-stage limit; add opt-in `max_waiting_workers: 2` for host worker admission. Defaults leave other tasks unchanged.
- Task grader maps trusted research/engineering mode to researcher/engineer; diagnostic host entrypoint fixes analyzer role. Model diagnostic payloads cannot choose priority.
- `reserve_idle_gpus(..., source_role='researcher', execution_group=None, execution_limit=1)` preserves existing call sites and one-shot semantics.
- Queue registrations carry PID/start identity, deadline, role, GPU count, sharing policy and execution group. Successful reservation retains registration until release; exceptions/cancellation/dead owners remove stale registrations.
- Feasible FIFO planning skips requests whose GPU or execution-group conditions cannot be met. Granting a lower-priority request must not consume cards virtually allocated to an eligible higher-priority request.
- Never kill, restart or mark a waiting evaluation as infra solely to yield priority. Retain quality artifacts and existing Attempt/run history.

## Tasks

### 1. Shared role/FIFO GPU admission

Files: new task grader `scheduling.py`, modify `gpu.py`, extend `test_gpu_allocator.py`.

- [x] Add failing deterministic tests: low-priority request waits first, later engineer/analyzer registers, release GPUs, verify high starts first; same-class FIFO; eight-card disjoint grants; one-active-stage group limit; high infeasible on busy cards does not block eligible shared quality.
- [x] Observe unsupported priority API failures before implementation.
- [x] Implement atomic request records, process/deadline cleanup, feasible grant planning inside allocator.lock, reservation release cleanup.
- [x] Verify cancellation, process crash, partial acquisition and recheck failures preserve locks/queue correctness.

### 2. Role and run-limit integration

Files: task `grader.py`, `diagnostic_sandbox.py`, related integration tests.

- [x] Extend existing integration doubles to reject missing role/group/limit arguments; run red.
- [x] Pass source role from trusted host context. Research quality/timing both remain researcher priority. Analyzer diagnostics use the same queue, request 1–4 GPUs and preserve diagnostic budgets.
- [x] Set execution group to canonical coral_dir and execution limit from trusted parallel config; diagnostics use the same limit.
- [x] Verify no quality rerun between stages and no new terminal Attempt/retry for priority waiting.

### 3. Bounded dynamic daemon admission

Files: `coral/config.py`, `coral/grader/daemon.py`, `tests/test_grader_daemon.py`, config tests.

- [x] Add failing test with a waiting research worker, later engineering submission and active GPU-slot probe; engineering must finish first, research must continue once without restart.
- [x] Add validation/serialization tests for nonnegative integer max_waiting_workers, including rejection of bool/fractional values.
- [x] Implement opt-in continuously rescanning pool of max_workers + max_waiting_workers, duplicate suppression and reserved high-priority admission capacity. Keep legacy default daemon/drain behavior unchanged.
- [x] Verify stop/heartbeat behavior, failure propagation, history preservation, and original serial/concurrent daemon tests.

### 4. Documentation, review, verification and commit

Files: task.yaml, GPU_COORDINATION.md, this plan.

- [x] Enable two waiting workers for this task; document active-stage versus host-worker limits, strict role ordering and deployment restart.
- [x] Run full task suite plus daemon/config/subprocess/sandbox regressions and changed-file Ruff checks.
- [x] Request independent GPT-6.1-Sol medium review; fix substantive findings with failing regression tests first.
- [x] Verify diff scope/whitespace, record evidence, commit completed feature as explicitly requested. Do not alter live runs or launch real GPU workloads.

## Execution record — 2026-10-18

- Implemented shared role/FIFO request registration and feasible grants, strict per-run GPU-stage limits, trusted Analyzer/Engineer role integration, bounded continuously admitted host workers, and reserved high-priority worker capacity.
- Tests were written and observed failing before the missing priority API, dispatch path and role/group integration were implemented. Added a regression for a delayed wakeup after the request deadline.
- Real grader subprocess integration (synthetic GPU inventory, no GPU workload) confirms a later Engineer starts before a waiting Researcher and both retain exactly one grader invocation/run history record. Separate tests cover three pending Researchers, FIFO, infeasible high requests, cancellation and crashed owners.
- Final verification: task tests plus daemon/config/subprocess/sandbox regressions: **371 passed, 1 skipped**. Changed production and test Ruff checks and diff whitespace checks passed.
- Independent GPT-6.1-Sol medium review found no blocking defects and approved commit conditional on the full suite, which passed.
- Existing live runs/configs and real GPU workloads were not modified. New runs enable two extra waiting workers; existing runs require operator configuration synchronization and daemon restart.
