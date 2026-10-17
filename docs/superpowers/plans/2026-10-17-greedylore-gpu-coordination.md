# GreedyLoRE GPU Coordination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task in the current session. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Coordinate four-GPU quality/timing stages across independent CORAL runs, with optional sharing of other users' pretraining GPUs.

**Architecture:** Keep each run's serial evaluation queue. Task-local GPU allocation uses a common per-host/per-UID lock domain, complete reservations and bounded cancellable waiting. Separate quality/timing reservations; monitor timing process membership and persist resource evidence.

**Tech Stack:** Python 3.11+, Linux procfs, flock, pidfd where available, pytest.

**Spec:** `docs/superpowers/specs/2026-10-17-greedylore-autoresearch-gpu-coordination-design.md`

## Global Constraints

- Four GPUs per stage; keep `grader.parallel.max_workers: 1`.
- Default sharing disabled; sharing never admits current-UID or unknown-owner processes.
- Preserve training contract/hash and numerical scoring; record allocation policy exceptions explicitly.
- Do not touch running training processes or historical references; missing monitoring compatibility evidence blocks formal speedup.
- No new global daemon, FIFO service, combinatorial placement engine or implementation delegation. A minimal trusted parent-identity context in SubprocessGrader was added after review reproduced early reparenting.

## Task 1: Allocation and ownership

**Files:** modify `tasks/greedylore_autoresearch/grader/src/greedylore_autoresearch_grader/gpu.py`; create `.../runtime.py`; extend `tasks/greedylore_autoresearch/tests/test_gpu_allocator.py`.

**Interfaces:** `ProcessIdentity.read(pid)`, `reserve_idle_gpus(count, lock_dir, policy, *, allow_other_users=False, current_uid=None, wait_timeout=0, poll_interval=5, check_cancelled=None, on_event=None, inventory_provider=...) -> GpuReservation`; add optional process identities to `GpuDevice` without breaking existing positional constructors.

- [x] Write sharing tests where a foreign-UID PID is accepted only with the switch and current/unknown UIDs are rejected. Example: `assert reservation.indices == (1, 2, 3, 4)` when GPU 0 contains the current UID and GPU 4 is foreign-owned.
- [x] Run `.venv/bin/pytest tasks/greedylore_autoresearch/tests/test_gpu_allocator.py -q`; verify new tests fail because the sharing arguments are unsupported.
- [x] Add procfs identity reads, bounded inventory queries, shared directory validation, allocator lock, atomic complete reservation and retry loop. Keep default one-shot API behavior for callers not requesting waiting.
- [x] Test actual independent processes acquire disjoint four-card sets from an eight-card inventory, wait for release, release partial reservations, and reject cancellation/timeout.
- [x] Run allocation tests until green; inspect changes before moving on.

## Task 2: Process cancellation and timing guard

**Files:** modify `.../process.py`; extend `.../runtime.py`, `.../gpu.py`; extend `tests/test_process.py` and allocation tests.

**Interfaces:** `run_process_group(spec, *, check_cancelled=None, monitor=None, monitor_interval=1) -> ProcessResult`; `monitor(pid)` receives the actual session leader; process result exposes `infra_failure`. `Lifecycle` captures parent identity and installs/restores main-thread signals.

- [x] Add tests that launch real short Python subprocesses: a monitor detects interference and execution returns an infra reason; cancellation terminates descendants; callbacks never signal unrelated processes.
- [x] Observe new keyword-argument tests fail before implementation.
- [x] Replace blocking wait with a bounded polling loop, preserving timeout behavior and process-group cleanup. Convert monitor/inventory failures to infra diagnostics, including final checks.
- [x] Test real parent exit and signal handling, including restoration of handlers and PID-identity validation. Scope procfs/torchrun ownership to the current session/group, never UID alone.
- [x] Run allocation and process tests together until green.

## Task 3: Grader stage orchestration and reference validation

**Files:** modify `.../grader.py`; create `.../resources.py` for validated resource options, event evidence and monitoring-reference compatibility; extend `tests/test_grader.py`; create `tests/test_resources.py`.

**Interfaces:** `ResourceOptions.from_args(args)`, `ResourceJournal(root, options)`, `validate_monitoring_reference(reference, options)`. Grader's `grade()` wraps `super().grade()` in `Lifecycle`; `_run()` supplies its callbacks to the process runner.

- [x] Add failing integration tests that write candidate output artifacts and observe two distinct reservations, quality release before timing, and only timing hardware used for scoring.
- [x] Add failing tests rejecting incompatible hardware or missing reference monitoring evidence before the first timing command; monitor failures must be infra rather than candidate failures.
- [x] Implement configuration validation and stage acquisition; retain all successful quality outputs while the same worker waits for timing.
- [x] Persist per-attempt waiting/reservation/release/interference evidence and references in both success and failure bundles; keep existing evidence consumers compatible.
- [x] Implement exact monitoring-policy reference compatibility and a conservative explicit legacy-equivalence evidence path. Do not fabricate performance measurements.
- [x] Run all task tests and repair actual regressions without weakening reference validation.

## Task 4: Configuration, documentation, review and final verification

**Files:** `tasks/greedylore_autoresearch/task.yaml`, create task `GPU_COORDINATION.md`, update this plan/spec status as warranted.

- [x] Add defaults: sharing false, lock directory null, poll 5 s, wait 86400 s, monitor 1 s. Document common lock domains, deployment restart, runtime exception, reference evidence format and failure recovery.
- [x] Run `.venv/bin/pytest tasks/greedylore_autoresearch/tests -q`, appropriate daemon/process regression tests, and `.venv/bin/ruff check` on changed Python files.
- [x] Request an independent read-only implementation review; resolve substantive findings with regression tests.
- [x] Run `git diff --check`, inspect changed file scope, record verification and any unavailable real-GPU performance validation. Commit only feature files after verification; do not merge or publish.

## Execution record

Implemented in the existing feature branch at the user’s request. Initial component tests were run red before implementation. Independent GPT-6.1-Sol medium review identified early-parent-exit and disappearing-rank races; both were reproduced in regression tests, fixed and re-reviewed. Final combined task/daemon/subprocess/sandbox suite: 260 passed. No real GPU workloads or historical references were modified; monitoring performance/reference acceptance remains an operational prerequisite.
