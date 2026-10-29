# GreedyLoRE runtime reliability: validation and decisions

This record covers the implementation through `8d206cb` on 2026-10-29. All scientific execution boundaries in the staged simulations were synthetic. The active experiment was not deployed, restarted or modified; no GPU, training, model invocation, package fetch, external network access or push was performed. Independent whole-range code review of `e0c3302..8d206cb` found 0 Critical and 0 Important issues and one minor documentation issue, addressed in this record and the linked design update. The reviewer did not rerun the broad suites.

## Stages and commits

| Stage | Implementation commits | Scope |
|---|---|---|
| Preparation | `a73bddc`, `0d8774c` | Design and implementation plan |
| 0 | `fa57162` | Engineer command reference and methodology separation |
| 1 | `304aa04`, `3520968` | Bounded responses, stage status and envelope/progress corrections |
| 2 | `f08bd0e`, `513c4c8` | Declared resources, checkpoint views and traversal validation |
| 3 | `8c23328` | Authenticated preflight, namespace and IPv4 helper |
| 4 | `8d206cb` | Durable acceptance receipts, Engineering finish wait and unavailable resource catalog |

## Offline verification

Commands used the existing repository venv, offline mode and local namespace/socket escalation for actual sandbox probes. Set the installed tool path for the commands below:

```sh
export PATH=/home/wyr/.local/lib/coral-diagnostic-tools/unpacked/usr/bin:/home/wyr/.local/lib/coral-engineer-tools/bin:/home/wyr/.local/lib/coral-engineer-tools/node-direct/bin:/home/wyr/.local/lib/coral-engineer-tools/srt/node_modules/.bin:/home/wyr/.local/bin:/usr/local/bin:/usr/bin:/bin
export UV_OFFLINE=1 PYTHONPATH=/home/wyr/CORAL CORAL_SRT_SMOKE=1
```

- The affected core and task suite ran `pytest` against `tests/test_hooks.py`, `tests/test_cli_island_scope.py`, `tests/test_trusted_writes.py`, `tests/test_sandbox_srt_integration.py`, and task tests `test_preflight.py`, `test_diagnostics.py`, `test_staged_environment.py`, `test_submission.py`, `test_analysis_cli.py`, `test_sandbox_boundaries.py`: **234 passed, 2 failed in 23.63s**. The first command omitted the SRT directories from `PATH`, so the two failures were `FileNotFoundError: srt`. With the full path above, `.venv/bin/python -m pytest -q tests/test_sandbox_srt_integration.py` returned **2 passed in 1.43s**; namespace skips were not counted as successful probes.
The initial affected-suite command, including its incomplete PATH, was:

```sh
PATH=/home/wyr/.local/lib/coral-diagnostic-tools/unpacked/usr/bin:/usr/local/bin:/usr/bin:/bin UV_OFFLINE=1 PYTHONPATH=/home/wyr/CORAL CORAL_SRT_SMOKE=1 .venv/bin/python -m pytest -q tests/test_hooks.py tests/test_cli_island_scope.py tests/test_trusted_writes.py tests/test_sandbox_srt_integration.py tasks/greedylore_autoresearch/tests/test_preflight.py tasks/greedylore_autoresearch/tests/test_diagnostics.py tasks/greedylore_autoresearch/tests/test_staged_environment.py tasks/greedylore_autoresearch/tests/test_submission.py tasks/greedylore_autoresearch/tests/test_analysis_cli.py tasks/greedylore_autoresearch/tests/test_sandbox_boundaries.py
```

- A current-task source snapshot under `/tmp/coral-task3-compat`, with only the two historical legacy role assets supplied there, ran `test_workflow_state.py`, `test_engineer_diagnostics.py`, `test_runtime_resources.py`, `test_handoff_assets.py`, and `test_contract_static.py`: **115 passed in 8.04s**. A further test using the authenticated handler and actual Analysis publication verified `engineering finish --wait`: **1 passed in 0.35s**. The missing-resource regression demonstrated that frozen publication proceeds while execution validation still rejects the missing declaration.
The compatibility command, with the full tool path and environment above, was:

```sh
.venv/bin/python -m pytest -q /tmp/coral-task3-compat/tasks/greedylore_autoresearch/tests/test_workflow_state.py /tmp/coral-task3-compat/tasks/greedylore_autoresearch/tests/test_engineer_diagnostics.py /tmp/coral-task3-compat/tasks/greedylore_autoresearch/tests/test_runtime_resources.py /tmp/coral-task3-compat/tasks/greedylore_autoresearch/tests/test_handoff_assets.py /tmp/coral-task3-compat/tasks/greedylore_autoresearch/tests/test_contract_static.py
.venv/bin/python -m pytest -q /tmp/coral-task3-compat/tasks/greedylore_autoresearch/tests/test_workflow_state.py::test_engineering_finish_wait_completes_real_publication
```

- `CORAL_STAGED_SIMULATION_DIR=/tmp/coral-runtime-reliability-staged-final .venv/bin/python -m pytest -q tasks/greedylore_autoresearch/tests/test_staged_workflow_simulation.py` returned **6 passed in 194.44s**. These six scenarios use real CLI, trusted identities, controller, recovery, sealing and publication with synthetic training, GPU, model and resource boundaries.
- The following complementary current-HEAD run returned **97 passed in 6.29s**, without skips or warnings:

```sh
.venv/bin/python -m pytest -q tests/test_eval_wait.py tests/test_cli_test_eval_alias.py tasks/greedylore_autoresearch/tests/test_final_evaluation.py tasks/greedylore_autoresearch/tests/test_staged_research.py tasks/greedylore_autoresearch/tests/test_staged_scheduling.py tasks/greedylore_autoresearch/tests/test_runtime_artifacts.py tasks/greedylore_autoresearch/tests/test_artifact_paths.py tasks/greedylore_autoresearch/tests/test_research_wait.py tasks/greedylore_autoresearch/tests/test_diagnostic_roundtrip.py
```

Ruff checked all **18 changed source files** and **16 changed test files** in `e0c3302..8d206cb`, with `--ignore S101` for test assertions; both groups passed, and `git diff --check e0c3302..8d206cb` was clean. The file groups were selected with `git diff --name-only --diff-filter=AM e0c3302..HEAD`, splitting `.py` files by test path.

The two legacy role deletions predated this work and remain untouched and unstaged in the main checkout. Historical assets were supplied only to the temporary compatibility snapshot; these results do not assert a green full main-checkout suite.

## Independent operator simulations

These earlier simulations had synthetic model boundaries. The subsequent [fresh Luna behavior test](greedylore-luna-behavior-validation-20261029.md) separately records actual model-authored diagnostics, publication and downstream reference checks, including failures and coverage limits.

Each simulation used disposable synthetic fixtures and no active run or real GPU/model work:

1. A real CLI, UDS transport and controller exercised a 1.24 MB staged journal, bounded summary and same-request recovery. Fixture identity resolution was synthetic, so that exercise alone did not verify production registry authentication.
2. Actual sandbox probes read declared data through original and alias paths, rejected an undeclared link, exposed public checkpoint metadata and only the selected readonly tensor view, and verified pruning/cleanup status. A real SRT probe checked the managed shared-memory boundary.
3. Real UDS with registered temporary process identities exercised preflight success, a missing-resource failure and recovery for the same SHA. Actual bubblewrap and two-rank CPU Gloo verified localhost and the IPv4 launcher; fixture state and Attempt/journal counts remained unchanged.
4. Real CLI subprocesses, UDS and registered peer PIDs verified the full SHA receipt after the pending Attempt was durable, before completion. Validator rejection and moving HEAD produced no receipt, Attempt or slot growth. Engineering timing returned a receipt and parseable final JSON; finish returned `finalizing`, then identical-body `--wait` completed real fixture publication. Replaying created no second version; a changed body was rejected.

## Decisions and limits

- Work in the user's main feature checkout was explicitly authorized for this integration. This avoided detached work, while explicit file staging and ownership protected unrelated changes.
- The two existing legacy deletions were preserved as user state. This retains a known limitation for main-checkout legacy initialization tests; preflight reports missing assets for the selected protocol, and the temporary compatibility fixture supplies historical assets only for its tests.
- The user requested a commit after each completed stage. Local stage commits were made; push and active-run deployment require separate work.
- The IPv4 helper uses single-node static torchrun at `127.0.0.1`, selecting an automatic port via `AF_INET bind(0)` or accepting `--port`. Installed Torch c10d advertised `getfqdn()` with dynamic rendezvous; a private framework patch or custom process manager would be disproportionate. The automatic port is released before torchrun binds, leaving a short race; a conflicting bind needs a fresh request or an explicitly managed port.
- Preflight statically inspects committed entry sources; it does not prove dependency loading, method semantics or scientific equivalence. The current task configuration installs the package editable from retained task assets. Extracted-wheel smoke verified helper and name-service assets, not a complete standalone-wheel workflow.
- The configured C4 link target was missing on the host during this work. The host must restore the authorized data before rollout; execution and preflight fail closed if it remains unavailable. Public catalog unavailability does not prevent already-frozen publication and cleanup.

## Active-source check

The read-only before/after SHA-256 comparison found all five selected files unchanged: `coral/grader/trusted_writes.py`, `coral/hooks/post_commit.py`, and task `staged_research.py`, `diagnostic_sandbox.py`, and `artifact_paths.py`. This establishes only those selected source bytes; it does not assert that every process or runtime artifact was inspected. Scientific contracts, evaluation budgets and historical evidence were not changed by this implementation.
