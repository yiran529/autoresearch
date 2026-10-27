# Unified artifact access implementation plan

> Execute inline with the executing-plans skill. The user approved implementation in chat.

**Goal:** Give agents readable diagnostic/checkpoint paths through one shared mapping, while keeping host storage and evidence provenance intact.
**Architecture:** A dependency-free `artifact_paths.py` owns host-to-view mappings. Sandbox mounts, Analyzer prompt paths and diagnostic responses consume that mapping. Normal worktrees and public node documents keep their existing access rules.
**Tech stack:** Python, pytest, bubblewrap, existing SRT installation.
**Spec:** Approved design in this conversation: shared checkpoint views, public diagnostic results, unchanged storage/cleanup/permissions.

## Constraints
- Dynamic tensors remain under the configured dynamic artifact root, default `/dev/shm/wyr_tmp`.
- Only host-selected checkpoints cross into a diagnostic session, read-only.
- Host evidence records retain original paths and identities; response projection never mutates them.
- Preserve GPU budgets, ownership checks, same-request recovery and node cleanup.
- Preserve unrelated prompt changes and deletions in the main checkout.

## Task 1: Shared mapping and actual sandbox access
Files: new `autoresearch/src/autoresearch/artifact_paths.py`; update `diagnostic_sandbox.py`, `diagnostic_session.py`, `diagnostic_client.py`; tests in `test_artifact_paths.py` and `test_diagnostic_roundtrip.py`.
- [x] Add failing tests: nested namespaces must read a selected checkpoint; code/records remain available; an unselected checkpoint and outside files stay hidden.
- [x] Run tests and record the failure before implementation.
- [x] Implement `PathView.path`, `PathView.rewrite_text`, `PathView.project_result`, and `diagnostic_view(workspace, checkpoint_paths=())`.
- [x] Build readonly alias mounts and render prompt paths from `diagnostic_view`; preserve frozen writable sources. Return mapped Analyzer results from the host boundary, removing shim-specific rewriting.
- [x] Verify corrected execution and readonly access using real nested bubblewrap/socket/CLI.

## Task 2: Public results for other roles
Files: update `engineer_diagnostics.py`, `staged_execution.py`; extend diagnostic/CLI simulation tests.
- [x] Add failing cases for dynamic-root Engineer and Researcher responses: all returned diagnostic file paths must open in the agent's sandbox and survive cleanup.
- [x] Project results only after public archival, using `PathView([(host_root, public_root)])`; leave raw record JSON and private ledgers unchanged.
- [x] Verify repeat request IDs return the same projected response without reexecution.

## Task 3: Validation and integration
- [x] Run diagnostic, checkpoint, sandbox, staged-research and analysis recovery tests.
- [x] Run all staged workflow simulation scenarios, with CPU replacements only for expensive execution/GPU reservation.
- [x] Run lint/diff checks, inspect permission boundaries and record verification below.
- [x] Commit the completed change and integrate it into the original branch without modifying unrelated files.

## Verification
- Baseline: 119 diagnostic/Engineer/analysis recovery tests passed.
- Main regression run: 191 tests passed in 183.11 seconds, including six full staged scenarios: success, transfer quality failure, transfer execution failure, no eligible development, vision execution failure and vision quality failure.
- Checkpoint/runtime/evidence handoff regression run: 57 tests passed in 11.99 seconds.
- Final directly affected regression run: 103 tests passed in 6.46 seconds; production-generated SRT permissions and Analyzer prompt mapping included.
- Real bubblewrap namespaces, trusted Unix socket and standalone CLI exercised with CPU fixtures. Selected checkpoint aliases are readable and read-only; unselected files remain hidden. The same alias works in the simulated model namespace and real diagnostic executor.
- Real SRT with `build_srt_settings` reads archived logs/results before and after dynamic cleanup and denies evidence writes. Unarchived/empty outputs are reported unavailable instead of returning missing paths.
- Sol high review found no remaining blocker after strengthening production SRT/prompt coverage. No live model invocation, formal training, GPU work, role prompt changes or old-run runtime updates were performed.
- Static E9/F/I checks and `git diff --check` passed.

## Implementation notes
- `diagnostic_view` deterministically maps selected roots to the shared workspace; checkpoint aliases use a digest of the host root, so same-name checkpoints cannot collide in one view.
- `PathView` supplies path/prompt/result projection. Host evidence JSON and request ledgers retain their original paths; projection happens at the agent response boundary.
- Own-worktree and other-node public-document access rules are unchanged. Authoritative checkpoint selection/ownership remains in the existing host ledger and sandbox validation.
- Heavy files stay in the configured dynamic root. Archived public paths remain useful after normal node cleanup. Existing runs pinned to a separate CORAL checkout must be updated explicitly to use this implementation.
