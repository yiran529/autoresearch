# GreedyLoRE runtime reliability implementation plan

> For agentic workers: use subagent-driven-development task by task; implementers do not delegate. The user approved implementation and a commit after each verified stage.

**Goal:** Repair communication, declared-resource access, diagnostic startup and acceptance feedback without changing the research contract.

**Architecture:** Keep host ownership and immutable evidence. Share bounded transport helpers, derive status views from saved state, resolve declared resources once for mounts and session descriptions, and split operational guidance from performance methodology.

**Tech Stack:** Python 3.11+, argparse, pytest, Unix sockets, bubblewrap/SRT, Markdown.

**Spec:** `docs/greedylore-runtime-reliability-design-20261029.md`.

**Implementation status:** Tasks 0–4 are complete at `8d206cb`; whole-range code review of `e0c3302..8d206cb` found no Critical or Important issues; its one minor documentation finding is addressed in the final record. Stage commits: Task 0 `fa57162`; Task 1 `304aa04`, `3520968`; Task 2 `f08bd0e`, `513c4c8`; Task 3 `8c23328`; Task 4 `8d206cb`. See the [validation record](../../greedylore-runtime-reliability-validation-20261029.md) for evidence, limits and operational decisions.

## Global Constraints

- Work in `/home/wyr/CORAL`, branch `feature/greedylore-autoresearch`; preserve unrelated changes and the two pre-existing legacy-role deletions.
- Do not modify or restart the active experiment run, its source checkout, storage root, journal or scientific evidence.
- Requests remain 64 KiB; responses have a separate 4 MiB cap, with bounded valid errors.
- Preserve legacy status keys and default behavior; add progress and an explicit summary option.
- State/resource projection must not mutate saved records or add evidence to sealed Analysis.
- Only host-declared data targets and host-selected checkpoints may be mounted, read-only; never follow arbitrary model-supplied links.
- Preserve evaluation protocols, SHAs, trajectory/diagnostic/formal budgets, saved history and permission boundaries.
- Preserve low-rank/sparsification and mechanism preferences, combination/adaptive guidance, and explicit 1800-second waits with the original session/request identity.
- No GPUs, formal training, live model calls, push or deployment are needed for this development task.
- Use red/green regression tests; each task ends in a focused commit after verification and self-review. Stage only the task's files.
- Use per-command Git identity `Codex <codex@openai.com>` as existing commits do; do not set global configuration.

## Task 0: Separate Engineer command guidance

**Files:** Create `tasks/greedylore_autoresearch/references/engineer_commands.md`; modify `references/engineer_runtime.md`, `roles/engineer.md`, relevant prompt/reference tests, and the design implementation record.

**Interfaces:** `engineer_commands.md` owns command syntax, inputs, outputs and recovery. `engineer_runtime.md` owns performance analysis methods. Shared rules remain in `workflow_rules.md`.

- [x] Read the runtime reference and Engineer role end-to-end. Move managed-execution interfaces and final Analysis JSON/submission syntax to the new command guide. Keep analysis/stopping requirements in the role without weakening them.
- [x] Update role links and the task README where it names interface locations. Keep useful methodology links pointing to runtime.
- [x] Verify all moved requirements remain reachable: diagnostics budgets/paths/ownership, fixed Research anchor, timing retries, finalizing vs published, complete attempt coverage, waiting and stopping conditions. Do not invent future commands in delivered prompts.
- [x] Run affected prompt/role tests (discover with `rg --files .../tests | rg 'prompt|role|reference'`) and check links. This reversible prose move needs coverage review, not tests that mirror text.
- [x] Update the design record and commit only these changes as `docs(autoresearch): separate Engineer command reference`.

## Task 1: Bounded transport and stage-aware status

**Files:** `coral/grader/trusted_writes.py`; a dependency-free shared protocol helper if useful; task `diagnostic_client.py`, `diagnostic_session.py`, `staged_research.py`, `cli.py`; focused new status helper/tests; existing `tests/test_trusted_writes.py`; `references/research_commands.md`; design record.

**Consumes:** existing newline JSON envelopes and staged journal/ResearchRecord. **Produces:** shared request/response constants and framing helpers; backward-compatible `status()` plus `status(summary=True)` and additive progress; CLI `research status --summary`.

- [x] Add failing UDS tests: valid response >64 KiB round-trips; >4 MiB response returns bounded valid error; empty/truncated newline frames are distinguished; request cap remains 64 KiB. Include copied Analyzer client, not only host client.

```python
def test_status_summary_is_readonly(tmp_path):
    from test_staged_research import host
    h, _, _ = host(tmp_path)
    h.enqueue('train', {'iteration': 1, 'sha': 'b' * 40}, 'train', {})
    before = h.path.read_bytes()
    value = h.status(summary=True)
    assert value['jobs']['train']['status'] == 'queued'
    assert value['jobs']['train']['sha'] == 'b' * 40
    assert h.path.read_bytes() == before
```

- [x] Run focused tests red. Implement shared framing without duplicated limit constants in the copied shim; copy the needed stdlib-only helper alongside it. Preserve authentication and protocol version.
- [x] Add the summary projection and explicit CLI flag; keep ordinary status/wait keys. Derive progress from host state/record and authoritative published Attempt when available; pending/unknown must not become eligible merely because execution completed.
- [x] Cover development passed + Final pending; partial Final failure; publication pending/completed; legacy journal with missing new fields; large timing/probe data; state/Analysis immutability.
- [x] Run `tests/test_trusted_writes.py`, `test_staged_research.py`, `test_staged_scheduling.py`, relevant diagnostic/client and CLI tests. Document actual fields/summary behavior, update design, commit as `fix(autoresearch): bound responses and explain staged progress`.

## Task 2: Shared declared resources and readable checkpoint responses

**Files:** new task `runtime_resources.py` or similarly focused helper; `artifact_paths.py`, `runtime_artifacts.py`, `staged_diagnostics.py`, `engineer_diagnostics.py`, `diagnostic_sandbox.py`, `diagnostic_session.py`, `checkpoints.py`; staged public-response boundary; task configuration; resource/path/Engineer/roundtrip tests; shared rules and both command guides; design record.

**Consumes:** trusted diagnostic configuration, private checkpoint ledger and original execution state. **Produces:** a resolved resource description used by bindings and prompt catalog; public checkpoint descriptors separate from host storage.

- [x] Add red tests for explicit dataset root with a symlink to an authorized external target, missing/out-of-scope/looping target rejection, bounded discovery, and a real sandbox that replaces `/dev/shm` then reads the selected shard.

```python
def test_declared_link_target_must_be_explicit(tmp_path):
    from autoresearch.runtime_resources import resolve_resources
    data, target = tmp_path / 'data', tmp_path / 'target'
    data.mkdir(); target.mkdir()
    (target / 'shard').write_text('original stream')
    (data / 'shard').symlink_to(target / 'shard')
    import pytest
    with pytest.raises(ValueError):
        resolve_resources([{'id': 'c4', 'path': str(data), 'target_roots': []}])
```

- [x] Run red, then implement a host-only resource schema supporting `id`, `path`, explicit `target_roots`, optional stable `view`; existing runtime-library roots remain valid. Resolve once, install read-only bindings after tmpfs, render descriptions from the same object. Preserve original paths required by saved streams.
- [x] Wire all three roles and actual diagnostic execution. Do not evaluate candidate code in the trusted controller or grant a whole shared-memory filesystem.
- [x] Add public response projection for checkpoint metadata/ref/access/retention. Preserve private host paths and hashes. A legacy public checkpoint field can be projected to readable metadata only if its access scope is explicit. Managed diagnostics still select exact tensor roots via the host ledger.
- [x] Cover ordinary agent metadata read, selected tensor read in diagnostic view, foreign tensor denial, post-cleanup/pruning reporting, repeated request identity, and no evidence-manifest growth on reads.
- [x] Run focused resource, artifact, diagnostic, staged and handover tests, including real SRT/nested bwrap where available. Update docs with the actual config and interface, then commit as `fix(autoresearch): resolve declared resources across sandbox views`.

## Task 3: Local DDP environment and preflight

**Files:** new focused preflight/launcher helpers; `diagnostic_sandbox.py`, `diagnostic_session.py`, `staged_execution.py`, task `cli.py`; submission validator reuse; sandbox/preflight/launcher tests; command guides, shared rules, design record.

**Consumes:** resolved resources from Task 2 and existing committed submission validation. **Produces:** deterministic local-name support, IPv4 launch capability, explicit CPU-only preflight output tied to candidate SHA/contract.

- [x] Add red real-namespace localhost resolution and loopback connection tests; preserve deliberately configured name-service mounts. Add a parser/launch test for the actual proposed IPv4 helper and port handling.
- [x] Implement minimal localhost hosts support and the reusable launcher. A 2-process CPU rendezvous is sufficient to test mechanics; no formal GPU execution or new budget gate.
- [x] Add explicit preflight CLI using existing submission validation with a genuine authenticated context/host execution when protected inputs are needed. Validate required assets and resources before work starts, and report phase-specific errors; no arbitrary host-path requests or candidate imports in the trusted process.
- [x] Static entry reporting may inspect committed files or run isolated loading. It must state its limits and never certify full semantic equivalence. Detect missing legacy protocol assets without restoring user deletions or silently switching protocols.
- [x] Test invalid metadata, missing data/entry/protocol file, valid staged setup, host credential isolation, and no GPU reservation or journal/budget mutation during checks.
- [x] Run diagnostic sandbox, environment, submission, CLI and real namespace tests. Update actual help/examples and design record; commit as `fix(autoresearch): preflight isolated diagnostic environments`.

## Task 4: Submission receipt and final integration

**Files:** `coral/hooks/post_commit.py`, `coral/cli/eval.py` as required; task `cli.py` Engineering timing branch; submission/CLI tests; `engineer_commands.md`, `research_commands.md`, `workflow_rules.md`; current architecture/artifact docs; final design record.

**Consumes:** authenticated durable Attempt acceptance. **Produces:** immediate accepted-SHA receipt before grade waiting, with final output compatibility.

- [x] Add red test that acceptance feedback is emitted after trusted return but before grade polling; rejected/moving HEAD produces no accepted event or Attempt. Check `--no-wait`, lost reply/request replay and machine-output separation.
- [x] Implement stderr/callback progress feedback without weakening HEAD/ownership checks or creating a second submit transaction. Apply the same distinction to Engineering timing.
- [x] Review moved prompts and source-of-truth coverage: roles reference command guides; runtime contains methodology; shared rules retain complete 1800-second waits and recovery requirements; commands parse and paths match actual selected views.
- [x] Run all affected tests plus complete staged simulation, actual SRT/bwrap probes, lint and diff checks; explain pre-existing failures separately.
- [x] Complete the independent whole-range review and address its documentation finding.
- [x] Update architecture/artifact docs by linking the new design and correcting touched descriptions; do not rewrite unrelated historical scientific protocols.
- [x] Commit as `fix(autoresearch): acknowledge accepted submissions before waiting`; report commits, exact validation and that live deployment remains separate.
