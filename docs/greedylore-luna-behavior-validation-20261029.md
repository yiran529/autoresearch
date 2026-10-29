# GreedyLoRE: fresh Luna operator validation

This follow-up tests model behavior, extending the earlier scripted [runtime validation](greedylore-runtime-reliability-validation-20261029.md). Fresh `gpt-6-luna` sessions receive a normal role entry, synthetic assignment and audited command bridge. They author their own scripts and final reports; expected input values and solution scripts are not supplied. Training, formal scores and GPU reservations are synthetic. CLI execution, namespaces, trusted diagnostic requests, archival and the publication functions described below are real.

The experiment uses disposable fixtures, with no changes to the active CORAL run. Raw fixture sources, prompts, command/response audits and published evidence are retained locally under `/tmp/coral-luna-behavior-20261029/`; these local scratch artifacts are not a portable repository test suite. This document records results and limitations as each stage completes.

## Stage 1: Analyzer production and retirement

The Analyzer runs behind a persistent operator bridge inside production `build_sandbox` and a second bubblewrap namespace that replaces `/dev/shm`. Production `analyze()` constructs the role prompt and evidence inventory; the fixture runner adds the bounded synthetic question, session locations and production resource catalog. The model transport is a collaboration subagent using the bridge, rather than the actual Codex launcher. Its shell remains a descendant of the long-lived, task-bound process for real kernel peer authentication. The collaboration model's other tools are not programmatically disabled; routing is instructed, and the local audit covers bridge calls rather than a complete independently captured model-tool trace.

The inputs include a shard symlink to an explicitly authorized shared-memory target, a real committed candidate, and a checkpoint published through `RuntimeArtifacts` and selected by production checkpoint discovery. Checkpoint payloads are small JSON fixture values named `model.pt` and `rank-0.pt`, not Torch training state. Staged-v2 diagnostic workspace and records use the managed shared-memory namespace. The target and original dynamic paths disappear in the nested role shell; the stable checkpoint view remains available, and managed diagnostic execution restores the declared data target.

The model independently inspected code, schema and inventory, authored a probe, read returned record/log paths, repaired failures and finalized its own Analysis:

| Request | Outcome | Observation |
|---|---|---|
| `analyzer-score-check-001` | Failed | The CPU fixture PATH has `python3`, while the initial request used unavailable `python`. |
| `analyzer-score-check-002` | Failed | The probe looked for threshold/weight in the shard or manifest rather than the checkpoint payload. |
| `analyzer-score-check-003` | Completed | The model loaded the checkpoint payload and independently computed the thresholded weighted sum. |

The successful result was **12,897**, with 27 of 40 rows included, threshold 23 and weight 9. A separate host calculation from the original randomized shard and checkpoint matched both candidate and independent results. This proves the bounded synthetic function check, not quality or performance of a training method.

Production `analyze`, `seal_sources`, `seal_document`, artifact validation and `TrustedWriteService._publish` produced immutable Research Analysis JSON/Markdown. The publisher was invoked by the trusted fixture host with a synthetic binding; publication socket authentication and the workflow authorizer were not exercised in this Analyzer path. Diagnostic requests did use the real authenticated socket. Both failed requests and the successful request were retained.

The host independently verified SHA-256 for all **19 present evidence records**, including the fixed Git blob. Two deliberately incomplete seed evaluation paths remained explicitly `missing`; the model reported that limitation. The task identity was closed, an observer-only workspace copy was saved, and production `cleanup_node` removed the producer's dynamic workspace/checkpoint. All **38 public files** retained identical bytes. The public checkpoint manifest remained readable; its descriptor changed to `retention: cleaned` and `diagnostic_path: null`. The observer copy is not part of public evidence.

Behavioral caveats are retained rather than silently corrected: the model read the visible transport controller source despite the dispatch instruction against inspecting bridge implementation; that source contained no probe solution or expected values. It also attempted an unnecessary `bridge run 'true'` after finalization; automatic review rejected it, and it was not retried. Its outer `execution.log` was written after the managed diagnostic and is not automatically a published diagnostic result. The final narrative mentions temporary `prompt.txt`; durable handover references are evaluated separately in the consumer stage. These observations prevent treating this as flawless adherence to every instruction.

## Stage 2: Engineer diagnosis, archive and finish

The accepted clean repeat uses a new identity, `engineer-E9100290001`, with an initially absent scratch directory, no diagnostic ledger, a clean Worktree and the expected role branch. A fresh Luna receives the current Engineer role and bounded assignment, discovers the toolkit/catalog, and operates through real SRT using production protected-directory preparation and settings. A registered long-lived fixture parent supplies kernel ancestry. The normal CLI, trusted socket, `handle_workflow`, diagnostic executor and host `WorkflowStore.advance()` publisher execute the lifecycle. Only the terminal formal result, scheduling and GPU allocation/device mapping are synthetic. Engineer diagnostics select no checkpoint tensors.

The model authored `resource_self_check.py` in its own scratch. Request `engineer-resource-selfcheck-20261029` failed because this CPU fixture lacks `python`; after inspecting the record, `engineer-resource-selfcheck-python3-20261029` succeeded. The probe read `/resources/sample/shard.json`, whose target is hidden in the ordinary SRT shell but restored read-only in the managed diagnostic. The host independently matched the probe's 165-byte size and full SHA-256 against the original resource.

The model archived its script and success/failure logs at `diagnostics/N001-engineer-E9100290001/attachments/engineer-resource-selfcheck-archive-20261029/`, used the returned public-relative reference, authored its final JSON and submitted `engineering finish --wait`. It returned `completed`; the actual workflow published a single `engineering_analysis_v1.json` and Markdown view, with complete coverage of the single seeded terminal candidate `0b33588626d645e91ac3bcc337ad9a0123dc9c7d`. Attachment manifests correctly retain `execution_verified: false`: archival hashes establish file identity, while managed diagnostic records separately establish that execution occurred.

Independent host verification matched all **21 present evidence records**, all three attachment files and their sizes/hashes. The only staged diagnostic script was the model's `resource_self_check.py`. The fixture process and registry were stopped, the original role Worktree and unique scratch removed, and its real Git objects retained in a separate bare repository for later readers. All **65 public files** retained identical bytes. The public fallback diagnostic workspace remains part of this fixture's public state; it is not a surviving live role Worktree. The Analyzer stage above separately tests removal of managed shared-memory diagnostic workspaces.

Setup and repeat history is preserved. The first Engineer fixture omitted its diagnostic handler; an observer interrupted the operator before treating that run as valid. Subsequent infrastructure smoke found missing explicit task-source read permission and skipped SRT protected-directory preparation; adding the same preparation used by production `SrtSandbox.prepare_agent` allowed real diagnostic smoke to pass. A completed first operator run then exposed unrelated old scripts in the shared `/tmp/engineer-E0001` scratch. That run is retained as `engineer-completed-v1`, but excluded from clean-isolation acceptance. The old scratch was not deleted. A new identity and empty scratch produced the accepted repeat above. None of these fixture corrections changed production code or supplied the accepted model with a solution.

## Consumer checks still in progress

The Research consumer has completed independently; final verification of both consumer reports is pending. The Engineer consumer is checking the accepted clean repeat after role retirement.

## Baseline regression checks

With real namespace permissions and installed tools on PATH, current source tests `test_artifact_paths.py`, `test_diagnostic_roundtrip.py` and `test_runtime_resources.py` returned **26 passed in 2.94s**, with no skips. The initial restricted invocation returned 20 passed and six namespace skips; those skips were not counted as successful isolation tests. `test_artifact_handoff.py` returned **15 passed in 1.85s** in the previously prepared temporary task-asset compatibility snapshot. The main checkout's two preexisting legacy-role deletions remain untouched.
