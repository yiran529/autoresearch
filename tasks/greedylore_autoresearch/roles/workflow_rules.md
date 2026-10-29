# Shared Execution Supplement

This supplement keeps Researcher and Engineer execution recoverable, resource use consistent with host requirements, and evidence available for later inspection. Role documents define objectives and operations; Researcher's additional execution conditions are in `workflow_research.md`. Current limits follow host records. Interpret historical experiments under their saved contracts; new documentation does not relabel historical results.

## Resources and delegation

- Publish shared records through task commands; do not directly modify Tree, Spec, Attempts, formal logs, historical Artifacts or shared conclusions. Private evaluation material and credentials are outside your authorization.
- Use host-provided checkpoint/artifact paths. Store large tensors under dynamic-storage rules; do not copy them to the root partition or arbitrary paths, or change retention or cleanup policies yourself. Report space or recovery failures to the host.
- You may delegate substantive, bounded tasks to subagents. Their sandbox, resource, evaluation and publication permissions remain those of this task.

The host publishes `.codex/notes/toolkit/runtime_resources.json` at session preparation. It lists declared data paths, optional `/resources/<id>` views, and the exact read-only mounts used by managed diagnostics. If the catalog reports `status: unavailable`, report its error; reconciliation and saved publication continue, but diagnostic execution and preflight still reject missing resources before GPU use. These entries describe diagnostic access; they do not grant direct data or tensor access in the ordinary role shell. Original paths remain mounted in managed execution so saved data-stream state keeps its meaning. Missing, looping or undeclared link targets fail before diagnostic GPU execution; report the resource error to the host.

Checkpoint responses expose `checkpoint.manifest_path` and public-relative `checkpoint.ref` for ordinary read-only metadata access, plus `access` and `retention` (`available`, `pruned`, `cleaned`, or `unavailable`). The compatibility `checkpoint_path` is the public metadata directory, not a tensor directory. `checkpoint.diagnostic_path` supplies the address for a host-selected diagnostic mount, and is null when tensors are not retained. Tensor access requires a host-selected view: Researcher diagnostics select the iteration's saved checkpoints; Analyzer diagnostics select the fixed Attempt's checkpoints. Engineer diagnostics currently select no checkpoint tensors. A retained manifest does not imply retained tensors. Read these projections without editing original observations, manifests or evidence.

## Waiting and request identity

- Use 1800-second long waits (`yield_time_ms=1800000`) on a running command's terminal session, including `write_stdin` and enclosing `functions.exec`/`functions.wait`. These waits may return early when work ends; continue as soon as they return.
- Do not create separate watchers, poll at short intervals or read logs periodically to wait for completion.
- For a lost reply, reuse the original request ID and unchanged payload. Use a new ID for a new execution. Inspect the failure reason and preserve records before deciding whether to repair or retry.

## Persistent evidence

- Self-check archives contain small regular files with unique basenames: at most 32 KiB per bundle and 64 bundles per assignment. Preserve larger execution evidence through managed diagnostics.
- Save the public-relative paths returned by archiving. Logs should state actual commands, inputs and scope. Temporary or retired Worktrees are not evidence anchors.
- Attachments are uploader-provided evidence. The host records the uploader, declared code SHA and content hashes; this verifies content identity and SHA existence, not that those files were executed.

## Historical records and evaluation

Use fixed SHAs and host records to recover evidence identity. Current HEAD and historical `source_worktree` cannot replace them. Analysis Markdown is for reading; matching JSON preserves the complete frozen record. Legacy `.md` files may contain JSON directly. When Evaluation needs rebuilding, use `autoresearch refresh-evaluation --node Nxxx`.

- Fixed workloads, quality gates and timing validity follow `references/evaluation_tasks.md` and saved contracts.
- New contracts require matching dense/Root evidence. Historical seed results cannot directly serve as references for a new contract.

## Explicit environment preflight

Researcher and Engineer owners may run `autoresearch preflight --sha "$(git rev-parse HEAD)"` from their Worktree. The host reports immutable metadata, selected-protocol assets, declared resources, static entry sources and CPU-only namespace checks without an Attempt, lease, GPU reservation or budget mutation. Success is limited to the reported checks and does not certify method semantics or scientific quality. See the Researcher/Engineer command references for phase errors and the managed IPv4 DDP launcher.
