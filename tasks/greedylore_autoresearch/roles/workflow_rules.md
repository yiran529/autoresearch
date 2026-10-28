# Shared Execution Supplement

This supplement keeps Researcher and Engineer execution recoverable, resource use consistent with host requirements, and evidence available for later inspection. Role documents define objectives and operations; Researcher's additional execution conditions are in `workflow_research.md`. Current limits follow host records. Interpret historical experiments under their saved contracts; new documentation does not relabel historical results.

## Resources and delegation

- Publish shared records through task commands; do not directly modify Tree, Spec, Attempts, formal logs, historical Artifacts or shared conclusions. Private evaluation material and credentials are outside your authorization.
- Use host-provided checkpoint/artifact paths. Store large tensors under dynamic-storage rules; do not copy them to the root partition or arbitrary paths, or change retention or cleanup policies yourself. Report space or recovery failures to the host.
- You may delegate substantive, bounded tasks to subagents. Their sandbox, resource, evaluation and publication permissions remain those of this task.

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
