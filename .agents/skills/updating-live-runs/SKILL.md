---
name: updating-live-runs
description: Use when synchronizing code, pausing, or restarting a system that owns long-running training, evaluation, or other durable jobs while preserving progress.
---

# Updating Live Runs

Manage source versions, loaded process versions, task identity, and completed
results separately. Preserve the run's scientific contract and durable history
while changing its execution environment.

## Establish the interruption boundary

Follow the strategy already chosen by the user. Inspect dependencies among the
manager, grader, agents, child jobs, and resource leases before assuming compute
can survive a control restart. A lease tied to a controller or agent PID may
cancel training when that process exits.

If the requested preservation strategy cannot work with the current design,
explain the concrete dependency and obtain a decision on the changed interruption
boundary. Continue preparation that does not depend on that decision. Preserve
existing authorization; avoid repeatedly asking about the same agreed strategy.

## Prepare the deployment

- Compare target source with the deployed copy. Identify necessary local
  adaptations and merge them deliberately; do not overwrite them blindly.
- Record source revisions, configuration, accepted request identities, sessions,
  and the durable state required for recovery. Keep credentials and private
  tokens out of user-facing logs.
- Establish which completed results must remain unchanged, which components may
  restart, and which version existing and subsequent jobs will use.
- Verify the prepared source and applicable runtime configuration before
  interrupting work. Keep a recoverable source/configuration snapshot and
  an operator record proportional to the change.

## Stop, synchronize, and recover

Identify processes by ownership and start identity before signaling them. Drain
or stop dependencies according to the agreed strategy, then synchronize source
and configuration and restart the same run through its normal launcher.

Preserve completed results, candidate identities, evaluation contracts, budgets,
and accepted request payloads. Recover incomplete work through existing recovery
entrypoints. Record infrastructure interruptions separately from method failures;
do not treat an interrupted partial training as a completed quality measurement.

If accepted work still references an expired process identity, prefer the formal
recovery interface. A necessary operator repair must validate current ownership
and process start identities, use appropriate locks, preserve scientific payloads
and accounting, and retain a before-and-after audit. A stale identity is not
permission to weaken authorization or reset budgets.

## Verify the resumed run

Check these independently:

- Deployed revision and retained local adaptations match the prepared source.
- The saved evaluation contract, candidate selection, budgets, and completed
  evidence remain intact.
- Control processes and expected agents are live, and the intended sessions
  and instructions were restored.
- Accepted jobs progress through recovery, resource leasing, actual workers,
  and log or step advancement. A PID or a `running` field alone is insufficient
  to establish resumed computation.
- Subsequent tasks receive the new code and instructions through the actual
  provisioning/reload path.

Report deployment completion and compute recovery separately. Identify remaining
queue, lock, dependency, or identity waits explicitly. A health-check timeout
requires checking actual process and job state; preserve the timeout record and
update the operator record when later evidence establishes recovery.
