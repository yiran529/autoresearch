---
name: testing-agent-workflows
description: Use when changes affect agent-visible commands, sandbox mounts, artifact publication, or handoffs between roles, and unit tests cannot establish correct agent behavior.
---

# Testing Agent Workflows

Test the agent's observable behavior under representative constraints, beyond
the underlying functions. A successful producer command alone does not establish
that another role can consume its output.

## Obtain approval before execution

**Before starting a model-backed agent simulation, present its scope and obtain
explicit user approval.** This applies to simulation subagents, external model
calls, repeated behavioral samples, and GPU-backed workflow exercises. Installing
or automatically loading this skill is not approval to incur those costs.

Prepare a concrete proposal containing:

- The scenarios and observable behavior to verify.
- Agent/model choice, number of agents, and maximum runs or repetitions.
- A time/token/spend bound or estimate; label unknown prices as unknown.
- GPU or other scarce-resource use, if any, and the stopping condition.

Read-only inspection, scenario preparation, and ordinary local checks may proceed
while awaiting approval. Wait for an explicit answer before launching the costly
part; elapsed time is not consent. An existing explicit approval covering the
same scope is sufficient. Generic requests to test, installation approval, and
approval for an unrelated earlier simulation are insufficient.

Stay within the approved scope. Obtain approval before increasing the model
tier, agent count, repetitions, resource use, or cost ceiling. Do not repeatedly
request approval for actions already covered by the agreed scope. Report the
approval requirement with a link to this SKILL.md when asking for consent.

## Build a representative simulation

- Use an isolated environment with representative prompts, command entrypoints,
  permissions, and filesystem views. Use small synthetic tasks to exercise the
  workflow without requiring full training.
- Give an independent agent the task, inputs, and normal documentation. Avoid
  extra coaching that reveals the intended path, expected answer, or internal
  implementation. Where available and approved, use a fresh context.
- For path changes, exercise producer and consumer separately. Verify the
  producer uses paths visible inside its environment and a later role can read
  artifacts through the published interface after the producer workspace retires.
- Choose boundary scenarios relevant to the change: symlink targets, retired
  workspaces, publication failures, duplicate requests, oversized status
  responses, or interrupted execution. These are examples, not a mandatory
  suite for every change.
- Use the authoritative operation entrypoints. Keep simulations isolated from
  live runs unless the user explicitly authorizes live-system effects.

## Evaluate and report

Use executed commands, accessed paths, publication records, exit outcomes, and
consumer reads as evidence. An agent's claim that it used a mount or published a
file is insufficient on its own. Inspect unexpected behavior before proposing
changes; repeat simulations only within the approved scope and stopping bound.

Report coverage and gaps separately for automated local tests, live-model
simulations, GPU execution, and scientific quality/performance validation.
Synthetic workflow success establishes execution and handoff behavior, not
training quality or speed. Include observed resource usage when available.
