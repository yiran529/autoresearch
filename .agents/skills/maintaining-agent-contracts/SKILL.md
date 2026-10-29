---
name: maintaining-agent-contracts
description: Use when agent-facing behavior spans role prompts, command references, runtime instructions, or validators, and ownership or consistency is unclear.
---

# Maintaining Agent Contracts

Give each rule an authoritative home and propagate it through references or
generation. Keep the user-approved objective and interface consistent across
roles, source checkouts, generated instructions, and deployed copies.

## Locate the authority

Identify the target repository and delivery branch, development checkout, and
deployed source. Track modified, committed, integrated, and deployed states
separately. A temporary worktree is a development location; complete delivery
to the intended repository and branch within the existing authorization.

For the behavior being changed, identify the authoritative source:

| Concern | Authoritative home |
| --- | --- |
| Shared objective | The shared objective source, using the wording approved by the user |
| Operations and arguments | Command reference and the implementation defining the interface |
| Environment and execution constraints | Runtime guidance |
| Assignment ownership and state transitions | Lifecycle rules and their enforcing implementation |
| Accepted data and artifact formats | Schemas and validators |

Follow the repository's actual layout. If operation instructions are buried
in runtime guidance, establish a command reference and link to it rather than
copying the same instructions into every role.

## Propagate the change

- Change the authoritative source first. Keep role prompts focused on their
  assignment and references to the applicable rules.
- Trace affected templates, generated files, installed copies, and existing
  workspaces. Update generators rather than only their rendered output.
- Check file synchronization separately from process reload. An updated
  source file does not prove a running process or resumed agent uses it.
- Update design documentation to describe the implemented behavior, previous
  behavior, compatibility boundaries, and how future edits propagate.
- Run checks relevant to the changed contract. Follow the user's requested
  stage and commit cadence; preserve unrelated working-tree changes.

## Handoff

State the authoritative files changed, affected consumers, verification, and
commit/integration/deployment status. Distinguish repository changes, necessary
deployment adaptations, and temporary operator tools. Resolve conflicting
instructions at their source instead of adding another competing explanation.
