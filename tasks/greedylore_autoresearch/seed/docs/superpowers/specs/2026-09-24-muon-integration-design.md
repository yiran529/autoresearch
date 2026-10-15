# Muon Integration Design

## Scope

Add Muon as an optimizer peer of AdamW while preserving every existing training,
communication, data, and scheduling path. Reuse the ARC-TopK Muon implementation
without changing GreedyLore or other communication hooks.

## Architecture

- `optimizer/muon.py` implements matrix Muon and an AdamW fallback in one
  `torch.optim.Optimizer`.
- `optimizer/muon_utils.py` owns Muon CLI flags and deterministic model-parameter
  grouping.
- Existing C4, GLUE, and CIFAR entrypoints gain only a `muon` optimizer choice,
  Muon CLI flags, and a construction branch.
- Existing AdamW and all compressor branches remain byte-for-byte unchanged
  except where a surrounding optimizer-selection list must include `muon`.

## Validation

Use CPU unit tests for Muon update semantics, grouping, state restoration, CLI
defaults, and scheduler compatibility. Compile all modified entrypoints without
executing datasets or distributed jobs.
