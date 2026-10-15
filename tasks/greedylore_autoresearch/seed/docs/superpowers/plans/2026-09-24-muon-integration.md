# Muon Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ARC-TopK's Muon optimizer to all existing training entrypoints without changing existing training or compression behavior.

**Architecture:** A self-contained optimizer module combines matrix Muon with AdamW fallback groups. A small builder classifies model parameters, while each training entrypoint only exposes flags and adds one optimizer-construction branch.

**Tech Stack:** Python 3.11, PyTorch 2.4, unittest, DDP-compatible optimizer APIs.

**Spec:** `docs/superpowers/specs/2026-09-24-muon-integration-design.md`

## Global Constraints

- Do not change existing training-loop, data-loading, scheduler, or communication-hook behavior.
- Do not upgrade or add dependencies.
- Reuse ARC-TopK Muon semantics: Nesterov momentum, five-step BF16 Polar Express, spectral LR scaling, and AdamW fallback.

---

### Task 1: Muon optimizer and parameter builder

**Files:**
- Create: `tests/test_muon.py`
- Create: `optimizer/muon.py`
- Create: `optimizer/muon_utils.py`
- Modify: `optimizer/__init__.py`

**Interfaces:**
- Produces: `Muon`, `polar_express`, `add_muon_args(parser, ...)`, and `build_muon_optimizer(model, ...) -> Muon`.

- [ ] Write tests for Nesterov update, weight decay, convolution flattening, AdamW fallback, parameter assignment, state restoration, CLI defaults, and scheduler behavior.
- [ ] Run `PYTHONPATH=. .venv/bin/python -m unittest tests.test_muon -v` and verify failure because Muon modules do not exist.
- [ ] Port the minimal ARC-TopK optimizer and builder implementation.
- [ ] Re-run the unit tests and verify they pass.

### Task 2: Training-entrypoint integration

**Files:**
- Modify: `c4/run_llama_pretraining.py`
- Modify: `glue/run_glue_no_trainer_HF.py`
- Modify: `pytorch-cifar/main.py`

**Interfaces:**
- Consumes: `add_muon_args` and `build_muon_optimizer` from Task 1.
- Produces: `--optimizer muon` in each existing entrypoint.

- [ ] Add failing CLI/import tests demonstrating that each entrypoint accepts or exposes Muon configuration.
- [ ] Verify the tests fail before modifying entrypoints.
- [ ] Add imports, Muon arguments, supported-optimizer entries, and one Muon builder branch per entrypoint.
- [ ] Verify CLI/import tests and Muon unit tests pass.

### Task 3: Scope and syntax verification

**Files:**
- Verify all files above.

**Interfaces:**
- Consumes: completed optimizer and entrypoint integrations.
- Produces: evidence that existing non-optimizer logic was not changed.

- [ ] Run `PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v`.
- [ ] Run `.venv/bin/python -m py_compile optimizer/muon.py optimizer/muon_utils.py c4/run_llama_pretraining.py glue/run_glue_no_trainer_HF.py pytorch-cifar/main.py`.
- [ ] Review the unified diff and confirm changes outside new files are limited to optimizer imports, flags, choices, and construction branches.
