# Greedy Lore uv Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an isolated uv-managed Python environment for `greedy_lore`, including the project metadata and lockfile requested by the user.

**Architecture:** Use a project-local `.venv` under `greedy_lore`, with Python 3.11 and the CUDA 12.1 PyTorch wheel index declared in `pyproject.toml`. Resolve all runtime dependencies into `uv.lock`, then synchronize the environment from the lockfile.

**Tech Stack:** uv, Python 3.11, PyTorch 2.4.0 CUDA 12.1, Hugging Face tooling, CuPy CUDA 12.x.

**Spec:** User request in the current conversation: manage a new environment directly in `~/greedy_lore`, and produce `pyproject.toml` and `uv.lock`; do not poll during downloads.

## Global Constraints

- Keep `/home/wyr/dion/.venv` untouched.
- Create the new environment at `/home/wyr/greedy_lore/.venv`.
- Include the dependencies listed in `/home/wyr/greedy_lore/requirements.txt`.
- Follow the repository README's Python 3.11 and CUDA 12.1 PyTorch setup where available.
- Run dependency resolution and installation as one-shot commands without progress polling.

---

### Task 1: Define the uv project

**Files:**
- Create: `/home/wyr/greedy_lore/pyproject.toml`

- [x] **Step 1: Add project metadata and runtime dependencies**

Declare Python 3.11 compatibility, the listed requirements, PyTorch 2.4.0 with matching torchvision and torchaudio, and use `cupy-cuda12x` to provide the `cupy` import for CUDA 12.x.

- [x] **Step 2: Declare the CUDA 12.1 PyTorch index**

Pin `torch`, `torchvision`, and `torchaudio` to the explicit `pytorch-cu121` uv index at `https://download.pytorch.org/whl/cu121`.

### Task 2: Resolve and synchronize the environment

**Files:**
- Create: `/home/wyr/greedy_lore/uv.lock`
- Create: `/home/wyr/greedy_lore/.venv/`

- [x] **Step 1: Install Python 3.11 with uv if needed**

Use uv's managed Python installation because no Python 3.11 interpreter is currently available on PATH or in uv's managed installations.

- [x] **Step 2: Run one-shot lock and sync**

Run uv lock followed by uv sync from `greedy_lore`, without polling the download process.

### Task 3: Verify the deliverables

**Files:**
- Verify: `/home/wyr/greedy_lore/pyproject.toml`
- Verify: `/home/wyr/greedy_lore/uv.lock`
- Verify: `/home/wyr/greedy_lore/.venv/bin/python`

- [x] **Step 1: Check metadata and lockfile presence**

Confirm both requested files exist and the lockfile contains the project dependencies.

- [x] **Step 2: Import key packages and report versions**

Use the new environment's Python to import torch, transformers, accelerate, datasets, evaluate, wandb, loguru, cupy, scipy, and sklearn, and report the torch CUDA version.
