"""Prepare a fixed Root and pending Attempt; resume explicitly to start workers.

Run with CORAL's existing Python environment. This preparation does not start
models or acquire GPUs. Normal project creation prepares the grader environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

SOURCE = Path(__file__).resolve().parents[3]
TASK = SOURCE / "tasks/greedylore_autoresearch"
sys.path[:0] = [str(TASK / "autoresearch/src"), str(TASK / "grader/src"), str(SOURCE)]


def recover_run(output):
    from autoresearch.bootstrap import run_seed
    from greedylore_autoresearch_grader.contract import EvalContract

    from coral.workspace.project import reconstruct_paths

    paths = reconstruct_paths(Path(output).resolve() / ".coral")
    EvalContract.load(paths.coral_dir / "private/eval_contract.yaml")
    run_seed(paths.coral_dir)
    return paths


def prepare_run(identity, output):
    from autoresearch.atomic import atomic_write_text
    from autoresearch.bootstrap import bootstrap_root, run_seed
    from greedylore_autoresearch_grader.contract import EvalContract

    from coral.config import CoralConfig
    from coral.grader.trusted_writes import TrustedWriteService
    from coral.workspace.project import create_project

    filename = {"greedylore": "task_greedylore.yaml", "arctopk": "task_arctopk.yaml"}[identity]
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output already exists; use --resume to read its frozen configuration")
    raw = yaml.safe_load((TASK / filename).read_text())
    raw["workspace"].update(
        repo_path=str(TASK / "seed"), results_dir=str(output.parent), run_dir=str(output)
    )
    # Both identities intentionally use the same protocol seed directory. The
    # project's automatic seed overlay therefore preserves the imported ARC hook.
    config = CoralConfig.from_dict(raw)
    config.task_dir = TASK
    paths = create_project(config, config_dir=TASK)
    atomic_write_text(paths.run_dir / ".coral_dir", str(paths.coral_dir) + "\n")
    contract = EvalContract.load(paths.coral_dir / "private/eval_contract.yaml")
    manifest = json.loads((paths.repo_dir / "source_manifest.json").read_text())

    def publish(kind, content, context):
        target = paths.coral_dir / "public/notes/nodes/N000/spec.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, content)
        return {"path": "nodes/N000/spec.md"}

    manifest["publish"] = publish
    submission = bootstrap_root(
        paths.repo_dir,
        manifest,
        seed=run_seed(paths.coral_dir),
        timing_period_steps=contract.raw["compressor"].get(
            "timing_period_steps", contract.raw["compressor"].get("update_gap")
        ),
    )
    declaration = yaml.safe_load(submission.read_text())
    policy = dict(
        version=1,
        source_commit=declaration["base_commit"],
        node_id="N000",
        mode="research",
        submission_hash=hashlib.sha256(submission.read_bytes()).hexdigest(),
    )
    atomic_write_text(
        paths.coral_dir / "private/root_bootstrap.json", json.dumps(policy, indent=2) + "\n"
    )
    a0 = subprocess.check_output(
        ["git", "-C", str(paths.repo_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    TrustedWriteService(paths.coral_dir, config).submit_root_attempt(
        worktree=paths.repo_dir, commit_hash=a0
    )
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", choices=["greedylore", "arctopk"], default="arctopk")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-python", type=Path,
        help="Prepare and CPU-check a public ML venv for diagnostics; stop a live run first",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and reuse the saved run without resetting its Root or Attempts",
    )
    args = parser.parse_args()
    paths = recover_run(args.output) if args.resume else prepare_run(args.seed, args.output)
    if args.diagnostic_python is not None:
        from autoresearch.diagnostic_runtime import prepare_diagnostics

        prepare_diagnostics(paths.coral_dir, args.diagnostic_python)
    print(
        json.dumps(
            {
                "run": str(paths.run_dir),
                "coral_dir": str(paths.coral_dir),
                "next_command": f"cd {shlex.quote(str(paths.run_dir))} && {shlex.quote(sys.executable)} -m coral.cli resume",
            }
        )
    )


if __name__ == "__main__":
    main()
