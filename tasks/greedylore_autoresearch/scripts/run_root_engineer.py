"""Run only the normal Engineer against an isolated copy of a real Root run.

Example: uv run python tasks/greedylore_autoresearch/scripts/run_root_engineer.py
  --source-run /path/to/root/run --output /path/to/new/experiment
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[3]
TASK = SOURCE / "tasks/greedylore_autoresearch"
sys.path[:0] = [str(TASK / "autoresearch/src"), str(TASK / "grader/src"), str(SOURCE)]


def command(*args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    from autoresearch.atomic import atomic_write_text

    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def validate_root(coral):
    from coral.hub.attempts import read_attempts

    attempts = read_attempts(coral, include_archived=True)
    if len(attempts) != 1:
        raise ValueError("source must contain exactly one formal Root Attempt")
    root = attempts[0]
    if (root.metadata.get("node_id") != "N000" or root.metadata.get("mode") != "research"
            or root.status in {"pending", "crashed", "timeout"} or root.archived
            or root.metadata.get("scores", {}).get("quality_gate", {}).get("value") != 1):
        raise ValueError("Root requires terminal passing quality evidence")
    if not list((coral / "public/notes/nodes/N000").glob("research_analysis_v*.md")):
        raise ValueError("source requires published Root Research Analysis")
    if (coral / "public/trusted.sock").exists():
        raise ValueError("source has an active trusted endpoint")
    return root


def snapshot(output):
    target = output / "coral-source"
    command("git", "clone", "--quiet", "--no-hardlinks", SOURCE, target)
    # Preserve the reviewed working-tree implementation without committing user work.
    changed = subprocess.check_output(
        ["git", "-C", str(SOURCE), "diff", "--name-only", "HEAD", "-z"]
    ).decode().split("\0")
    changed.append(str(Path(__file__).relative_to(SOURCE)))
    for relative in filter(None, changed):
        src, dst = SOURCE / relative, target / relative
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        elif dst.is_file():
            dst.unlink()
    return target


def prepare_srt_worktree(worktree):
    from coral.sandbox.srt import prepare_srt_worktree as prepare

    prepare(worktree)


def build_config(source_coral, output, model, effort):
    import yaml

    from coral.config import CoralConfig

    raw = yaml.safe_load((source_coral / "config.yaml").read_text())
    # Older Root runs predate the formal task's trusted-IPC configuration.
    task_sandbox = yaml.safe_load((TASK / "task.yaml").read_text())["agents"]["sandbox"]
    raw["agents"]["sandbox"].setdefault("extra_settings", {}).setdefault("network", {}).update(
        task_sandbox.get("extra_settings", {}).get("network", {}))
    raw["task"]["name"] = "GreedyLoRE Root Engineer workflow"
    raw["workspace"].update(repo_path=str(source_coral.parent / "repo"),
                            results_dir=str(output), run_dir=str(output / "run"),
                            setup=["uv sync --frozen", shlex.join([
                                "uv", "pip", "install", "--python", ".venv/bin/python",
                                "-e", str(SOURCE), "-e", str(TASK / "autoresearch")]), shlex.join([
                                ".venv/bin/python", str(Path(__file__).resolve()),
                                "--prepare-sandbox-worktree"])])
    raw["grader"]["private"] = [str(source_coral / "private/eval_contract.yaml"),
                                 str(source_coral / "private/dense_references")]
    raw["grader"]["setup"] = [shlex.join(["uv", "pip", "install", "--offline", "-e", str(SOURCE)]),
        shlex.join(["uv", "pip", "install", "--offline", "-e", str(TASK / "grader"),
                    "-e", str(TASK / "autoresearch")])]
    args = raw["grader"]["args"]
    # Short reconciliation publishes Engineering; no model worker is needed.
    args.pop("trusted_workflow_worker", None)
    args.pop("trusted_memory_worker", None)
    args.pop("trusted_memory_authorizer", None)
    raw["agents"]["dynamic"].update(enabled=True, max_researchers=0, max_engineers=1)
    raw["agents"].update(model=model, heartbeat_enabled=False)
    if shutil.which("srt"):
        raw["agents"]["sandbox"]["srt_command"] = [shutil.which("srt")]
    for assignment in raw["agents"]["assignments"]:
        assignment["model"] = model
        assignment["runtime_options"].update(model_reasoning_effort=effort,
            role_file=str(TASK / "roles" / ("engineer.md" if assignment["role"] == "engineering" else "researcher.md")))
    # Public local runtime/data dependencies are readable; old runs remain private.
    reads = raw["agents"]["sandbox"].setdefault("allow_read", [])
    reads.append(str(TASK / "autoresearch/src"))
    for root in args.get("diagnostic_runtime_roots", []):
        if root not in reads:
            reads.append(root)
    config = CoralConfig.from_dict(raw)
    config.task_dir = TASK
    return config


def prepare_codex_home(output, config):
    """Use Analyzer-style clean configuration, retaining sessions across resumes."""
    home = output / "run/.sandbox/codex-cli-home"
    home.mkdir(parents=True, exist_ok=True)
    auth = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
    target = home / "auth.json"
    copy_auth = auth.is_file() and not target.exists()
    if auth.is_file() and target.exists() and auth.resolve() != target.resolve():
        source_refresh = json.loads(auth.read_text()).get("last_refresh")
        target_refresh = json.loads(target.read_text()).get("last_refresh")
        copy_auth = bool(source_refresh and target_refresh and source_refresh > target_refresh)
    if copy_auth:
        shutil.copyfile(auth, target)
        target.chmod(0o600)
    for allowed in (config.agents.sandbox.allow_read, config.agents.sandbox.allow_write):
        if str(home) not in allowed:
            allowed.append(str(home))
    os.environ["CODEX_HOME"] = str(home)


def execute(args):
    from autoresearch.state import _engineering_budget

    from coral.agent.manager import AgentManager
    from coral.config import CoralConfig
    from coral.hub.attempts import read_attempts
    from coral.workspace.project import create_project, reconstruct_paths

    output, source_coral = args.output, args.source_run / ".coral"
    contract_path = output / "run_contract.json"
    if not args.resume:
        root = validate_root(source_coral)
        config = build_config(source_coral, output, args.model, args.effort)
        preserved = {str(path.relative_to(source_coral)): digest(path)
                     for path in [*sorted((source_coral / "public/attempts").glob("*.json")),
                                  *sorted((source_coral / "public/notes/nodes/N000").glob("research_analysis_v*.md"))]}
        write_json(contract_path, {"model": args.model, "reasoning_effort": args.effort,
            "source_run": str(args.source_run), "source_commit": subprocess.check_output(
                ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
            "source_diff_sha256": hashlib.sha256(subprocess.check_output(
                ["git", "-C", str(SOURCE), "diff", "HEAD"])).hexdigest(),
            "root_sha": root.commit_hash, "contract_hash": root.metadata["contract_hash"],
            "engineer_role_sha256": digest(TASK / "roles/engineer.md"),
            "default_prompt": True, "real_grader": True, "researcher_capacity": 0,
            "analyzer_and_memory_launches": False, "preserved_source_files": preserved})
        paths = create_project(config, config_dir=TASK)
        command("git", "-C", paths.repo_dir, "reset", "--hard", root.commit_hash)
        for name in ("attempts", "eval_logs", "notes", "diagnostics"):
            if not (source_coral / "public" / name).is_dir():
                continue
            command("cp", "-a", "--reflink=auto", str(source_coral / "public" / name) + "/.",
                    paths.coral_dir / "public" / name)
        shutil.copyfile(source_coral / "private/root_bootstrap.json",
                        paths.coral_dir / "private/root_bootstrap.json")
        # Normal role/toolkit initialization must use this source snapshot.
    else:
        paths = reconstruct_paths(output / "run/.coral")
        config = CoralConfig.from_yaml(paths.coral_dir / "config.yaml")
        config.task_dir = TASK
    prepare_codex_home(output, config)
    config.to_yaml(paths.coral_dir / "config.yaml")
    manager = AgentManager(config, config_dir=TASK)
    finished = threading.Event()
    result = {"stage": "running", "model": args.model, "reasoning_effort": args.effort}
    write_json(output / "status.json", result)
    started = time.monotonic()

    def observe():
        try:
            while not finished.wait(2):
                for worktree in paths.agents_dir.glob("engineer-*"):
                    for source, target in [(worktree / "AGENTS.md", "engineer.instructions.txt"),
                        (worktree / ".codex/roles" / f"{worktree.name}.md", "engineer.role.txt")]:
                        if source.is_file():
                            shutil.copyfile(source, output / target)
                for log in (paths.coral_dir / "public/logs").glob("engineer-*.log"):
                    target = output / f"{log.stem}.startup.json"
                    if not target.exists():
                        with log.open() as stream:
                            first = stream.readline()
                        if first.strip():
                            write_json(target, json.loads(first))
                state_path = paths.coral_dir / "private/autoresearch_state.json"
                if state_path.exists():
                    state = json.loads(state_path.read_text())
                    if set(state["nodes"]) != {"N000"}:
                        raise RuntimeError("unexpected child Node")
                    workers = list(state["engineering_workers"].values())
                    if workers and all(w["status"] == "completed" for w in workers):
                        result["stage"] = "completed"
                        manager.stop_all()
                        return
                if time.monotonic() - started > args.timeout:
                    raise TimeoutError("Engineer workflow observation timed out; resume preserves progress")
        except Exception as exc:
            result.update(stage="failed", error=f"{type(exc).__name__}: {exc}")
            manager.stop_all()

    watcher = None
    try:
        manager.resume_all(paths)
        watcher = threading.Thread(target=observe, daemon=True)
        watcher.start()
        manager.monitor_loop(check_interval=2)
    except Exception as exc:
        result.update(stage="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        finished.set()
        manager.stop_all()
        if watcher:
            watcher.join(timeout=10)
        if result["stage"] == "running":
            result.update(stage="failed", error="Manager ended before Engineering completion")
        attempts = [a for a in read_attempts(paths.coral_dir, include_archived=True)
                    if a.metadata.get("mode") == "engineering"]
        successful, total, pending = _engineering_budget(attempts)
        result.update(successful_runs=successful, total_runs=total, pending=pending,
                      attempts=[a.to_dict() for a in attempts], elapsed_seconds=time.monotonic() - started)
        contract = json.loads(contract_path.read_text())
        if not all(digest(source_coral / path) == value
                   for path, value in contract["preserved_source_files"].items()):
            result.update(stage="failed", error="source evidence changed")
        write_json(output / "status.json", result)
    if result["stage"] != "completed":
        raise RuntimeError(f"Engineer run did not complete: {output / 'status.json'}")


def main():
    if sys.argv[1:] == ["--prepare-sandbox-worktree"]:
        prepare_srt_worktree(Path.cwd())
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepared", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    # A terminal-launched manager has none of the parent coding session's
    # execution/routing controls. Keep the user's credential/config location.
    for key in list(os.environ):
        if key.startswith("CODEX_") and key != "CODEX_HOME":
            os.environ.pop(key)
    os.environ.pop("UV_PROJECT_ENVIRONMENT", None)
    os.environ.update(UV_OFFLINE="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", WANDB_MODE="disabled")
    args.source_run, args.output = args.source_run.resolve(), args.output.resolve()
    if (args.output == args.source_run or args.source_run in args.output.parents
            or args.output in args.source_run.parents):
        parser.error("output must be independent of source-run")
    if not args.prepared:
        if args.resume:
            target = args.output / "coral-source"
        else:
            validate_root(args.source_run / ".coral")
            args.output.mkdir(parents=True, exist_ok=False)
            target = snapshot(args.output)
        script = target / Path(__file__).relative_to(SOURCE)
        os.execv(sys.executable, [sys.executable, str(script), *sys.argv[1:], "--prepared"])
    logging.basicConfig(level=logging.INFO)
    try:
        execute(args)
    except Exception as exc:
        write_json(args.output / "failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
