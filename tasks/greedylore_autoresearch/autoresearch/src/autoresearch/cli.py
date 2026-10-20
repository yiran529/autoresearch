"""Small task CLI exposing only user-facing workflow commands."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import uuid
from collections.abc import Sequence
from pathlib import Path

from autoresearch.client import request_trusted
from autoresearch.evaluation import refresh_node_evaluation
from coral.grader.timeouts import DIAGNOSTIC_CLIENT_TIMEOUT, DIAGNOSTIC_EXECUTION_TIMEOUT
from coral.hub._island import island_root
from coral.hub.attempts import read_attempts
from coral.workspace.breadcrumbs import (
    find_breadcrumb_file,
    find_coral_breadcrumb,
    read_island_breadcrumb,
)


def _workflow_context() -> tuple[Path, Path, str | None]:
    found = find_coral_breadcrumb(Path.cwd().resolve())
    if found is None:
        raise SystemExit("No CORAL run found from the current directory")
    coral_dir, breadcrumb_dir = found
    island_id = read_island_breadcrumb(coral_dir, breadcrumb_dir)
    root = island_root(coral_dir, island_id)
    return coral_dir, root / "notes", island_id


def _coral_dir() -> Path:
    return _workflow_context()[0]


def _publish(coral_dir: Path, kind: str, content: str, context: dict) -> dict:
    return request_trusted(
        coral_dir,
        "publish_artifact",
        {
            "kind": kind,
            "node_id": context.get("node_id"),
            "mode": context.get("mode"),
            "version": context.get("version"),
            "content": content,
        },
    )


def _agent_id() -> str:
    breadcrumb = find_breadcrumb_file(".coral_agent_id")
    if breadcrumb is None:
        raise SystemExit("No CORAL Agent identity found in this Worktree")
    return breadcrumb.read_text(encoding="utf-8").strip()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True,
    ).stdout.strip()


def _require_clean_worktree() -> None:
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("Worktree must be clean before changing its Git base")


def _checkout_base(commit_sha: str) -> None:
    _require_clean_worktree()
    branch = f"autoresearch-{_agent_id()}"
    current = _git("branch", "--show-current")
    if current == branch:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit_sha, "HEAD"],
            check=True, capture_output=True,
        )
        return
    exists = subprocess.run(
        ["git", "show-ref", "--verify", f"refs/heads/{branch}"],
        capture_output=True, check=False,
    ).returncode == 0
    if exists:
        _git("switch", branch)
    else:
        _git("switch", "-c", branch, commit_sha)
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit_sha, "HEAD"],
        check=True, capture_output=True,
    )


def _initialize_node(args: argparse.Namespace) -> dict:
    agent_id = _agent_id()
    if not agent_id.startswith("research-N") or len(agent_id) != len("research-N001"):
        raise SystemExit("node initialize requires a reserved Researcher Agent")
    _require_clean_worktree()
    result = request_trusted(
        _coral_dir(), "initialize_research",
        {
            "node_id": agent_id.removeprefix("research-"),
            "parent_node": args.parent,
            "hypothesis": args.hypothesis,
            "selection_rationale": args.selection_rationale,
            "delta": args.delta,
            "risk": args.risk,
            "falsification_criteria": args.falsification_criteria,
        },
    )
    _checkout_base(result["base_commit"])
    return result


def _claim_engineering(node_id: str) -> dict:
    _require_clean_worktree()
    result = request_trusted(_coral_dir(), "claim_engineering", {"node_id": node_id})
    _checkout_base(result["research_sha"])
    return result


def _refresh_evaluation(args: argparse.Namespace) -> dict:
    coral_dir, notes_root, island_id = _workflow_context()
    root = island_root(coral_dir, island_id)
    content = refresh_node_evaluation(
        args.node,
        read_attempts(coral_dir, island_id=island_id),
        root / "eval_logs",
    )
    result = _publish(
        coral_dir,
        "evaluation",
        content,
        {"node_id": args.node, "mode": None, "version": None},
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoresearch")
    commands = parser.add_subparsers(dest="command", required=True)

    evidence = commands.add_parser('evidence', help='archive small supporting scripts and logs')
    evidence_commands = evidence.add_subparsers(dest='evidence_command', required=True)
    archive = evidence_commands.add_parser('archive')
    archive.add_argument('--node', required=True)
    archive.add_argument('--request-id', default=None, help='reuse this ID after a channel timeout')
    archive.add_argument('--files', type=Path, nargs='+', required=True,
                         help='regular files, unique basenames; at most 32 KiB total')

    diagnostic = commands.add_parser("diagnostic", help="run a bounded Analyzer or Engineer diagnostic")
    diagnostic_commands = diagnostic.add_subparsers(dest="diagnostic_command", required=True)
    diagnostic_run = diagnostic_commands.add_parser("run")
    diagnostic_run.add_argument("--gpus", type=int, default=1)
    diagnostic_run.add_argument("--timeout", type=float, default=DIAGNOSTIC_EXECUTION_TIMEOUT)
    diagnostic_run.add_argument("--request-id", default=None, help="reuse this ID after a channel timeout")
    diagnostic_run.add_argument("--scripts-dir", type=Path, help="Engineer diagnostic scripts in your Worktree or /tmp/<agent-id>/")
    diagnostic_run.add_argument("argv", nargs=argparse.REMAINDER)

    retry = commands.add_parser("retry", help="retry one same-SHA infrastructure failure")
    retry.add_argument("commit_hash")

    root = commands.add_parser(
        "bootstrap-root-analysis", help="publish fixed N000 Analysis from the host"
    )
    root.add_argument("--coral-dir", type=Path, required=True)
    root.add_argument("--root-worktree", type=Path, required=True)

    memory = commands.add_parser("consolidate-memory", help="run the one-shot Global Memory gate")
    memory.add_argument("--if-due", action="store_true")
    memory.add_argument("--async", dest="async_launch", action="store_true")
    memory.add_argument("--coral-dir", type=Path)

    node = commands.add_parser("node", help="manage Research Tree Nodes")
    node_commands = node.add_subparsers(dest="node_command", required=True)
    initialize = node_commands.add_parser("initialize", help="initialize this reserved Node")
    initialize.add_argument("--parent", required=True)
    initialize.add_argument("--hypothesis", required=True)
    initialize.add_argument("--selection-rationale", required=True)
    initialize.add_argument("--delta", required=True)
    initialize.add_argument("--risk", required=True)
    initialize.add_argument("--falsification-criteria", required=True)
    complete = node_commands.add_parser("complete", help="complete this Node assignment")
    complete.add_argument("--mode", choices=("research", "engineering"), required=True)
    node_commands.add_parser("cancel", help="cancel an unstarted assignment")

    engineering = commands.add_parser("engineering", help="manage Engineering claims")
    engineering_commands = engineering.add_subparsers(dest="engineering_command", required=True)
    claim = engineering_commands.add_parser("claim", help="atomically claim one Research Node")
    claim.add_argument("node_id")
    engineering_commands.add_parser("cancel", help="cancel an unbound Engineer assignment")
    finish = engineering_commands.add_parser("finish", help="submit your final Engineering Analysis")
    finish.add_argument("--analysis-file", type=Path, required=True,
                        help="scratch JSON with summary, findings, limitations and all attempt_shas")

    refresh = commands.add_parser(
        "refresh-evaluation", help="rebuild and publish one Node evaluation"
    )
    refresh.add_argument("--node", required=True)

    analyze_parser = commands.add_parser(
        "analyze", help="publish a fixed-evidence Research Analysis"
    )
    analyze_parser.add_argument("--node", required=True)
    analyze_parser.add_argument("--mode", choices=("research",), required=True)
    analyze_parser.add_argument("--revise", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == 'evidence':
        from autoresearch.attachments import MAX_ATTACHMENT_BYTES

        files, total = [], 0
        for path in args.files:
            if path.is_symlink() or not path.is_file():
                raise SystemExit('evidence files must be regular files, not symlinks')
            with path.open('rb') as stream:
                content = stream.read(MAX_ATTACHMENT_BYTES + 1)
            total += len(content)
            if total > MAX_ATTACHMENT_BYTES:
                raise SystemExit('archive exceeds 32 KiB; use managed diagnostics for larger outputs')
            files.append({'name': path.name, 'content': base64.b64encode(content).decode('ascii')})
        result = request_trusted(_coral_dir(), 'archive_evidence', {
            'node_id': args.node, 'request_id': args.request_id or uuid.uuid4().hex,
            'commit_sha': _git('rev-parse', 'HEAD'), 'files': files,
        })
    elif args.command == "diagnostic":
        from autoresearch.diagnostics import _validate_request

        run = os.environ.get("CORAL_DIAGNOSTIC_RUN")
        argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        payload = {"request_id": args.request_id or uuid.uuid4().hex,
                   "argv": argv, "gpus": args.gpus, "timeout": args.timeout}
        if args.scripts_dir is not None:
            payload["scripts_dir"] = str(args.scripts_dir.absolute())
        _validate_request(payload)
        result = request_trusted(Path(run) if run else _coral_dir(), "run_diagnostic", payload,
                                 timeout=DIAGNOSTIC_CLIENT_TIMEOUT)
    elif args.command == "retry":
        result = request_trusted(_coral_dir(), "retry_attempt", {"commit_hash": args.commit_hash})
    elif args.command == "bootstrap-root-analysis":
        from autoresearch.bootstrap import bootstrap_root_analysis

        path = bootstrap_root_analysis(args.coral_dir, args.root_worktree)
        result = {"analysis_path": str(path)}
    elif args.command == "consolidate-memory":
        from autoresearch.memory import MemoryContext, consolidate_memory, memory_due

        coral_dir = args.coral_dir or _coral_dir()
        context = MemoryContext(coral_dir)
        if args.async_launch:
            if args.coral_dir is not None:
                raise SystemExit("async Memory launch must use the current Agent run")
            if args.if_due and not memory_due(context.notes_root).due:
                result = {"status": "not_due"}
            else:
                result = request_trusted(coral_dir, "start_memory_if_due", {})
        else:
            memory_result = consolidate_memory(if_due=args.if_due, context=context)
            result = {"status": memory_result.status}
    elif args.command == "node" and args.node_command == "initialize":
        result = _initialize_node(args)
    elif args.command == "node" and args.node_command == "complete":
        result = request_trusted(_coral_dir(), "complete_assignment", {"mode": args.mode})
    elif args.command == "node" and args.node_command == "cancel":
        result = request_trusted(_coral_dir(), "cancel_assignment", {})
    elif args.command == "engineering" and args.engineering_command == "claim":
        result = _claim_engineering(args.node_id)
    elif args.command == "engineering" and args.engineering_command == "cancel":
        result = request_trusted(_coral_dir(), "cancel_assignment", {})
    elif args.command == "engineering" and args.engineering_command == "finish":
        from autoresearch.validation import load_document

        result = request_trusted(_coral_dir(), "finish_assignment", {
            "mode": "engineering", "analysis": load_document(args.analysis_file),
        })
    elif args.command == "refresh-evaluation":
        result = _refresh_evaluation(args)
    elif args.command == "analyze":
        from autoresearch.analysis import analyze

        analysis = analyze(args.node, args.mode, revise=args.revise)
        result = {
            "analysis_version": analysis.analysis_version,
            "analysis_path": str(analysis.analysis_path),
        }
    else:
        raise SystemExit("unsupported command")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
