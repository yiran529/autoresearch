"""Small task CLI exposing only user-facing workflow commands."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from autoresearch.client import request_trusted
from autoresearch.evaluation import refresh_node_evaluation
from autoresearch.nodes import NodeError, create_node
from coral.hub._island import island_root
from coral.hub.attempts import read_attempts
from coral.workspace.breadcrumbs import find_coral_breadcrumb, read_island_breadcrumb


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


def _create_node(args: argparse.Namespace) -> dict:
    coral_dir, notes_root, island_id = _workflow_context()
    attempts = [
        attempt
        for attempt in read_attempts(coral_dir, island_id=island_id)
        if attempt.metadata.get("mode") == "research"
        and attempt.metadata.get("node_id") == args.parent
        and attempt.status not in {"pending", "crashed", "timeout"}
    ]
    commits = {attempt.commit_hash for attempt in attempts}
    if len(commits) != 1:
        raise NodeError(f"parent {args.parent} must have exactly one terminal Research SHA")
    parent_commit = next(iter(commits))
    version = args.analysis_version
    evidence = {
        "notes_root": notes_root,
        "parent_node": args.parent,
        "parent_commit": parent_commit,
        "base_commit": parent_commit,
        "analysis_ref": f"nodes/{args.parent}/research_analysis_v{version}.md",
        "review_ref": f"nodes/{args.parent}/research_review_v{version}.md",
        "publish": lambda kind, text, context: _publish(coral_dir, kind, text, context),
    }
    path = create_node(
        {
            "hypothesis": args.hypothesis,
            "selection_rationale": args.selection_rationale,
            "delta": args.delta,
            "risk": args.risk,
            "falsification_criteria": args.falsification_criteria,
        },
        evidence,
    )
    return {"path": str(path.relative_to(notes_root))}


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

    retry = commands.add_parser("retry", help="retry one same-SHA infrastructure failure")
    retry.add_argument("commit_hash")

    node = commands.add_parser("node", help="manage Research Tree Nodes")
    node_commands = node.add_subparsers(dest="node_command", required=True)
    create = node_commands.add_parser(
        "create", help="create a child from an accepted Research Review"
    )
    create.add_argument("--parent", required=True)
    create.add_argument("--analysis-version", type=int, required=True)
    create.add_argument("--hypothesis", required=True)
    create.add_argument("--selection-rationale", required=True)
    create.add_argument("--delta", required=True)
    create.add_argument("--risk", required=True)
    create.add_argument("--falsification-criteria", required=True)

    refresh = commands.add_parser(
        "refresh-evaluation", help="rebuild and publish one Node evaluation"
    )
    refresh.add_argument("--node", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "retry":
        result = request_trusted(_coral_dir(), "retry_attempt", {"commit_hash": args.commit_hash})
    elif args.command == "node" and args.node_command == "create":
        result = _create_node(args)
    elif args.command == "refresh-evaluation":
        result = _refresh_evaluation(args)
    else:
        raise SystemExit("unsupported command")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
