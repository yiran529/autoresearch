"""Small task CLI exposing only user-facing workflow commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoresearch.client import request_trusted
from coral.workspace.breadcrumbs import find_coral_breadcrumb


def _coral_dir() -> Path:
    found = find_coral_breadcrumb(Path.cwd().resolve())
    if found is None:
        raise SystemExit("No CORAL run found from the current directory")
    return found[0]


def main() -> None:
    parser = argparse.ArgumentParser(prog="autoresearch")
    commands = parser.add_subparsers(dest="command", required=True)
    retry = commands.add_parser("retry", help="retry one same-SHA infrastructure failure")
    retry.add_argument("commit_hash")
    args = parser.parse_args()
    if args.command == "retry":
        result = request_trusted(
            _coral_dir(), "retry_attempt", {"commit_hash": args.commit_hash}
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
