#!/usr/bin/env python3
"""Copy a pinned, clean GreedyLoRE checkout into a CORAL task seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

AGENT_GUIDANCE_DESTINATION = Path("docs/greedylore-agent-guidance.md")


def git(repository: Path, *arguments: str) -> str:
    """Return stdout from a non-interactive git command."""
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_seed(
    source: Path,
    destination: Path,
    expected_sha: str,
    *,
    coral_sha: str,
) -> dict[str, Any]:
    """Copy only tracked files after verifying the source commit and cleanliness."""
    source = source.resolve()
    destination = destination.resolve()

    head = git(source, "rev-parse", "HEAD").strip()
    if head != expected_sha:
        raise RuntimeError(f"source HEAD {head} != pinned {expected_sha}")
    if git(source, "status", "--porcelain").strip():
        raise RuntimeError("source repository is dirty")
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"destination is not empty: {destination}")

    tracked = [
        relative
        for relative in git(source, "ls-files", "-z").split("\0")
        if relative
    ]
    destination.mkdir(parents=True, exist_ok=True)

    files: dict[str, dict[str, str]] = {}
    relocations: dict[str, str] = {}
    for relative in tracked:
        seed_relative = (
            AGENT_GUIDANCE_DESTINATION
            if relative == "AGENTS.md"
            else Path(relative)
        )
        if relative == "AGENTS.md":
            relocations[relative] = seed_relative.as_posix()

        source_path = source / relative
        destination_path = destination / seed_relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        files[relative] = {
            "seed_path": seed_relative.as_posix(),
            "sha256": _sha256(source_path),
        }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_repository": str(source),
        "source_commit": head,
        "locked_coral_commit": coral_sha,
        "relocations": relocations,
        "files": files,
    }
    (destination / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--coral-sha", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    manifest = copy_seed(
        arguments.source,
        arguments.destination,
        arguments.expected_sha,
        coral_sha=arguments.coral_sha,
    )
    print(f"copied={len(manifest['files'])}")


if __name__ == "__main__":
    main()
