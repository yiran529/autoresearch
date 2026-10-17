"""Run an optional bounded startup command, then exec the real Agent runtime."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Sequence

logger = logging.getLogger(__name__)


def run_startup_hook(hook: Sequence[str], runtime: Sequence[str]) -> None:
    if not hook or not all(isinstance(item, str) and item for item in hook):
        raise ValueError("startup Hook must be a nonempty argv list")
    if not runtime or not all(isinstance(item, str) and item for item in runtime):
        raise ValueError("runtime command must be a nonempty argv list")
    try:
        subprocess.run(list(hook), check=False)
    except OSError:
        logger.exception("startup Hook could not run; continuing to Agent runtime")
    os.execvpe(runtime[0], list(runtime), os.environ)


def main(argv: Sequence[str] | None = None) -> None:
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) < 4 or values[0] != "--hook-json" or "--" not in values:
        raise SystemExit("usage: startup_hook --hook-json JSON -- runtime argv...")
    separator = values.index("--")
    if separator != 2:
        raise SystemExit("startup Hook accepts exactly one JSON argv option")
    try:
        hook = json.loads(values[1])
    except json.JSONDecodeError as exc:
        raise SystemExit("startup Hook JSON is invalid") from exc
    if not isinstance(hook, list):
        raise SystemExit("startup Hook JSON must be an argv list")
    run_startup_hook(hook, values[separator + 1:])


if __name__ == "__main__":
    main()
