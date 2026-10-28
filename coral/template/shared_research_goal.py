"""Compose task-owned research goals with role-specific instructions."""

from __future__ import annotations

from pathlib import Path


def with_shared_research_goal(prompt: str, task_dir: Path | None) -> str:
    """Insert an optional task goal after the prompt title, preserving its body."""
    if task_dir is None:
        return prompt
    try:
        shared_goal = (Path(task_dir) / "prompts/shared_research_goal.md").read_text(
            encoding="utf-8"
        ).strip()
    except FileNotFoundError:
        return prompt
    if not shared_goal:
        return prompt
    if shared_goal.startswith("# "):
        shared_goal = "#" + shared_goal
    title, separator, body = prompt.partition("\n\n")
    if title.startswith("# ") and separator:
        return f"{title}\n\n{shared_goal}\n\n{body}"
    return f"{shared_goal}\n\n{prompt}"
