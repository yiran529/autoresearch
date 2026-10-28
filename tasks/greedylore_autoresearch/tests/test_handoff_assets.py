# ruff: noqa: S101
from __future__ import annotations

import re
from pathlib import Path

import yaml

from coral.agent.dynamic import DynamicAssignment, spec_for_assignment
from coral.config import CoralConfig

TASK_ROOT = Path(__file__).resolve().parents[1]


def test_task_has_two_distinct_codex_roles():
    config = yaml.safe_load((TASK_ROOT / "task.yaml").read_text())
    assignments = config["agents"]["assignments"]
    parsed = CoralConfig.from_dict(config)
    assert parsed.grader.args['staged_research'] is True
    identities = (
        DynamicAssignment('research-N001', 'research', 'N001', 'active'),
        DynamicAssignment('engineer-E0001', 'engineering', 'N001', 'active'),
    )
    assert [spec_for_assignment(parsed, item).runtime_options['role_file'] for item in identities] == [
        './roles/researcher.md', './roles/engineer.md',
    ]
    assert all(item.get("count", 1) == 1 for item in assignments)
    assert config["agents"]["heartbeat_enabled"] is False
    assert config["agents"]["sandbox"]["enabled"] is True
    assert config["agents"]["sandbox"]["provider"] == "srt"
    assert config["agents"]["sandbox"]["network"] == "allowlist"
    domains = config["agents"]["sandbox"]["allowed_domains"]
    assert "api.openai.com" in domains and "chatgpt.com" in domains
    assert config["grader"]["parallel"]["max_workers"] == 1
    setup = config["workspace"]["setup"]
    assert setup and any("autoresearch" in command for command in setup)
    assert all("/home/wyr/greedy_lore/.venv" not in command for command in setup)


def test_required_handoff_assets_exist_and_are_nonempty():
    relative_paths = [
        "roles/workflow_rules.md",
        "roles/researcher.md",
        "roles/engineer.md",
        "prompts/research_analysis.md",
        "references/algorithm_construction.md",
        "references/research_commands.md",
        "templates/research_origin.json",
        "references/engineer_runtime.md",
        "references/semantic_criteria.md",
        "templates/node_spec.md",
        "templates/evaluation.md",
        "templates/research_analysis.md",
        "templates/engineering_analysis.md",
        "templates/submission.yaml",
    ]
    for relative in relative_paths:
        content = (TASK_ROOT / relative).read_text()
        assert len(content.strip()) > 80, relative


def test_reviewer_assets_are_removed():
    for relative in (
        "prompts/reviewer.md",
        "templates/review.md",
        "autoresearch/schemas/review.schema.json",
    ):
        assert not (TASK_ROOT / relative).exists()


def test_formal_roles_and_prompts_are_english():
    paths = [*TASK_ROOT.glob('roles/*.md'), *TASK_ROOT.glob('prompts/*.md')]
    for path in paths:
        assert re.search(r'[\u4e00-\u9fff]', path.read_text()) is None, path


def test_prompts_expose_workflow_not_transport():
    paths = [*TASK_ROOT.glob("roles/*.md"), *TASK_ROOT.glob("prompts/*.md")]
    combined = "\n".join(path.read_text() for path in paths)
    for required in ("autoresearch research register", "autoresearch research train",
                     "autoresearch research finish", "autoresearch engineering timing",
                     "autoresearch engineering finish", "autoresearch diagnostic run"):
        assert required in combined
    for internal in ("trusted.sock", "flock", "grader_runs", "ControlProvider"):
        assert internal not in combined
    assert "--attempt" not in combined
    assert "--focus-attempt" not in combined


def test_semantic_criteria_are_preserved():
    semantic = (TASK_ROOT / "references/semantic_criteria.md").read_text()
    assert "Prompt 软约束" in semantic
    assert "Researcher" in semantic and "Engineer" in semantic
