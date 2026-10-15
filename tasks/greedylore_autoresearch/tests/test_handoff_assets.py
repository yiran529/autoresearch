from __future__ import annotations

from pathlib import Path

import yaml


TASK_ROOT = Path(__file__).resolve().parents[1]


def test_task_has_two_distinct_codex_roles():
    config = yaml.safe_load((TASK_ROOT / "task.yaml").read_text())
    assignments = config["agents"]["assignments"]
    assert [item["runtime_options"]["role_file"] for item in assignments] == [
        "./roles/researcher.md",
        "./roles/engineer.md",
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
        "prompts/engineering_analysis.md",
        "prompts/reviewer.md",
        "references/algorithm_ref.md",
        "references/system_ref.md",
        "references/semantic_criteria.md",
        "templates/node_spec.md",
        "templates/evaluation.md",
        "templates/research_analysis.md",
        "templates/engineering_analysis.md",
        "templates/review.md",
        "templates/submission.yaml",
    ]
    for relative in relative_paths:
        content = (TASK_ROOT / relative).read_text()
        assert len(content.strip()) > 80, relative


def test_prompts_expose_workflow_not_transport():
    paths = [*TASK_ROOT.glob("roles/*.md"), *TASK_ROOT.glob("prompts/*.md")]
    combined = "\n".join(path.read_text() for path in paths)
    for required in ("coral eval", "autoresearch retry", "autoresearch analyze"):
        assert required in combined
    for internal in ("trusted.sock", "flock", "grader_runs", "ControlProvider"):
        assert internal not in combined
    assert "--attempt" not in combined
    assert "--focus-attempt" not in combined


def test_reference_toolkit_and_semantic_criteria_are_preserved():
    algorithm = (TASK_ROOT / "references/algorithm_ref.md").read_text()
    system = (TASK_ROOT / "references/system_ref.md").read_text()
    semantic = (TASK_ROOT / "references/semantic_criteria.md").read_text()
    for term in ("Eckart", "Davis", "Error Feedback"):
        assert term in algorithm
    for term in ("Roofline", "Collective", "Overlap"):
        assert term in system
    assert "Prompt 软约束" in semantic
    assert "Researcher" in semantic and "Engineer" in semantic
