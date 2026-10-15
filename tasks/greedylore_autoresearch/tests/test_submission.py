from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "autoresearch" / "src"))

from autoresearch.artifacts import validate_artifact  # noqa: E402
from autoresearch.submission import SubmissionError, validate_submission  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "model.py").write_text("baseline = True\n")
    _git(repo, "add", "model.py")
    _git(repo, "commit", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def _contract(tmp_path: Path) -> Path:
    path = tmp_path / "eval_contract.yaml"
    path.write_text("schema_version: 1\nrecovery:\n  max_infra_retries: 2\n")
    return path


def _submission(base: str) -> dict[str, object]:
    return {
        "mode": "research",
        "node_id": "N023",
        "node_type": "research",
        "base_commit": base,
        "quality_anchor": None,
        "analysis_refs": {"parent_research": None, "parent_engineering": None},
        "change_summary": "first research implementation",
        "selection_rationale": "bounded test",
        "semantic_self_check": {"status": "not_run", "commands": [], "limitations": "unit test"},
    }


def _write_node_spec(
    coral_dir: Path,
    base: str,
    *,
    parent_commit: str | None = None,
    analysis_refs: dict[str, str | None] | None = None,
    review_refs: dict[str, str | None] | None = None,
    island_id: str | None = None,
) -> None:
    if island_id is None:
        notes_root = coral_dir / "public" / "notes"
    else:
        notes_root = coral_dir / "islands" / island_id / "notes"
    node_dir = notes_root / "nodes" / "N023"
    node_dir.mkdir(parents=True)
    spec = {
        "node_id": "N023",
        "base_commit": base,
        "parent_commit": parent_commit or base,
        "analysis_refs": analysis_refs
        or {"parent_research": None, "parent_engineering": None},
        "review_refs": review_refs
        or {"parent_research": None, "parent_engineering": None},
    }
    (node_dir / "spec.md").write_text(yaml.safe_dump(spec, sort_keys=False))


def test_validator_reads_committed_submission_not_worktree(tmp_path):
    repo, base = _repo(tmp_path)
    (repo / "submission.yaml").write_text(yaml.safe_dump(_submission(base), sort_keys=False))
    _git(repo, "add", "submission.yaml")
    _git(repo, "commit", "-m", "research")
    commit = _git(repo, "rev-parse", "HEAD")
    (repo / "submission.yaml").write_text("mode: forged\n")

    coral_dir = tmp_path / ".coral"
    _write_node_spec(coral_dir, base)
    result = validate_submission(
        commit,
        {
            "worktree": str(repo),
            "coral_dir": str(coral_dir),
            "contract_path": str(_contract(tmp_path)),
            "parent_hash": base,
            "island_id": None,
        },
    )

    assert result["mode"] == "research"
    assert result["node_id"] == "N023"
    assert result["max_infra_retries"] == 2
    assert len(result["contract_hash"]) == 64


def test_validator_rejects_unknown_submission_fields(tmp_path):
    repo, base = _repo(tmp_path)
    declaration = _submission(base)
    declaration["path"] = "../../private"
    (repo / "submission.yaml").write_text(yaml.safe_dump(declaration, sort_keys=False))
    _git(repo, "add", "submission.yaml")
    _git(repo, "commit", "-m", "invalid")

    with pytest.raises(SubmissionError, match="schema"):
        validate_submission(
            _git(repo, "rev-parse", "HEAD"),
            {"worktree": str(repo), "contract_path": str(_contract(tmp_path))},
        )


def test_root_submission_diff_only_contains_submission(tmp_path):
    repo, base = _repo(tmp_path)
    declaration = _submission(base)
    declaration["node_id"] = "N000"
    declaration["node_type"] = "seed"
    (repo / "submission.yaml").write_text(yaml.safe_dump(declaration, sort_keys=False))
    (repo / "model.py").write_text("changed = True\n")
    _git(repo, "add", "submission.yaml", "model.py")
    _git(repo, "commit", "-m", "bad root")

    with pytest.raises(SubmissionError, match="Root diff"):
        validate_submission(
            _git(repo, "rev-parse", "HEAD"),
            {"worktree": str(repo), "contract_path": str(_contract(tmp_path))},
        )


def test_artifact_validator_returns_only_canonical_targets():
    result = validate_artifact(
        "analysis",
        "# Analysis\n",
        {"node_id": "N007", "mode": "research", "version": 2},
    )
    assert result["relative_path"] == "nodes/N007/research_analysis_v2.md"

    with pytest.raises(ValueError):
        validate_artifact("analysis", "x", {"node_id": "../../private", "mode": "research", "version": 1})


def test_validator_rejects_node_provenance_mismatch(tmp_path):
    repo, base = _repo(tmp_path)
    declaration = _submission(base)
    (repo / "submission.yaml").write_text(yaml.safe_dump(declaration, sort_keys=False))
    _git(repo, "add", "submission.yaml")
    _git(repo, "commit", "-m", "research")
    coral_dir = tmp_path / ".coral"
    _write_node_spec(coral_dir, "f" * 40)

    with pytest.raises(SubmissionError, match="Node spec base_commit"):
        validate_submission(
            _git(repo, "rev-parse", "HEAD"),
            {
                "worktree": str(repo),
                "coral_dir": str(coral_dir),
                "contract_path": str(_contract(tmp_path)),
                "parent_hash": base,
                "island_id": None,
            },
        )


def test_validator_requires_existing_fixed_analysis_and_review_refs(tmp_path):
    repo, base = _repo(tmp_path)
    declaration = _submission(base)
    declaration["analysis_refs"] = {
        "parent_research": "nodes/N000/research_analysis_v1.md",
        "parent_engineering": None,
    }
    (repo / "submission.yaml").write_text(yaml.safe_dump(declaration, sort_keys=False))
    _git(repo, "add", "submission.yaml")
    _git(repo, "commit", "-m", "research")
    coral_dir = tmp_path / ".coral"
    _write_node_spec(
        coral_dir,
        base,
        analysis_refs=declaration["analysis_refs"],
        review_refs={
            "parent_research": "nodes/N000/research_review_v1.md",
            "parent_engineering": None,
        },
    )

    with pytest.raises(SubmissionError, match="referenced Artifact does not exist"):
        validate_submission(
            _git(repo, "rev-parse", "HEAD"),
            {
                "worktree": str(repo),
                "coral_dir": str(coral_dir),
                "contract_path": str(_contract(tmp_path)),
                "parent_hash": base,
                "island_id": None,
            },
        )


def test_validator_rejects_symlinked_node_spec_parent(tmp_path):
    repo, base = _repo(tmp_path)
    declaration = _submission(base)
    (repo / "submission.yaml").write_text(yaml.safe_dump(declaration, sort_keys=False))
    _git(repo, "add", "submission.yaml")
    _git(repo, "commit", "-m", "research")
    coral_dir = tmp_path / ".coral"
    outside = tmp_path / "outside"
    _write_node_spec(outside, base)
    notes = coral_dir / "public" / "notes"
    (notes / "nodes").mkdir(parents=True)
    (notes / "nodes" / "N023").symlink_to(
        outside / "public" / "notes" / "nodes" / "N023",
        target_is_directory=True,
    )

    with pytest.raises(SubmissionError, match="canonical Node spec"):
        validate_submission(
            _git(repo, "rev-parse", "HEAD"),
            {
                "worktree": str(repo),
                "coral_dir": str(coral_dir),
                "contract_path": str(_contract(tmp_path)),
                "parent_hash": base,
                "island_id": None,
            },
        )


def test_engineering_anchor_is_scoped_to_authenticated_island(tmp_path):
    repo, base = _repo(tmp_path)
    anchor = "a" * 40
    declaration = _submission(base)
    declaration["mode"] = "engineering"
    declaration["quality_anchor"] = anchor
    (repo / "submission.yaml").write_text(yaml.safe_dump(declaration, sort_keys=False))
    _git(repo, "add", "submission.yaml")
    _git(repo, "commit", "-m", "engineering")
    coral_dir = tmp_path / ".coral"
    _write_node_spec(coral_dir, base, island_id="A")
    attempts_b = coral_dir / "islands" / "B" / "attempts"
    attempts_b.mkdir(parents=True)
    (attempts_b / f"{anchor}.json").write_text(
        json.dumps(
            {
                "commit_hash": anchor,
                "status": "improved",
                "metadata": {"mode": "research", "node_id": "N023"},
            }
        )
    )

    with pytest.raises(SubmissionError, match="unique terminal Research SHA"):
        validate_submission(
            _git(repo, "rev-parse", "HEAD"),
            {
                "worktree": str(repo),
                "coral_dir": str(coral_dir),
                "contract_path": str(_contract(tmp_path)),
                "parent_hash": base,
                "island_id": "A",
            },
        )
