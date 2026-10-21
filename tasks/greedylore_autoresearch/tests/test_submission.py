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


def test_committed_timing_request_is_canonical_metadata_not_a_timer_override(tmp_path):
    repo, base = _repo(tmp_path)
    declaration = _submission(base)
    declaration["timing"] = {"period_steps": 800, "requested_steps": 10000}
    (repo / "submission.yaml").write_text(yaml.safe_dump(declaration))
    _git(repo, "add", "submission.yaml")
    _git(repo, "commit", "-m", "request long timing")
    coral_dir = tmp_path / ".coral"
    _write_node_spec(coral_dir, base)
    result = validate_submission(_git(repo, "rev-parse", "HEAD"), {
        "worktree": str(repo), "coral_dir": str(coral_dir),
        "contract_path": str(_contract(tmp_path)), "parent_hash": base,
    })
    assert result["timing_declaration"] == {"period_steps": 800, "requested_steps": 10000}
    assert "measured_iterations" not in result


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


def test_validator_requires_existing_fixed_analysis_refs(tmp_path):
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


def test_validator_accepts_fixed_parent_analysis_without_review(tmp_path):
    repo, base = _repo(tmp_path)
    declaration = _submission(base)
    reference = "nodes/N000/research_analysis_v1.md"
    declaration["analysis_refs"] = {
        "parent_research": reference,
        "parent_engineering": None,
    }
    (repo / "submission.yaml").write_text(yaml.safe_dump(declaration, sort_keys=False))
    _git(repo, "add", "submission.yaml")
    _git(repo, "commit", "-m", "research")
    coral_dir = tmp_path / ".coral"
    _write_node_spec(coral_dir, base, analysis_refs=declaration["analysis_refs"])
    parent = coral_dir / "public" / "notes" / reference
    parent.parent.mkdir(parents=True)
    parent.write_text("fixed Analysis")
    result = validate_submission(
        _git(repo, "rev-parse", "HEAD"),
        {
            "worktree": str(repo),
            "coral_dir": str(coral_dir),
            "contract_path": str(_contract(tmp_path)),
            "parent_hash": base,
            "island_id": None,
        },
    )
    assert result["node_id"] == "N023"


def test_review_artifact_kind_is_removed():
    with pytest.raises(ValueError, match="unsupported artifact kind"):
        validate_artifact(
            "review", "{}", {"node_id": "N001", "mode": "research", "version": 1}
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

    with pytest.raises(SubmissionError, match="frozen terminal Research SHA"):
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


@pytest.mark.parametrize("node_id", ["N023", "N000"])
def test_engineering_provenance_uses_same_node_research_sha_and_allows_commit_chain(tmp_path, node_id):
    repo, base = _repo(tmp_path)
    (repo / "model.py").write_text("research = True\n")
    _git(repo, "add", "model.py")
    _git(repo, "commit", "-m", "research result")
    research_sha = _git(repo, "rev-parse", "HEAD")
    coral_dir = tmp_path / ".coral"
    _write_node_spec(coral_dir, base)
    if node_id == "N000":
        node = coral_dir / "public/notes/nodes/N023"
        spec = yaml.safe_load((node / "spec.md").read_text())
        spec["node_id"] = node_id
        (node / "spec.md").write_text(yaml.safe_dump(spec))
        node.rename(node.with_name(node_id))
    attempts = coral_dir / "public" / "attempts"
    attempts.mkdir(parents=True)
    (attempts / f"{research_sha}.json").write_text(json.dumps({
        "commit_hash": research_sha,
        "agent_id": "researcher", "title": "research", "timestamp": "2026-01-01",
        "status": "improved",
        "metadata": {"mode": "research", "node_id": node_id},
    }))
    declaration = _submission(research_sha)
    declaration.update(mode="engineering", quality_anchor=research_sha)
    declaration.update(node_id=node_id, node_type="seed" if node_id == "N000" else "research")
    (repo / "submission.yaml").write_text(yaml.safe_dump(declaration, sort_keys=False))
    _git(repo, "add", "submission.yaml")
    _git(repo, "commit", "-m", "first engineering candidate")
    first = _git(repo, "rev-parse", "HEAD")
    assert validate_submission(first, {
        "worktree": str(repo), "coral_dir": str(coral_dir),
        "contract_path": str(_contract(tmp_path)), "parent_hash": research_sha,
        "island_id": None,
    })["quality_anchor"] == research_sha

    (repo / "model.py").write_text("research = True\noptimized = True\n")
    _git(repo, "add", "model.py")
    _git(repo, "commit", "-m", "second engineering candidate")
    second = _git(repo, "rev-parse", "HEAD")
    assert validate_submission(second, {
        "worktree": str(repo), "coral_dir": str(coral_dir),
        "contract_path": str(_contract(tmp_path)), "parent_hash": first,
        "island_id": None,
    })["base_commit"] == research_sha


def test_engineering_rejects_spec_base_instead_of_research_sha(tmp_path):
    repo, base = _repo(tmp_path)
    (repo / "model.py").write_text("research = True\n")
    _git(repo, "add", "model.py")
    _git(repo, "commit", "-m", "research result")
    research_sha = _git(repo, "rev-parse", "HEAD")
    declaration = _submission(base)
    declaration.update(mode="engineering", quality_anchor=research_sha)
    (repo / "submission.yaml").write_text(yaml.safe_dump(declaration, sort_keys=False))
    _git(repo, "add", "submission.yaml")
    _git(repo, "commit", "-m", "wrong engineering base")
    coral_dir = tmp_path / ".coral"
    _write_node_spec(coral_dir, base)
    attempts = coral_dir / "public" / "attempts"
    attempts.mkdir(parents=True)
    (attempts / f"{research_sha}.json").write_text(json.dumps({
        "commit_hash": research_sha, "status": "improved",
        "metadata": {"mode": "research", "node_id": "N023"},
    }))
    with pytest.raises(SubmissionError, match="Research SHA"):
        validate_submission(_git(repo, "rev-parse", "HEAD"), {
            "worktree": str(repo), "coral_dir": str(coral_dir),
            "contract_path": str(_contract(tmp_path)), "parent_hash": research_sha,
            "island_id": None,
        })


def test_iterative_research_allows_refinement_chain_with_fixed_spec(tmp_path):
    repo, base = _repo(tmp_path)
    coral_dir = tmp_path / '.coral'
    _write_node_spec(coral_dir, base)
    private = coral_dir / 'private'
    private.mkdir()
    (private / 'autoresearch_state.json').write_text(json.dumps({
        'nodes': {'N023': {'research': {'iterative': True}}},
    }))
    declaration = _submission(base)
    (repo / 'submission.yaml').write_text(yaml.safe_dump(declaration))
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-m', 'first candidate')
    first = _git(repo, 'rev-parse', 'HEAD')
    (repo / 'model.py').write_text('refresh_period = 16\n')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-m', 'refine refresh period')
    second = _git(repo, 'rev-parse', 'HEAD')
    context = {'worktree': str(repo), 'coral_dir': str(coral_dir),
               'contract_path': str(_contract(tmp_path)), 'parent_hash': first, 'island_id': None}
    assert validate_submission(second, context)['base_commit'] == base
    declaration['base_commit'] = first
    (repo / 'submission.yaml').write_text(yaml.safe_dump(declaration))
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-m', 'invalid rebase of Node')
    context['parent_hash'] = second
    with pytest.raises(SubmissionError, match='Node spec base_commit'):
        validate_submission(_git(repo, 'rev-parse', 'HEAD'), context)
