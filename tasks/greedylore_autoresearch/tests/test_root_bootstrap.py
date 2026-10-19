from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from autoresearch.bootstrap import BootstrapError, bootstrap_root


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _fixture(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "model.py").write_text("baseline = True\n")
    _git(repo, "add", "model.py")
    _git(repo, "commit", "-m", "S0")
    source = _git(repo, "rev-parse", "HEAD")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_commit": source,
                "files": {
                    "model.py": {
                        "sha256": hashlib.sha256((repo / "model.py").read_bytes()).hexdigest()
                    }
                },
            }
        )
    )
    return repo, manifest, source


def test_root_bootstrap_is_idempotent_and_only_adds_submission(tmp_path):
    repo, manifest, source = _fixture(tmp_path)
    first = bootstrap_root(repo, manifest)
    second = bootstrap_root(repo, manifest)

    assert second == first
    assert first == repo / "submission.yaml"
    assert _git(repo, "diff", "--name-only", source, "HEAD").splitlines() == ["submission.yaml"]
    assert yaml.safe_load((repo / "submission.yaml").read_text())["node_id"] == "N000"


def test_root_bootstrap_rejects_manifest_mismatch(tmp_path):
    repo, manifest, _ = _fixture(tmp_path)
    (repo / "model.py").write_text("changed = True\n")
    with pytest.raises(BootstrapError, match="manifest"):
        bootstrap_root(repo, manifest)


def test_root_bootstrap_publishes_canonical_n000_spec(tmp_path):
    repo, manifest_path, source = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    published = []
    manifest["publish"] = lambda kind, content, context: (
        published.append((kind, yaml.safe_load(content), context)) or {"path": "nodes/N000/spec.md"}
    )

    bootstrap_root(repo, manifest)

    assert published[0][0] == "spec"
    assert published[0][1]["node_id"] == "N000"
    assert published[0][1]["parent_node"] is None
    assert published[0][1]["base_commit"] == source


def test_root_uses_run_seed_commit_not_upstream_source_sha(tmp_path):
    repo, manifest_path, seed_commit = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["source_commit"] = "f" * 40

    bootstrap_root(repo, manifest)

    assert yaml.safe_load((repo / "submission.yaml").read_text())["base_commit"] == seed_commit


def test_root_retry_publishes_missing_spec_after_a0_exists(tmp_path):
    repo, manifest_path, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    calls = []

    def publish(kind, content, context):
        calls.append((kind, content, context))
        if len(calls) == 1:
            raise RuntimeError("publication interrupted")
        return {"path": "nodes/N000/spec.md"}

    manifest["publish"] = publish
    with pytest.raises(RuntimeError, match="interrupted"):
        bootstrap_root(repo, manifest)

    path = bootstrap_root(repo, manifest)

    assert path == repo / "submission.yaml"
    assert len(calls) == 2


def test_checked_in_source_manifest_matches_seed():
    from autoresearch.bootstrap import _validate_manifest

    task_root = Path(__file__).resolve().parents[1]
    manifest = json.loads((task_root / "seed" / "source_manifest.json").read_text())

    _validate_manifest(task_root / "seed", manifest)


def test_manifest_checks_task_added_checkpoint_helper(tmp_path):
    from autoresearch.bootstrap import _validate_manifest

    repo, manifest_path, _ = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest['task_files'] = {'diagnostic_checkpoint.py': {
        'sha256': hashlib.sha256(b'original helper').hexdigest()}}
    (repo / 'diagnostic_checkpoint.py').write_bytes(b'tampered helper')
    with pytest.raises(BootstrapError, match='diagnostic_checkpoint.py'):
        _validate_manifest(repo, manifest)
