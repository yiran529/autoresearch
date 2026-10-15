from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = REPO_ROOT / "tasks" / "greedylore_autoresearch"
SEED_DIR = TASK_ROOT / "seed"
SOURCE_REPO = Path("/home/wyr/greedy_lore")
SOURCE_SHA = "d2c38bc2062e06fdfd73185c508e7cc652933de6"
CORAL_SHA = "0123dfb939b35228cf2c1fde224cd0561e727408"


def _tracked_source_files() -> list[str]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=SOURCE_REPO
    ).decode()
    return [relative for relative in raw.split("\0") if relative]


def test_manifest_covers_exact_tracked_source_files() -> None:
    manifest = json.loads((SEED_DIR / "source_manifest.json").read_text())

    assert manifest["source_commit"] == SOURCE_SHA
    assert manifest["locked_coral_commit"] == CORAL_SHA
    assert sorted(manifest["files"]) == sorted(_tracked_source_files())
    assert not (SEED_DIR / ".git").exists()


def test_manifest_hashes_match_source_and_seed() -> None:
    manifest = json.loads((SEED_DIR / "source_manifest.json").read_text())

    for source_relative, record in manifest["files"].items():
        source_hash = hashlib.sha256(
            (SOURCE_REPO / source_relative).read_bytes()
        ).hexdigest()
        seed_hash = hashlib.sha256(
            (SEED_DIR / record["seed_path"]).read_bytes()
        ).hexdigest()
        assert record["sha256"] == source_hash
        adaptation = manifest["adaptations"].get(source_relative)
        if adaptation is None:
            assert seed_hash == source_hash
        else:
            assert adaptation["seed_sha256"] == seed_hash
            assert adaptation["reason"]

    assert set(manifest["adaptations"]) == {
        "c4/run_llama_pretraining.py",
        "c4/table_v_timing.py",
        "tests/test_c4_model_setup.py",
        "tests/test_table_v_timing.py",
    }
    assert manifest["relocations"] == {
        "AGENTS.md": "docs/greedylore-agent-guidance.md"
    }
    assert not (SEED_DIR / "AGENTS.md").exists()


def test_seed_excludes_untracked_runtime_data() -> None:
    assert len(_tracked_source_files()) == 119
    assert not (SEED_DIR / ".venv").exists()
    assert not (SEED_DIR / "c4" / "c4_en").exists()
    assert not (SEED_DIR / "c4" / "c4_en_table_iv_96").exists()
    assert not list(SEED_DIR.rglob("*.json.gz"))
    assert not list(SEED_DIR.rglob("*.log"))
