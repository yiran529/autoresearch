"""ARC is a second seed of the same fixed protocol, with frozen run identity."""

# ruff: noqa: S101
import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from autoresearch.bootstrap import _root_spec
from greedylore_autoresearch_grader.contract import EvalContract, canonical_contract_hash
from greedylore_autoresearch_grader.efficiency import build_timing_command
from greedylore_autoresearch_grader.quality import build_quality_command

TASK = Path(__file__).resolve().parents[1]
ARC = dict(
    name="group_topk",
    r=4,
    compress_ratio=0.2,
    use_error_feedback="ef14",
    error_decay=1.0,
    quality_start_step=1000,
    vision_start_step=4000,
    timing_period_steps=None,
)
SEED = dict(
    version=1,
    id="arctopk",
    upstream_repository="ARC-TopK-release",
    upstream_commit="c09d558" + "0" * 33,
    imported_files={"comm_hooks/group_topk_hook_no_reshape.py": "a" * 64},
    root_hypothesis="Provided ARC-TopK r4/EF14 baseline.",
)


def arc_contract(tmp_path):
    raw = yaml.safe_load((TASK / "taskdata/eval_contract.yaml").read_text())
    raw.update(schema_version=5, seed=copy.deepcopy(SEED), compressor=copy.deepcopy(ARC))
    for field in ("c4_path", "tokenizer_path"):
        path = tmp_path / field
        path.mkdir(parents=True, exist_ok=True)
        raw["data"][field] = str(path)
    raw["vision"]["data_path"] = str(tmp_path)
    path = tmp_path / "eval_contract.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_arc_contract_drives_all_three_commands(tmp_path):
    contract = EvalContract.load(arc_contract(tmp_path))
    assert contract.contract_hash == canonical_contract_hash(contract.path)
    profiles = [contract.quality["llama130m_muon"], contract.quality["resnet18_cifar100_muon"]]
    commands = [
        build_quality_command(p, tmp_path, tmp_path / p["name"], range(4)).command for p in profiles
    ]
    cell = next(iter(contract.efficiency.cells.values()))
    commands.append(build_timing_command(cell, 0, tmp_path, tmp_path / "timing", range(4)).command)
    for command in commands:
        assert "--r" in command and command[command.index("--r") + 1] == "4"
        assert "--compress_rank" not in command and "--compress-rank" not in command
        assert command[command.index("--compressor") + 1] == "group_topk"
    assert cell["model"] == "llama350m" and cell["measured_iterations"] == 400


def test_arc_root_spec_uses_recorded_identity():
    frozen = copy.deepcopy(SEED)
    assert _root_spec("b" * 40, frozen)["hypothesis"] == "Provided ARC-TopK r4/EF14 baseline."
    assert _root_spec("b" * 40)["hypothesis"] == "Provided GreedyLoRE r32/EF14 baseline."


def test_saved_v4_schema_and_hash_remain_readable(tmp_path):
    raw = yaml.safe_load((TASK / "taskdata/eval_contract_v4.yaml").read_text())
    for key in ("c4_path", "tokenizer_path"):
        raw["data"][key] = str(tmp_path)
    raw["vision"]["data_path"] = str(tmp_path)
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert EvalContract.load(path).contract_hash == canonical_contract_hash(path)


@pytest.mark.parametrize(
    "patch",
    [
        dict(lr=99),
        dict(r=0),
        dict(compress_ratio=1.2),
        dict(r=True),
        dict(error_decay=float("nan")),
    ],
)
def test_arc_method_rejects_invalid_or_training_fields(tmp_path, patch):
    path = arc_contract(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["compressor"].update(patch)
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        EvalContract.load(path)


def test_arc_bootstrap_and_recovery_bind_private_contract(tmp_path):
    import subprocess

    from autoresearch.bootstrap import bootstrap_root, run_seed

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "t@example.com")
    (repo / "model.py").write_text("baseline = True\n")
    git("add", ".")
    git("commit", "-qm", "seed")
    s0 = git("rev-parse", "HEAD")
    coral = tmp_path / ".coral"
    private = coral / "private"
    private.mkdir(parents=True)
    arc_contract(private)
    manifest = {
        "source_commit": "f" * 40,
        "files": {
            "model.py": {"sha256": hashlib.sha256((repo / "model.py").read_bytes()).hexdigest()}
        },
        "imports": {
            "arctopk": {"source_commit": SEED["upstream_commit"], "files": SEED["imported_files"]}
        },
    }
    published = []
    manifest["publish"] = lambda kind, content, context: (
        published.append(yaml.safe_load(content)) or {"path": "nodes/N000/spec.md"}
    )
    bootstrap_root(repo, manifest, seed=run_seed(coral))
    # Recovery reads the run-private snapshot even when the launch/default task is gone.
    bootstrap_root(repo, manifest, seed=run_seed(coral))
    assert published[0] == published[1] == _root_spec(s0, SEED)
    submission = yaml.safe_load((repo / "submission.yaml").read_text())
    assert (
        "ARC-TopK" in submission["change_summary"] and submission["timing"]["period_steps"] is None
    )


def test_real_arc_cpu_gloo_checkpoint_and_entrypoints(tmp_path):
    import os
    import subprocess

    contract = EvalContract.load(arc_contract(tmp_path))
    specs = [
        build_quality_command(contract.quality[name], TASK / "seed", tmp_path / name, range(4))
        for name in ("llama130m_muon", "resnet18_cifar100_muon")
    ]
    specs.append(
        build_timing_command(
            next(iter(contract.efficiency.cells.values())),
            0,
            TASK / "seed",
            tmp_path / "timing",
            range(4),
        )
    )
    greedy_raw = yaml.safe_load((TASK / "taskdata/greedylore/eval_contract.yaml").read_text())
    greedy_raw["data"] = dict(contract.raw["data"])
    greedy_raw["vision"]["data_path"] = str(tmp_path)
    greedy_path = tmp_path / "greedy.yaml"
    greedy_path.write_text(yaml.safe_dump(greedy_raw))
    greedy = EvalContract.load(greedy_path)
    specs.append(
        build_quality_command(
            greedy.quality["resnet18_cifar100_muon"],
            TASK / "seed",
            tmp_path / "greedy-vision",
            range(4),
        )
    )
    commands = tmp_path / "commands.json"
    commands.write_text(json.dumps([list(s.command) for s in specs]))
    result = subprocess.run(
        [
            "/home/wyr/greedy_lore/.venv/bin/python",
            str(Path(__file__).with_name("arctopk_cpu_cases.py")),
            str(TASK / "seed"),
            str(tmp_path),
        ],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "GLOO_SOCKET_IFNAME": "lo",
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ARC CPU/Gloo checks passed" in result.stdout


def test_seed_identity_is_declared_not_inferred_from_method_or_repository(tmp_path):
    from autoresearch.bootstrap import run_seed

    private = tmp_path / "private"
    private.mkdir()
    path = arc_contract(private)
    raw = yaml.safe_load(path.read_text())
    raw["seed"].update(
        id="lab_snapshot",
        upstream_repository="https://example.org/lab/source",
        imported_files={"algorithms/compressor.py": "e" * 64},
        root_hypothesis="Provided lab snapshot under the fixed protocol.",
    )
    path.write_text(yaml.safe_dump(raw))
    contract = EvalContract.load(path)
    assert contract.raw["seed"]["id"] == "lab_snapshot"
    assert _root_spec("d" * 40, run_seed(tmp_path))["hypothesis"] == raw["seed"]["root_hypothesis"]


@pytest.mark.parametrize("identity", ["greedylore", "arctopk"])
def test_prepare_entrypoint_freezes_real_config_and_root(tmp_path, monkeypatch, identity):
    import importlib.util

    from autoresearch.bootstrap import run_seed

    source = TASK / "scripts/prepare_seed_run.py"
    assert source.is_file(), "seed launch must have an executable entrypoint"
    spec = importlib.util.spec_from_file_location("prepare_seed_run", source)
    entrypoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entrypoint)
    # Only dependency installation is replaced: use the real project copy,
    # config/private contract persistence, bootstrap and trusted submission APIs.
    import os
    import sys

    from simulation_support import offline_grader_env

    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(sys.path))
    monkeypatch.setattr("coral.workspace.grader_env.setup_grader_env", offline_grader_env)
    paths = entrypoint.prepare_run(identity, tmp_path / identity)
    frozen = paths.coral_dir / "private/eval_contract.yaml"
    before = frozen.read_bytes()
    assert yaml.safe_load(before)["schema_version"] == 5
    saved = yaml.safe_load((paths.coral_dir / "config.yaml").read_text())
    assert saved["grader"]["args"]["staged_research"] is True
    assert (run_seed(paths.coral_dir) or {}).get("id", "greedylore") == identity
    root = yaml.safe_load((paths.coral_dir / "public/notes/nodes/N000/spec.md").read_text())
    assert ("ARC-TopK" if identity == "arctopk" else "GreedyLoRE") in root["hypothesis"]
    from coral.hub.attempts import read_attempts

    attempts = read_attempts(paths.coral_dir)
    assert len(attempts) == 1
    from autoresearch.state import WorkflowStore

    store = WorkflowStore(paths.coral_dir, repo_path=paths.repo_dir)
    assert store._engineering_design_source("N000", attempts[0].commit_hash)["iteration"] is None
    # Changing the current task location/defaults cannot change this run's Root.
    monkeypatch.setattr(entrypoint, "TASK", tmp_path / "removed-current-defaults")

    from coral.cli._helpers import find_coral_dir

    monkeypatch.chdir(paths.run_dir)
    assert find_coral_dir() == paths.coral_dir
    recovered = entrypoint.recover_run(tmp_path / identity)
    assert recovered.coral_dir == paths.coral_dir and frozen.read_bytes() == before
    assert (paths.repo_dir / "comm_hooks/group_topk_hook_no_reshape.py").is_file()


def test_arc_dense_reference_disables_only_the_method(tmp_path):
    contract = EvalContract.load(arc_contract(tmp_path))
    profile = dict(contract.quality["llama130m_muon"])
    profile["compressor"] = {**profile["compressor"], "name": "none"}
    command = build_quality_command(profile, tmp_path, tmp_path / "dense", range(4)).command
    assert command[command.index("--compressor") + 1] == "none"
    assert "--r" not in command and "--compress_rank" not in command


@pytest.mark.parametrize(
    "field,value",
    [("rank", 0), ("rank", True), ("update_gap", 0), ("beta_ef", float("nan")), ("ef_rank", 17)],
)
def test_new_greedy_contract_rejects_invalid_method_values(tmp_path, field, value):
    raw = yaml.safe_load((TASK / "taskdata/greedylore/eval_contract.yaml").read_text())
    raw["data"]["c4_path"] = raw["data"]["tokenizer_path"] = raw["vision"]["data_path"] = str(
        tmp_path
    )
    raw["compressor"][field] = value
    path = tmp_path / "bad-greedy.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="compression"):
        EvalContract.load(path)
