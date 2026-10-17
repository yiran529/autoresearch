import json
import math

import pytest
from greedylore_autoresearch_grader.runtime import InfraFailure


@pytest.mark.parametrize("key,value", [
    ("pretraining_allow_other_users", "false"),
    ("gpu_wait_timeout_seconds", True),
    ("gpu_wait_timeout_seconds", -1),
    ("gpu_poll_interval_seconds", 0),
    ("timing_monitor_interval_seconds", math.nan),
    ("gpu_lock_dir", "relative/path"),
])
def test_resource_options_reject_ambiguous_or_unbounded_values(key, value):
    from greedylore_autoresearch_grader.resources import ResourceOptions

    with pytest.raises(InfraFailure):
        ResourceOptions.from_args({key: value})


def test_reference_requires_real_compatible_monitoring_evidence(tmp_path):
    from greedylore_autoresearch_grader.resources import (
        ALLOCATION_POLICY_VERSION,
        ResourceOptions,
        validate_monitoring_reference,
    )

    options = ResourceOptions.from_args({})
    reference = {"contract_hash": "abc", "environment": {"gpu_uuids": ["a"]}}
    with pytest.raises(InfraFailure, match="monitoring"):
        validate_monitoring_reference(reference, options, tmp_path)
    reference["monitoring"] = {
        "allocation_policy_version": ALLOCATION_POLICY_VERSION,
        "interval_seconds": 1,
        "evidence_path": "monitor.json",
    }
    evidence = {
        "kind": "legacy_equivalence", "contract_hash": "abc",
        "environment": reference["environment"],
        "allocation_policy_version": ALLOCATION_POLICY_VERSION, "interval_seconds": 1,
        "predeclared": True, "relative_tolerance": 0.01,
        "relative_difference_interval": [-0.005, 0.005],
    }
    (tmp_path / "monitor.json").write_text(json.dumps(evidence))
    validate_monitoring_reference(reference, options, tmp_path)
    evidence["relative_difference_interval"] = [-0.02, 0.02]
    (tmp_path / "monitor.json").write_text(json.dumps(evidence))
    with pytest.raises(InfraFailure, match="equivalence"):
        validate_monitoring_reference(reference, options, tmp_path)
    reference["monitoring"]["evidence_path"] = "../outside.json"
    with pytest.raises(InfraFailure):
        validate_monitoring_reference(reference, options, tmp_path)


def test_resource_journal_throttles_waiting_and_preserves_stage_history(tmp_path):
    from greedylore_autoresearch_grader.resources import ResourceJournal, ResourceOptions

    journal = ResourceJournal(tmp_path, ResourceOptions.from_args({}))
    for _ in range(100):
        journal.event("quality", "waiting", reason="busy")
    journal.event("quality", "reserved", gpu_uuids=["a"])
    journal.event("quality", "released", gpu_uuids=["a"])
    journal.event("timing", "waiting", reason="busy")
    rows = [json.loads(line) for line in (tmp_path / "gpu_resources.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert [row["stage"] for row in rows] == ["quality", "quality", "quality", "timing"]
    assert json.loads((tmp_path / "gpu_resources.json").read_text())["stage"] == "timing"
