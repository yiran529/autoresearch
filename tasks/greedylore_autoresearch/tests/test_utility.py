"""Advisory utility must never become a quality or selection gate."""

import json
import math

import pytest
from greedylore_autoresearch_grader.grader import score_results
from greedylore_autoresearch_grader.utility import advisory_utility, root_reference


def reference():
    return {
        "mode": "research", "commit_hash": "a" * 40, "quality_anchor": "a" * 40,
        "contract_hash": "contract", "quality_source": "measured",
        "quality": {"llama60m_adamw": 3.0, "llama130m_muon": 4.0},
        "timing": {"130m_adamw_bf16": 0.11, "130m_adamw_fp32": 0.22},
    }


def test_utility_penalizes_worst_ppl_increase_without_rewarding_other_workload():
    value = advisory_utility(
        {"llama60m_adamw": 2.9, "llama130m_muon": 4.05},
        {"130m_adamw_bf16": 0.1, "130m_adamw_fp32": 0.2}, reference(),
    )
    assert value["speedup"] == pytest.approx(1.1)
    assert value["loss_penalty"] == pytest.approx(0.05)
    assert value["value"] == pytest.approx(1.0463523669507855)
    assert value["reference"]["commit_hash"] == "a" * 40
    assert value["advisory_only"] is True


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, -1])
def test_utility_is_unavailable_without_valid_evidence(bad):
    ref = reference()
    ref["quality"]["llama130m_muon"] = bad
    value = advisory_utility(
        {"llama60m_adamw": 3.0, "llama130m_muon": 4.0},
        {"130m_adamw_bf16": 0.1, "130m_adamw_fp32": 0.2}, ref,
    )
    assert value["value"] is None
    assert value["reason"]


def test_advisory_utility_does_not_change_aggregate_even_below_one():
    dense = {
        "quality": {"workloads": {
            "llama60m_adamw": {"final_eval_loss": 3.0},
            "llama130m_muon": {"final_eval_loss": 4.0},
        }},
        "efficiency": {"cells": {
            "130m_adamw_bf16": {"mean_iteration_seconds": 0.1},
            "130m_adamw_fp32": {"mean_iteration_seconds": 0.2},
        }},
    }
    quality = {k: {"final_eval_loss": v} for k, v in
               {"llama60m_adamw": 3.1, "llama130m_muon": 4.1}.items()}
    timing = {k: {"mean": v, "relative_spread": 0.01} for k, v in
              {"130m_adamw_bf16": 0.11, "130m_adamw_fp32": 0.22}.items()}
    result = score_results("research", quality, timing, dense, contract_hash="contract",
                          comparable=True, evidence_paths=[], utility_reference=reference())
    assert result.metadata["utility"]["value"] < 1
    assert result.scores["quality_gate"].value == 1
    assert result.aggregated == pytest.approx(1 / 1.1)
    assert result.metadata["evaluation_policy"]["max_relative_eval_loss"] == 1.05


def test_five_percent_gate_accepts_boundary_but_rejects_six_percent():
    from greedylore_autoresearch_grader.results import quality_gate

    assert quality_gate(3.15, 3.0)
    assert not quality_gate(3.16, 3.0)


def test_reference_uses_fixed_root_and_matching_implementation_stage(tmp_path):
    from coral.types import Attempt

    public = tmp_path / "public/attempts"
    public.mkdir(parents=True)
    private = tmp_path / "private"
    private.mkdir()
    (private / "autoresearch_state.json").write_text(json.dumps({"nodes": {"N000": {
        "research": {"selected_sha": "a" * 40}, "engineering_status": "completed",
    }}}))
    def write(sha, mode, score, node="N000"):
        scores = {
            "quality_gate": {"value": 1}, "llama60m_eval_loss": {"value": 3.0},
            "llama130m_eval_loss": {"value": 4.0},
            "bf16_timing_mean": {"value": 0.1}, "fp32_timing_mean": {"value": 0.2},
        }
        a = Attempt(sha, "agent", "test", score, "improved", None, "2026-01-01",
                    metadata={"node_id": node, "mode": mode, "scores": scores,
                              "contract_hash": "contract", "quality_anchor": "a" * 40})
        (public / f"{sha}.json").write_text(json.dumps(a.to_dict()))
    write("a" * 40, "research", 0.9)
    write("b" * 40, "engineering", 1.1)
    write("c" * 40, "engineering", 9.0, "N001")
    assert root_reference(tmp_path, "research", "contract")["commit_hash"] == "a" * 40
    engineering = root_reference(tmp_path, "engineering", "contract")
    assert engineering["commit_hash"] == "b" * 40
    assert engineering["quality_source"] == "inherited (provisional)"
    assert root_reference(tmp_path, "research", "other-policy") is None


def test_low_utility_does_not_prevent_research_selection():
    from autoresearch.research import quality_eligible, select_research_attempt

    from coral.types import Attempt

    candidate = Attempt("b" * 40, "agent", "informative method", 0.8,
                        "regressed", None, "2026-01-01", metadata={
        "scores": {"quality_gate": {"value": 1}, "advisory_utility": {"value": 0.1}},
        "utility": {"value": 0.1, "advisory_only": True}, "comparability": "comparable",
    })
    assert quality_eligible(candidate)
    assert select_research_attempt([candidate]) is candidate


def test_engineering_utility_uses_inherited_losses_and_keeps_provisional_label():
    ref = reference()
    ref["mode"] = "engineering"
    dense = {"quality": {}, "efficiency": {"cells": {
        name: {"mean_iteration_seconds": seconds} for name, seconds in ref["timing"].items()
    }}}
    result = score_results("engineering", {}, {
        name: {"mean": seconds, "relative_spread": 0.01}
        for name, seconds in ref["timing"].items()
    }, dense, contract_hash="contract", comparable=True, evidence_paths=[],
        utility_reference=ref, inherited_quality=ref["quality"])
    assert result.metadata["utility"]["value"] == pytest.approx(1.0)
    assert result.metadata["utility"]["quality_source"] == "inherited (provisional)"
    assert result.scores["quality_gate"].value is None
