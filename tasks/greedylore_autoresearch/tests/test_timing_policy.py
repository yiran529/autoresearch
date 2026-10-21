"""Cycle coverage is host-selected, bounded, and checked against actual events."""

import pytest
from greedylore_autoresearch_grader.timing_policy import (
    TimingPolicyError,
    plan_timing,
    validate_cycle_evidence,
)


@pytest.mark.parametrize("period,steps,covered", [
    (200, 400, True), (400, 400, True), (800, 800, True),
    (300, 600, True), (1600, 400, False), (None, 400, True),
])
def test_host_selects_integer_cycles_with_bounded_cost(period, steps, covered):
    plan = plan_timing({"period_steps": period}, minimum=400, maximum=800)
    assert plan["measured_iterations"] == steps
    assert plan["window_covers_period"] is covered


def test_request_is_not_authority_to_bypass_cap_or_cycle_coverage():
    plan = plan_timing({"period_steps": 800, "requested_steps": 99999}, minimum=400, maximum=800)
    assert plan["measured_iterations"] == 800
    assert plan["requested_steps"] == 99999
    with pytest.raises(TimingPolicyError, match="override"):
        plan_timing({"period_steps": 800}, minimum=400, maximum=800, override=400)
    with pytest.raises(TimingPolicyError, match="override"):
        plan_timing({"period_steps": 200}, minimum=400, maximum=800, override=801)


@pytest.mark.parametrize("period", [0, -1, True, "800"])
def test_invalid_declared_period_is_rejected(period):
    with pytest.raises(TimingPolicyError):
        plan_timing({"period_steps": period}, minimum=400, maximum=800)


def test_missing_or_disagreeing_cycle_events_cannot_certify_timing():
    plan = plan_timing({"period_steps": 800}, minimum=400, maximum=800)
    assert validate_cycle_evidence(plan, {})["confirmed"] is False
    assert validate_cycle_evidence(plan, {"timing_cycle": {
        "period_steps": 800, "rank_event_steps": [[], []],
    }})["confirmed"] is False
    good = {"timing_cycle": {"period_steps": 800, "rank_event_steps": [[901], [901]]},
            "first_measured_update": 102, "last_measured_update": 901}
    assert validate_cycle_evidence(plan, good)["confirmed"] is True
    good["timing_cycle"]["rank_event_steps"][1] = [900]
    assert validate_cycle_evidence(plan, good)["confirmed"] is False


def test_nonperiodic_algorithms_need_explicit_telemetry_and_no_refresh_events():
    plan = plan_timing({"period_steps": None}, minimum=400, maximum=800)
    assert validate_cycle_evidence(plan, {"timing_cycle": {
        "period_steps": None, "rank_event_steps": [[], []],
    }})["confirmed"] is True
    assert validate_cycle_evidence(plan, {})["confirmed"] is False


def test_reference_window_cache_requires_actual_matching_measurement():
    from greedylore_autoresearch_grader.timing_policy import reference_for_window

    reference = {"efficiency": {"cells": {"cell": {"mean_iteration_seconds": 0.1}},
        "timing_windows": {"800": {"measured_iterations": 800,
                                   "cells": {"cell": {"mean_iteration_seconds": 0.11}}}}}}
    result, matches = reference_for_window(reference, 800, 400)
    assert matches and result["efficiency"]["cells"]["cell"]["mean_iteration_seconds"] == 0.11
    assert not reference_for_window(reference, 600, 400)[1]
    reference["efficiency"]["timing_windows"]["800"]["measured_iterations"] = 400
    assert not reference_for_window(reference, 800, 400)[1]


def test_window_mismatch_cannot_produce_a_comparable_speedup():
    from greedylore_autoresearch_grader.grader import score_results

    result = score_results("engineering", {}, {
        "130m_adamw_bf16": {"mean": 0.1, "relative_spread": 0.01},
        "130m_adamw_fp32": {"mean": 0.1, "relative_spread": 0.01},
    }, {"efficiency": {"cells": {
        "130m_adamw_bf16": {"mean_iteration_seconds": 0.2},
        "130m_adamw_fp32": {"mean_iteration_seconds": 0.2},
    }}}, contract_hash="contract", comparable=True, evidence_paths=[],
        performance_eligible=False, ineligibility_reasons=["reference_window_mismatch"],
        timing_protocol={"measured_iterations": 800, "reference_window_matches": False})
    assert result.scores["bf16_timing_mean"].value == 0.1
    assert result.scores["bf16_speedup"].value is None
    assert result.metadata["performance_eligible"] is False
