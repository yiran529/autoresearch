# ruff: noqa: S101
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


def test_nonperiodic_algorithms_need_explicit_telemetry():
    plan = plan_timing({"period_steps": None}, minimum=400, maximum=800)
    assert validate_cycle_evidence(plan, {"timing_cycle": {
        "period_steps": None, "rank_event_steps": [[], []],
    }, "first_measured_update": 102, "last_measured_update": 501})["confirmed"] is True
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


@pytest.mark.parametrize("period,events", [
    (800, [301, 501, 701, 901]),
    (200, [137, 201, 337, 401]),
    (200, [137, 137, 201, 337, 337, 401]),  # simultaneous operations
    (None, [137, 203, 499]),
    (None, []),
])
def test_composite_cycles_and_adaptive_events_are_valid_measurements(period, events):
    plan = plan_timing({"period_steps": period}, minimum=400, maximum=800)
    payload = {"first_measured_update": 102,
               "last_measured_update": 101 + plan["measured_iterations"],
               "timing_cycle": {"period_steps": period, "rank_event_steps": [events, events]}}
    result = validate_cycle_evidence(plan, payload)
    assert result["confirmed"] is True
    assert result["event_count_per_rank"] == len(events)
    assert result["coverage_scope"] == ("measured_window_only" if period is None else "declared_periods")


@pytest.mark.parametrize("period,events", [
    (200, [137, 201, 337]),  # missing one event in the next cycle
    (200, [137, 201, 338, 401]),  # changed phase in the next cycle
    (None, [137, 136]),  # out of order
    (None, [101]),  # before the measured window
    (None, [502]),  # after the measured window
    (None, [True]),
    (None, [137.0]),
])
def test_malformed_or_nonrepeating_event_evidence_is_rejected(period, events):
    plan = plan_timing({"period_steps": period}, minimum=400, maximum=800)
    payload = {"first_measured_update": 102, "last_measured_update": 501,
               "timing_cycle": {"period_steps": period, "rank_event_steps": [events, events]}}
    assert validate_cycle_evidence(plan, payload)["confirmed"] is False


@pytest.mark.parametrize("bounds", [{}, {"first_measured_update": 102, "last_measured_update": 500}])
def test_nonperiodic_evidence_requires_the_full_measured_window(bounds):
    plan = plan_timing({"period_steps": None}, minimum=400, maximum=800)
    payload = {**bounds, "timing_cycle": {"period_steps": None, "rank_event_steps": [[], []]}}
    assert validate_cycle_evidence(plan, payload)["confirmed"] is False
