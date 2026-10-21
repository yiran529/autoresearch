"""Host-owned bounded windows; declarations are requests, never timer authority."""

from __future__ import annotations


class TimingPolicyError(ValueError):
    pass


def _integer(value, field):
    if type(value) is not int or value <= 0:
        raise TimingPolicyError(f"{field} must be a positive integer")
    return value


def validate_declaration(value):
    if not isinstance(value, dict) or set(value) - {"period_steps", "requested_steps"}:
        raise TimingPolicyError("timing declaration has an invalid schema")
    if "period_steps" not in value:
        raise TimingPolicyError("timing declaration requires period_steps (null for nonperiodic)")
    if value["period_steps"] is not None:
        _integer(value["period_steps"], "period_steps")
    if value.get("requested_steps") is not None:
        _integer(value["requested_steps"], "requested_steps")
    return dict(value)


def plan_timing(declaration, *, minimum, maximum, override=None):
    declaration = validate_declaration(declaration)
    minimum = _integer(minimum, "minimum")
    maximum = _integer(maximum, "maximum")
    if minimum > maximum:
        raise TimingPolicyError("timing minimum exceeds maximum")
    period = declaration["period_steps"]
    requested = declaration.get("requested_steps")
    target = min(maximum, max(minimum, requested or minimum))
    steps = ((target + period - 1) // period) * period if period else target
    if period and steps > maximum and period <= maximum:
        steps = (maximum // period) * period
    covered = minimum <= steps <= maximum
    if not covered:
        steps = minimum
    if override is not None:
        _integer(override, "override")
        if not minimum <= override <= maximum or (period and override % period):
            raise TimingPolicyError("host timing override violates bounds or complete-cycle alignment")
        steps, covered = override, True
    return {"version": "bounded-cycles-v2", "minimum_iterations": minimum,
            "maximum_iterations": maximum, "period_steps": period,
            "requested_steps": requested, "host_override": override,
            "measured_iterations": steps, "window_covers_period": covered}


def validate_cycle_evidence(plan, payload):
    result = {"confirmed": False, "reason": "missing_or_invalid_cycle_evidence"}
    if not plan["window_covers_period"]:
        return {"confirmed": False, "reason": "period_exceeds_measurement_cap"}
    cycle = payload.get("timing_cycle")
    if not isinstance(cycle, dict) or cycle.get("period_steps") != plan["period_steps"]:
        return result
    ranks = cycle.get("rank_event_steps")
    if (not isinstance(ranks, list) or not ranks
            or any(not isinstance(events, list) for events in ranks)
            or any(events != ranks[0] for events in ranks)):
        return result
    period = plan["period_steps"]
    events = ranks[0]
    first, last = payload.get("first_measured_update"), payload.get("last_measured_update")
    if (type(first) is not int or type(last) is not int or first <= 0
            or last - first + 1 != plan["measured_iterations"]
            or any(type(step) is not int or not first <= step <= last for step in events)
            or any(b < a for a, b in zip(events, events[1:]))):
        return result
    if period is not None:
        if not events or plan["measured_iterations"] % period:
            return result
        # A complete cycle may contain several events, including simultaneous
        # operations. Check the observed pattern, not a single-refresh schedule.
        offsets = [step - first for step in events if step < first + period]
        expected = [first + offset + cycle * period
                    for cycle in range(plan["measured_iterations"] // period)
                    for offset in offsets]
        if events != expected:
            return result
    return {"confirmed": True, "reason": None,
            "event_count_per_rank": len(events),
            "coverage_scope": "declared_periods" if period else "measured_window_only"}


def reference_for_window(reference, steps, default_steps):
    """Select an explicitly measured host reference; never rescale per-step time."""
    efficiency = reference["efficiency"]
    recorded = (efficiency.get("timing_protocol") or {}).get("measured_iterations", default_steps)
    if recorded == steps:
        return reference, True
    window = efficiency.get("timing_windows", {}).get(str(steps))
    if isinstance(window, dict) and window.get("measured_iterations") == steps:
        import math

        cells = window.get("cells")
        if isinstance(cells, dict) and set(cells) == set(efficiency["cells"]):
            try:
                if all(type(cells[name]["mean_iteration_seconds"]) in (int, float)
                       and math.isfinite(cells[name]["mean_iteration_seconds"])
                       and cells[name]["mean_iteration_seconds"] > 0 for name in cells):
                    return {**reference, "efficiency": {**efficiency, "cells": cells}}, True
            except (KeyError, TypeError):
                pass
    return reference, False
