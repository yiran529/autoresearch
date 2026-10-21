import math

import pytest
from greedylore_autoresearch_grader.results import (
    ResultError,
    TimingRepetition,
    aggregate_timing_cell,
    geometric_mean_speedup,
    quality_gate,
    timing_mean,
    timing_summary,
)


def test_quality_gate_boundary():
    assert quality_gate(1.05, 1.0)
    assert not quality_gate(1.0500001, 1.0)


def test_two_repetition_mean_and_speedup():
    assert timing_mean([0.20, 0.22]) == pytest.approx(0.21)
    assert geometric_mean_speedup({"bf16": 2.0, "fp32": 0.5}) == pytest.approx(1.0)
    assert timing_summary([0.20, 0.22]).relative_spread == pytest.approx(0.02 / 0.21)


@pytest.mark.parametrize("values", [[0.2], [0.2, 0], [0.2, math.nan], [0.2, math.inf]])
def test_timing_rejects_incomplete_or_nonfinite(values):
    with pytest.raises(ResultError):
        timing_mean(values)


def test_timing_cell_requires_matching_contract_bucket_and_hardware():
    base = TimingRepetition(0.2, "contract", "buckets", "hardware")
    for changed in (
        TimingRepetition(0.21, "other", "buckets", "hardware"),
        TimingRepetition(0.21, "contract", "other", "hardware"),
        TimingRepetition(0.21, "contract", "buckets", "other"),
    ):
        with pytest.raises(ResultError, match="metadata mismatch"):
            aggregate_timing_cell([base, changed])
    result = aggregate_timing_cell(
        [base, TimingRepetition(0.22, "contract", "buckets", "hardware")]
    )
    assert result.mean == pytest.approx(0.21)
