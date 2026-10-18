"""Strict aggregation helpers for formal evaluation results."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


class ResultError(ValueError):
    pass


@dataclass(frozen=True)
class TimingSummary:
    mean: float
    relative_spread: float


@dataclass(frozen=True)
class TimingRepetition:
    seconds_per_iteration: float
    contract_hash: str
    bucket_layout: str
    hardware_fingerprint: str


def quality_gate(candidate_loss: float, dense_loss: float, *, max_relative: float = 1.06) -> bool:
    return (
        all(math.isfinite(x) and x > 0 for x in (candidate_loss, dense_loss))
        and candidate_loss <= max_relative * dense_loss
    )


def timing_mean(values: Sequence[float]) -> float:
    if len(values) != 2 or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ResultError("timing cell requires exactly two finite positive repetitions")
    return sum(values) / 2.0


def timing_summary(values: Sequence[float]) -> TimingSummary:
    mean = timing_mean(values)
    return TimingSummary(mean, abs(values[0] - values[1]) / mean)


def geometric_mean_speedup(speedups: Mapping[str, float]) -> float:
    if not speedups or any(not math.isfinite(x) or x <= 0 for x in speedups.values()):
        raise ResultError("speedups must be finite and positive")
    return math.exp(sum(math.log(x) for x in speedups.values()) / len(speedups))


def aggregate_timing_cell(repetitions: Sequence[TimingRepetition]) -> TimingSummary:
    if len(repetitions) != 2:
        raise ResultError("timing cell requires exactly two repetitions")
    first = repetitions[0]
    expected = (first.contract_hash, first.bucket_layout, first.hardware_fingerprint)
    if any(
        (item.contract_hash, item.bucket_layout, item.hardware_fingerprint) != expected
        for item in repetitions[1:]
    ):
        raise ResultError("timing repetition metadata mismatch")
    return timing_summary([item.seconds_per_iteration for item in repetitions])
