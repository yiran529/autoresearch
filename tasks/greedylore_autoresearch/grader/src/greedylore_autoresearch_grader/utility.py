"""Optional quality-penalized efficiency, derived only from trusted evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path

WORKLOADS = {"llama60m_adamw": "llama60m", "llama130m_muon": "llama130m"}
CELLS = {"130m_adamw_bf16": "bf16", "130m_adamw_fp32": "fp32"}
UTILITY_VERSION = "mean-log-ppl-ratio-v2"
LEGACY_UTILITY_VERSION = "worst-ppl-discount-v1"


def losses_from_attempt(attempt):
    scores = attempt.metadata.get("scores", {})
    return {name: scores.get(f"{label}_eval_loss", {}).get("value")
            for name, label in WORKLOADS.items()}


def root_reference(coral_dir: Path, mode: str, contract_hash: str):
    """Read the pinned Research root or the completed root's best implementation.

    No Parent/Dense substitution and no changing baseline during root Engineering.
    Every returned value retains the fixed Attempt identities.
    """
    from coral.hub.attempts import read_attempts

    try:
        state = json.loads((coral_dir / "private/autoresearch_state.json").read_text())
        root = state.get("nodes", {}).get("N000", {})
        attempts = [a for a in read_attempts(coral_dir, include_archived=True)
                    if a.metadata.get("node_id") == "N000" and not a.archived
                    and a.metadata.get("contract_hash") == contract_hash
                    and a.status in {"improved", "baseline", "regressed", "reverted"}
                    and a.budget_class == "real" and a.metadata.get("performance_eligible") is not False
                    and not a.metadata.get("failure_class")]
        research = [a for a in attempts if a.metadata.get("mode") == "research"]
        selected = root.get("research", {}).get("selected_sha")
        if selected is None and root.get("research", {}).get("status") != "completed":
            return None
        research = [a for a in research if selected is None or a.commit_hash == selected]
        if len(research) != 1:
            return None
        quality = research[0]
        anchor = quality
        if mode == "engineering":
            if root.get("engineering_status") != "completed":
                return None
            implementations = [a for a in attempts if a.metadata.get("mode") == mode
                               and a.metadata.get("quality_anchor") == quality.commit_hash
                               and type(a.score) in (int, float) and math.isfinite(a.score)]
            if not implementations:
                return None
            anchor = min(implementations, key=lambda a: (-a.score, a.timestamp, a.commit_hash))
        scores = anchor.metadata.get("scores", {})
        return {
            "mode": mode, "commit_hash": anchor.commit_hash,
            "quality_anchor": quality.commit_hash, "contract_hash": contract_hash,
            "quality_source": "measured" if mode == "research" else "inherited (provisional)",
            "quality": losses_from_attempt(quality),
            "timing": {name: scores.get(f"{label}_timing_mean", {}).get("value")
                       for name, label in CELLS.items()},
            "timing_protocol": anchor.metadata.get("timing_protocol"),
            "hardware_comparison": anchor.metadata.get("hardware_comparison"),
        }
    except (OSError, ValueError, TypeError, KeyError):
        return None


def advisory_utility(quality, timing, reference, *, unavailable_reason=None, version=UTILITY_VERSION):
    if version not in {UTILITY_VERSION, LEGACY_UTILITY_VERSION}:
        raise ValueError('unsupported utility version')
    legacy = version == LEGACY_UTILITY_VERSION
    result = {
        "version": version, "advisory_only": True,
        "formula": ("S * exp(-max(0, candidate_loss - baseline_loss over workloads))" if legacy else
                    "S * exp(-mean(candidate_loss - baseline_loss over the two language workloads))"),
        "value": None, "speedup": None, "loss_penalty" if legacy else "mean_loss_delta": None,
        "reference": reference, "reason": unavailable_reason,
    }
    if unavailable_reason or reference is None:
        result["reason"] = unavailable_reason or "fixed_root_reference_unavailable"
        return result
    try:
        numbers = [*quality.values(), *timing.values(),
                   *reference["quality"].values(), *reference["timing"].values()]
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in numbers):
            raise ValueError("nonfinite or nonpositive measurement")
        deltas = [quality[k] - reference['quality'][k] for k in WORKLOADS]
        delta = max(0., *deltas) if legacy else sum(deltas) / len(WORKLOADS)
        speedup = math.exp(sum(math.log(reference["timing"][k] / timing[k]) for k in CELLS) / 2)
        result.update(value=speedup * math.exp(-delta), speedup=speedup,
                      **{'loss_penalty' if legacy else 'mean_loss_delta': delta}, reason=None)
    except (KeyError, TypeError, ValueError, OverflowError):
        result["reason"] = "invalid_or_missing_measurement"
    return result
