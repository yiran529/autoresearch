"""Restricted translation of the two seed methods into existing CLI arguments."""

import math
import re
from collections.abc import Mapping

_GREEDY = {
    "name",
    "rank",
    "ef_rank",
    "update_gap",
    "min_compression_rate",
    "beta_ef",
    "error_inherit",
    "quality_start_step",
}
_ARC = {
    "name",
    "r",
    "compress_ratio",
    "use_error_feedback",
    "error_decay",
    "quality_start_step",
    "vision_start_step",
    "timing_period_steps",
}


def _number(value, name, *, integer=False, minimum=0, maximum=None):
    if (
        type(value) not in ((int,) if integer else (int, float))
        or not math.isfinite(value)
        or value < minimum
        or maximum is not None
        and value > maximum
    ):
        raise ValueError(f"invalid compression parameter {name}")
    return value


def validate_seed(seed, compressor):
    required = {
        "version",
        "id",
        "upstream_repository",
        "upstream_commit",
        "imported_files",
        "root_hypothesis",
    }
    if (
        not isinstance(seed, Mapping)
        or set(seed) != required
        or type(seed["version"]) is not int
        or seed["version"] != 1
        or any(
            not isinstance(seed[key], str) or not seed[key].strip()
            for key in ("id", "upstream_repository", "root_hypothesis")
        )
        or not re.fullmatch("[a-z0-9][a-z0-9_-]*", seed["id"])
        or not re.fullmatch("[0-9a-f]{40}", str(seed["upstream_commit"]))
        or not isinstance(seed["imported_files"], Mapping)
        or not seed["imported_files"]
    ):
        raise ValueError("invalid seed declaration or Root identity")
    from pathlib import PurePosixPath

    for path, digest in seed["imported_files"].items():
        if (
            not isinstance(path, str)
            or not path
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or "\\" in path
            or not re.fullmatch("[0-9a-f]{64}", str(digest))
        ):
            raise ValueError("invalid seed imported file path or hash")


def compression_arguments(config, start_step, *, vision=False):
    """No training or optimizer fields can cross this boundary."""
    if not isinstance(config, Mapping):
        raise ValueError("compression configuration must be a mapping")
    name = config.get("name")
    allowed = (_ARC | _GREEDY) if name == "none" else _ARC if name == "group_topk" else _GREEDY
    if set(config) - allowed or name not in {"none", "top_subspace", "group_topk"}:
        raise ValueError("unknown compression parameters or method")

    def flag(value):
        return "--" + (value.replace("_", "-") if vision else value)

    args = ["--compressor", name]
    if name == "none" or vision and name == "top_subspace":
        return args
    _number(start_step, "start_step", integer=True)
    args += [flag("start_compress_iter"), str(start_step)]
    if name == "group_topk":
        _number(config["r"], "r", integer=True, minimum=1)
        _number(config["compress_ratio"], "compress_ratio", minimum=1e-12, maximum=1)
        _number(config["error_decay"], "error_decay", minimum=1, maximum=1)
        if config["use_error_feedback"] not in {"noef", "ef14", "ef21"}:
            raise ValueError("invalid compression error feedback")
        for key in ("quality_start_step", "vision_start_step"):
            _number(config[key], key, integer=True)
        if config["timing_period_steps"] is not None:
            raise ValueError("ARC per-step randomized projection has no fixed timing period")
        pairs = [
            ("r", config["r"]),
            ("compress_ratio", config["compress_ratio"]),
            ("use_error_feedback", config["use_error_feedback"]),
            ("error_decay", config["error_decay"]),
        ]
    else:
        _number(config["rank"], "rank", integer=True, minimum=1)
        _number(config["update_gap"], "update_gap", integer=True, minimum=1)
        _number(config["min_compression_rate"], "min_compression_rate", minimum=1e-12)
        _number(config["beta_ef"], "beta_ef", maximum=1)
        _number(config["error_inherit"], "error_inherit", maximum=1)
        if (
            config["error_inherit"] not in (0, 1)
            or type(config["ef_rank"]) is not int
            or config["ef_rank"] not in (14, 21)
        ):
            raise ValueError("invalid compression error feedback")
        pairs = [
            ("compress_rank", config["rank"]),
            ("update_proj_gap", config["update_gap"]),
            ("use_error_feedback", f"ef{config['ef_rank']}"),
            ("min_compression_rate", config["min_compression_rate"]),
            ("beta_ef", config["beta_ef"]),
            ("error_inherit", format(config["error_inherit"], "g")),
        ]
    for key, value in pairs:
        args += [flag(key), str(value)]
    return args
