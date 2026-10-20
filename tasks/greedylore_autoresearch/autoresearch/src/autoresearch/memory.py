"""Filename-only Global Memory gate and Summary validation."""

from __future__ import annotations

import fcntl
import json
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from coral.grader.trusted_writes import trusted_writes_lock
from coral.hub._island import island_root

from .analysis import run_fresh_codex
from .atomic import atomic_write_text

_REF = re.compile(
    r"^nodes/(N[0-9]{3})/(research|engineering)_analysis_v[1-9][0-9]*\.md$"
)


@dataclass(frozen=True)
class MemoryGate:
    due: bool
    trigger_research_refs: frozenset[str]


@dataclass(frozen=True)
class MemoryContext:
    coral_dir: Path
    island_id: str | None = None

    @property
    def island_root(self) -> Path:
        return island_root(Path(self.coral_dir).resolve(), self.island_id)

    @property
    def notes_root(self) -> Path:
        return self.island_root / "notes"


@dataclass(frozen=True)
class MemoryResult:
    status: str
    research_summary_path: Path | None = None
    engineering_summary_path: Path | None = None


def list_analysis_refs(notes_root: Path, mode: str | None = None) -> tuple[str, ...]:
    if mode not in {None, "research", "engineering"}:
        raise ValueError("invalid Analysis mode")
    notes_root = Path(notes_root)
    nodes = notes_root / "nodes"
    if not nodes.is_dir() or nodes.is_symlink():
        return ()
    refs = []
    for node in nodes.iterdir():
        if not node.is_dir() or node.is_symlink():
            continue
        for path in node.iterdir():
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(notes_root).as_posix()
            matched = _REF.fullmatch(relative)
            if matched and (mode is None or matched.group(2) == mode):
                refs.append(relative)
    return tuple(sorted(refs))


def _parse_summary(content: str) -> frozenset[str]:
    if not isinstance(content, str) or not content.startswith("---\n"):
        raise ValueError("Summary requires YAML frontmatter")
    marker = content.find("\n---\n", 4)
    if marker < 0:
        raise ValueError("Summary requires closing YAML frontmatter")
    try:
        frontmatter = yaml.safe_load(content[4:marker])
    except yaml.YAMLError as exc:
        raise ValueError("Summary frontmatter is invalid") from exc
    if not isinstance(frontmatter, dict) or set(frontmatter) != {"processed_analyses"}:
        raise ValueError("Summary frontmatter requires processed_analyses")
    refs = frontmatter["processed_analyses"]
    if not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs):
        raise ValueError("Summary processed_analyses must be a string list")
    if len(refs) != len(set(refs)):
        raise ValueError("Summary processed_analyses contains duplicates")
    if not content[marker + 5:].strip():
        raise ValueError("Summary Markdown body is empty")
    return frozenset(refs)


def read_processed_analyses(summary: Path) -> frozenset[str]:
    summary = Path(summary)
    if not summary.exists():
        return frozenset()
    return _parse_summary(summary.read_text(encoding="utf-8"))


def validate_summary(
    content: str,
    notes_root: Path,
    required_refs: frozenset[str] = frozenset(),
) -> frozenset[str]:
    refs = _parse_summary(content)
    notes_root = Path(notes_root)
    for ref in refs:
        if _REF.fullmatch(ref) is None:
            raise ValueError("Summary has a non-canonical Analysis reference")
        path = notes_root / ref
        if (
            path.is_symlink()
            or path.parent.is_symlink()
            or path.parent.parent.is_symlink()
            or not path.is_file()
        ):
            raise ValueError(f"Summary Analysis reference does not exist: {ref}")
    if not required_refs.issubset(refs):
        raise ValueError("Summary does not cover trigger references")
    return refs


def memory_due(notes_root: Path, threshold: int = 3) -> MemoryGate:
    if not isinstance(threshold, int) or threshold < 1:
        raise ValueError("Memory threshold must be a positive integer")
    notes_root = Path(notes_root)
    processed = (
        read_processed_analyses(notes_root / "research_summary.md")
        & read_processed_analyses(notes_root / "engineering_summary.md")
    )
    research_refs = frozenset(list_analysis_refs(notes_root, mode="research"))
    unprocessed = research_refs - processed
    node_ids = {ref.split("/")[1] for ref in unprocessed}
    return MemoryGate(len(node_ids) >= threshold, unprocessed)


def _host_publish(context: MemoryContext, kind: str, content: str) -> Path:
    targets = {
        "research_summary": "research_summary.md",
        "engineering_summary": "engineering_summary.md",
    }
    if kind not in targets:
        raise ValueError("unsupported Memory Summary target")
    target = context.notes_root / targets[kind]
    with trusted_writes_lock(context.coral_dir):
        return atomic_write_text(target, content)


def _stage_public_inputs(context: MemoryContext, scratch: Path) -> tuple[Path, ...]:
    """Copy only public navigation/evidence files for the Memory session."""
    staged_notes = scratch / "public" / "notes"
    staged_logs = scratch / "public" / "eval_logs"
    staged_notes.mkdir(parents=True)
    staged_logs.mkdir(parents=True)

    def copy_file(source: Path, destination: Path) -> None:
        if source.is_file() and not source.is_symlink():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

    for name in ("tree.md", "research_summary.md", "engineering_summary.md"):
        copy_file(context.notes_root / name, staged_notes / name)
    nodes = context.notes_root / "nodes"
    if nodes.is_dir() and not nodes.is_symlink():
        for node in nodes.iterdir():
            if not node.is_dir() or node.is_symlink() or not re.fullmatch(r"N[0-9]{3}", node.name):
                continue
            for name in ("spec.md", "evaluation.md"):
                copy_file(node / name, staged_notes / "nodes" / node.name / name)
    for ref in list_analysis_refs(context.notes_root):
        copy_file(context.notes_root / ref, staged_notes / ref)
        copy_file((context.notes_root / ref).with_suffix('.json'), (staged_notes / ref).with_suffix('.json'))

    log_root = context.island_root / "eval_logs"
    if log_root.is_dir() and not log_root.is_symlink():
        for source in log_root.rglob("*"):
            if (source.is_file() and not source.is_symlink()
                    and source.suffix in {".json", ".jsonl", ".log", ".md", ".txt"}
                    and source.stat().st_size <= 1_000_000
                    and not any(parent.is_symlink() for parent in source.parents if parent != log_root)):
                copy_file(source, staged_logs / source.relative_to(log_root))
    staged_diagnostics = _stage_diagnostics(context, staged_notes, scratch)
    return staged_notes, staged_logs, staged_diagnostics



def _stage_diagnostics(context: MemoryContext, staged_notes: Path, scratch: Path) -> Path:
    """Stage bounded text evidence only for published Analysis identities."""
    destination = scratch / "public" / "diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    source_root = context.island_root / "diagnostics"
    if not source_root.is_dir() or source_root.is_symlink():
        return destination
    remaining_bytes, remaining_files = 8_000_000, 256

    def safe_file(path: Path) -> bool:
        return (path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(source_root.resolve())
                and not any(p.is_symlink() for p in path.parents if p.is_relative_to(source_root))
                and path.stat().st_size <= 1_000_000)

    def copy_file(path: Path) -> None:
        nonlocal remaining_bytes, remaining_files
        if not safe_file(path):
            return
        size = path.stat().st_size
        if remaining_files <= 0 or size > remaining_bytes:
            return
        target = destination / path.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        remaining_bytes -= size
        remaining_files -= 1

    for ref in list_analysis_refs(staged_notes):
        match = _REF.fullmatch(ref)
        node_id, mode = match.groups()
        version = int(Path(ref).stem.rsplit("_v", 1)[1])
        try:
            from .validation import load_document

            analysis = load_document(staged_notes / ref)
            sha = analysis["research_sha"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        identity = f"{node_id}-{mode}-v{version}"
        records = source_root / identity / "records"
        if not records.is_dir() or any(p.is_symlink() for p in (records, records.parent)):
            continue
        for record in sorted(records.iterdir()):
            if remaining_files <= 0 or remaining_bytes <= 0:
                return destination
            if not record.is_dir() or record.is_symlink() or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", record.name):
                continue
            metadata = record / "record.json"
            if not safe_file(metadata):
                continue
            try:
                data = json.loads(metadata.read_text())
            except (OSError, ValueError):
                continue
            if (not isinstance(data, dict) or data.get("node_id") != node_id or data.get("mode") != mode
                    or data.get("analysis_version") != version or data.get("research_sha") != sha
                    or data.get("diagnostic_id") != f"{identity}/{record.name}"):
                continue
            copy_file(metadata)
            if not (destination / metadata.relative_to(source_root)).is_file():
                continue
            for name in ("stdout.log", "stderr.log"):
                copy_file(record / name)
            results = record / "results"
            if results.is_dir() and not results.is_symlink():
                for path in sorted(results.rglob("*")):
                    if remaining_files <= 0 or remaining_bytes <= 0:
                        return destination
                    if path.suffix in {".json", ".jsonl", ".csv", ".tsv", ".txt", ".md", ".log"}:
                        copy_file(path)
    return destination


def consolidate_memory(
    *,
    if_due: bool,
    context: MemoryContext | None = None,
    runner: Callable[..., Path] = run_fresh_codex,
    publisher: Callable[[str, str], Any] | None = None,
) -> MemoryResult:
    """Run one bounded read-only consolidation under a non-blocking file lock."""
    if context is None:
        raise ValueError("Memory context is required")
    if context.island_id is not None:
        raise ValueError("dynamic Memory currently supports one island")
    context.island_root.mkdir(parents=True, exist_ok=True)
    lock_path = context.island_root / "memory-consolidation.lock"
    with lock_path.open("a+b") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return MemoryResult("locked")
        gate = memory_due(context.notes_root)
        if if_due and not gate.due:
            return MemoryResult("not_due")
        prompt_path = Path(__file__).resolve().parents[3] / "prompts" / "global_memory_manager.md"
        schema_path = Path(__file__).resolve().parents[2] / "schemas" / "memory.schema.json"
        prompt = prompt_path.read_text(encoding="utf-8") + "\n\n" + json.dumps({
            "trigger_research_refs": sorted(gate.trigger_research_refs),
            "research_summary": "research_summary.md",
            "engineering_summary": "engineering_summary.md",
        }, sort_keys=True)
        with tempfile.TemporaryDirectory(prefix="autoresearch-memory-") as scratch_dir:
            scratch = Path(scratch_dir)
            read_roots = _stage_public_inputs(context, scratch)
            output = runner(
                prompt,
                read_roots,
                schema_path,
                scratch,
            )
            document = json.loads(Path(output).read_text(encoding="utf-8"))
        if not isinstance(document, dict) or set(document) != {
            "research_summary", "engineering_summary",
        }:
            raise ValueError("Memory worker output has an invalid schema")
        for kind in ("research_summary", "engineering_summary"):
            validate_summary(
                document[kind], context.notes_root,
                required_refs=gate.trigger_research_refs,
            )
        publish = publisher or (lambda kind, content: _host_publish(context, kind, content))
        publish("research_summary", document["research_summary"])
        publish("engineering_summary", document["engineering_summary"])
        return MemoryResult(
            "published",
            context.notes_root / "research_summary.md",
            context.notes_root / "engineering_summary.md",
        )


def run_worker(coral_dir: str) -> dict[str, str]:
    """Fixed grader-owned one-shot entrypoint; never accepts Agent paths."""
    result = consolidate_memory(if_due=True, context=MemoryContext(Path(coral_dir)))
    return {"status": result.status}
