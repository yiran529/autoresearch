"""Incremental completed-Node knowledge, navigation and durable publication."""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from coral.grader.trusted_writes import _atomic_json, trusted_writes_lock
from coral.hub._island import island_root
from coral.template.shared_research_goal import with_shared_research_goal

from .analysis import run_fresh_codex
from .atomic import atomic_write_text

_TARGETS = {'research_summary': 'research_summary.md',
            'engineering_summary': 'engineering_summary.md', 'wisdom': 'wisdom.md'}
_KNOWLEDGE_VERSION = 2
_REF = re.compile(
    r"^nodes/(N[0-9]{3})/(research|engineering)_analysis_v[1-9][0-9]*\.md$"
)


@dataclass(frozen=True)
class MemoryGate:
    due: bool
    trigger_refs: frozenset[str]
    pending_nodes: tuple[str, ...] = ()


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
    wisdom_path: Path | None = None


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


def _completed_analysis_refs(notes_root: Path) -> dict[str, frozenset[str]]:
    from .state import WorkflowStore

    path = notes_root.parent.parent / "private/autoresearch_state.json"
    if not path.exists():
        return {}
    state = json.loads(path.read_text())
    available = frozenset(list_analysis_refs(notes_root))
    result = {}
    for node_id, record in state["nodes"].items():
        if not WorkflowStore._node_completed(record):
            continue
        refs = frozenset(ref for ref in available if ref.split("/")[1] == node_id)
        research = (record.get("analysis") or {}).get("analysis_ref")
        if research not in refs:
            continue
        if record.get("engineering_claims") and not any("/engineering_analysis_" in ref for ref in refs):
            continue
        result[node_id] = refs
    return result


def memory_due(notes_root: Path) -> MemoryGate:
    """Discover completed Nodes and persist their pending/completed consolidation."""
    notes_root = Path(notes_root)
    coral_dir = notes_root.parent.parent
    path = coral_dir / "private/memory_state.json"
    with trusted_writes_lock(coral_dir):
        completed = _completed_analysis_refs(notes_root)
        processed = (read_processed_analyses(notes_root / 'research_summary.md')
                     & read_processed_analyses(notes_root / 'engineering_summary.md')
                     & read_processed_analyses(notes_root / 'wisdom.md'))
        first_run = not path.exists()
        state = json.loads(path.read_text()) if not first_run else {"version": 1, "nodes": {}}
        before = json.dumps(state, sort_keys=True)
        for node_id, refs in sorted(completed.items()):
            row = state["nodes"].get(node_id)
            legacy = row is not None and row.get('knowledge_version') != _KNOWLEDGE_VERSION
            if (row is None or row['analysis_refs'] != sorted(refs)
                    or legacy and (refs <= processed or row['status'] == 'completed')):
                covered_legacy = refs <= processed and (first_run or legacy)
                state["nodes"][node_id] = {
                    "status": "completed" if covered_legacy else "pending",
                    "knowledge_version": _KNOWLEDGE_VERSION if covered_legacy else None,
                    "analysis_refs": sorted(refs),
                    "sequence": row["sequence"] if row else len(state["nodes"]),
                    "last_error": None,
                }
        if first_run or json.dumps(state, sort_keys=True) != before:
            _atomic_json(path, state)
        pending = tuple(sorted(
            (node for node, row in state["nodes"].items()
             if row["status"] == "pending" and node in completed),
            key=lambda node: state["nodes"][node]["sequence"],
        ))
        return MemoryGate(bool(pending), completed[pending[0]] if pending else frozenset(), pending)


def _record_memory_result(context, node_id, refs, error=None):
    with trusted_writes_lock(context.coral_dir):
        path = context.coral_dir / "private/memory_state.json"
        state = json.loads(path.read_text())
        row = state["nodes"][node_id]
        if row["analysis_refs"] == sorted(refs):
            row.update(status="pending" if error else "completed", last_error=error)
            if not error:
                row['knowledge_version'] = _KNOWLEDGE_VERSION
            _atomic_json(path, state)


def _host_publish(context: MemoryContext, kind: str, content: str) -> Path:
    if kind not in _TARGETS:
        raise ValueError("unsupported Memory Summary target")
    target = context.notes_root / _TARGETS[kind]
    with trusted_writes_lock(context.coral_dir):
        return atomic_write_text(target, content, mode=0o644)


def _stage_public_inputs(context: MemoryContext, scratch: Path, *, node_ids=None, evidence_refs=None, analysis_refs=None) -> tuple[Path, ...]:
    """Copy only public navigation/evidence files for the Memory session."""
    staged_notes = scratch / "public" / "notes"
    staged_logs = scratch / "public" / "eval_logs"
    staged_notes.mkdir(parents=True)
    staged_logs.mkdir(parents=True)

    def copy_file(source: Path, destination: Path) -> None:
        if source.is_file() and not source.is_symlink():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)

    for name in ("tree.md", *_TARGETS.values()):
        copy_file(context.notes_root / name, staged_notes / name)
    nodes = context.notes_root / "nodes"
    if nodes.is_dir() and not nodes.is_symlink():
        for node in nodes.iterdir():
            if not node.is_dir() or node.is_symlink() or not re.fullmatch(r"N[0-9]{3}", node.name):
                continue
            if node_ids is not None and node.name not in node_ids:
                continue
            for name in ("spec.md", "evaluation.md"):
                copy_file(node / name, staged_notes / "nodes" / node.name / name)
    for ref in (list_analysis_refs(context.notes_root) if analysis_refs is None else sorted(analysis_refs)):
        if _REF.fullmatch(ref) is None:
            raise ValueError("invalid Memory Analysis reference")
        copy_file(context.notes_root / ref, staged_notes / ref)
        copy_file((context.notes_root / ref).with_suffix('.json'), (staged_notes / ref).with_suffix('.json'))

    # History reports are cheap to make available; only the newly required
    # reports supply raw evidence. Never sweep all runtime logs into a session.
    from .runtime_artifacts import safe_path
    from .validation import load_document

    budget = {'files': 256, 'bytes': 8_000_000}
    research_diagnostics = scratch / 'public/research_diagnostics'
    staged_diagnostics = scratch / 'public/diagnostics'
    for root in (research_diagnostics, staged_diagnostics):
        root.mkdir(parents=True, exist_ok=True)
    selected = list_analysis_refs(staged_notes) if evidence_refs is None else evidence_refs
    frozen_records = []
    for ref in sorted(selected):
        if _REF.fullmatch(ref) is None:
            raise ValueError('invalid Memory evidence Analysis reference')
        try:
            document = load_document(staged_notes / ref)
        except (OSError, ValueError):
            continue
        manifest = document.get('evidence_manifest')
        records = manifest.get('evidence_records', []) if isinstance(manifest, dict) else []
        if not isinstance(records, list):
            continue
        frozen_records.extend(records)
    # Preserve current raw evidence before spending the shared budget on history.
    for record in sorted(frozen_records, key=lambda r: (
        isinstance(r, dict) and isinstance(r.get('path'), str)
        and r['path'].startswith('diagnostics/analysis-context-')
    )):
        if not isinstance(record, dict) or record.get('status') != 'present':
            continue
        reference = record.get('path')
        if not isinstance(reference, str):
            continue
        relative = Path(reference)
        if (relative.is_absolute() or '..' in relative.parts or not relative.parts
                or relative.parts[0] not in {'eval_logs', 'research_diagnostics', 'diagnostics', 'notes'}
                or relative.parts[0] == 'notes' and not re.fullmatch(
                    r'notes/nodes/N[0-9]{3}/research\.json', reference)):
            continue
        source = context.island_root / relative
        try:
            safe_path(source)
        except ValueError:
            continue
        if (not source.is_file() or source.suffix not in
                {'.json', '.jsonl', '.log', '.md', '.txt', '.csv', '.tsv'}):
            continue
        size = source.stat().st_size
        if size > 1_000_000 or size > budget['bytes'] or budget['files'] <= 0:
            continue
        destination = scratch / 'public' / relative
        if destination.exists():
            continue
        # Read bounded bytes once: the copy must be exactly what was hashed,
        # not a later filesystem version. Missing legacy hashes stay omitted.
        with source.open('rb') as stream:
            data = stream.read(1_000_001)
        if len(data) > 1_000_000 or hashlib.sha256(data).hexdigest() != record.get('sha256'):
            continue
        if len(data) > budget['bytes']:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        budget['bytes'] -= len(data)
        budget['files'] -= 1
    return staged_notes, staged_logs, staged_diagnostics, research_diagnostics


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
        published = False
        while True:
            gate = memory_due(context.notes_root)
            if not gate.due and (if_due or published):
                return MemoryResult("published" if published else "not_due",
                                    context.notes_root / "research_summary.md" if published else None,
                                    context.notes_root / "engineering_summary.md" if published else None,
                                    context.notes_root / "wisdom.md" if published else None)
            node_id = gate.pending_nodes[0] if gate.pending_nodes else None
            try:
                _consolidate_node(context, gate, runner, publisher)
            except Exception as exc:
                if node_id is not None:
                    _record_memory_result(context, node_id, gate.trigger_refs, str(exc))
                raise
            if node_id is None:
                return MemoryResult("published", context.notes_root / "research_summary.md",
                                    context.notes_root / "engineering_summary.md", context.notes_root / "wisdom.md")
            _record_memory_result(context, node_id, gate.trigger_refs)
            published = True


def _body(content):
    if not isinstance(content, str) or not content.strip():
        raise ValueError('Memory document body must be nonempty Markdown')
    if content.startswith('---\n'):
        # Accept old response formatting, but never trust model-authored coverage.
        _parse_summary(content)
        return content[content.find('\n---\n', 4) + 5:].strip()
    return content.strip()


def _with_coverage(content, covered):
    header = yaml.safe_dump({'processed_analyses': sorted(covered)}, sort_keys=False).rstrip()
    return '---\n' + header + '\n---\n' + _body(content) + '\n'


def _validate_wisdom(content, notes_root, covered):
    # Exact source versions are mandatory in the writing contract. This boundary
    # checks cited Analysis paths, not whether a scientific claim is true.
    for ref in re.findall(r'nodes/[^\s<>\[\]()]+?_analysis_v[^\s<>\[\](),;]+?\.(?:md|json)', _body(content)):
        if ref not in covered or _REF.fullmatch(ref) is None:
            raise ValueError(f'wisdom reference is outside covered Analyses: {ref}')
        path = notes_root / ref
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError(f'wisdom reference is unavailable: {ref}')


def _consolidate_node(context, gate, runner, publisher):
    prompt_path = Path(__file__).resolve().parents[3] / 'prompts/global_memory_manager.md'
    schema_path = Path(__file__).resolve().parents[2] / 'schemas/memory.schema.json'
    prior = frozenset().union(*(read_processed_analyses(context.notes_root / name)
                                for name in _TARGETS.values()))
    covered = prior | gate.trigger_refs
    wisdom_covered = read_processed_analyses(context.notes_root / 'wisdom.md')
    # Bootstrap legacy summaries once. Afterwards only the trigger Node is
    # mandatory reading; retained provenance is cumulative, not a reread claim.
    required_read = gate.trigger_refs | (prior - wisdom_covered)
    role_prompt = with_shared_research_goal(
        prompt_path.read_text(encoding='utf-8'), prompt_path.parents[1],
    )
    prompt = role_prompt + '\n\n' + json.dumps({
        'trigger_node': gate.pending_nodes[0] if gate.pending_nodes else None,
        'trigger_refs': sorted(gate.trigger_refs), 'required_refs': sorted(required_read),
        'covered_refs': sorted(covered), **_TARGETS,
    }, sort_keys=True)
    node_id = gate.pending_nodes[0] if gate.pending_nodes else None
    sealed = context.coral_dir / 'private/memory_outputs' / f'{node_id}.json' if node_id else None
    document = None
    if sealed is not None and sealed.exists():
        cached = json.loads(sealed.read_text())
        candidate = cached.get('document', {})
        if (cached.get('version') == _KNOWLEDGE_VERSION
                and cached.get('trigger_refs') == sorted(gate.trigger_refs)
                and isinstance(candidate, dict) and set(candidate) == set(_TARGETS)
                and all(covered <= _parse_summary(candidate[kind]) for kind in _TARGETS)):
            document = candidate
    if document is None:
        with tempfile.TemporaryDirectory(prefix='autoresearch-memory-') as scratch_dir:
            scratch = Path(scratch_dir)
            staged_nodes = {ref.split('/')[1] for ref in required_read}
            read_roots = _stage_public_inputs(context, scratch, node_ids=staged_nodes,
                                             evidence_refs=required_read, analysis_refs=covered)
            output = runner(prompt, read_roots, schema_path, scratch)
            authored = json.loads(Path(output).read_text(encoding='utf-8'))
        if not isinstance(authored, dict) or set(authored) != set(_TARGETS):
            raise ValueError('Memory worker output requires research_summary, engineering_summary and wisdom')
        document = {kind: _with_coverage(authored[kind], covered) for kind in _TARGETS}
    for kind in _TARGETS:
        validate_summary(document[kind], context.notes_root, required_refs=covered)
    _validate_wisdom(document['wisdom'], context.notes_root, covered)
    if sealed is not None:
        with trusted_writes_lock(context.coral_dir):
            _atomic_json(sealed, {'version': _KNOWLEDGE_VERSION,
                'trigger_refs': sorted(gate.trigger_refs), 'document': document})
    publish = publisher or (lambda kind, content: _host_publish(context, kind, content))
    for kind in _TARGETS:
        publish(kind, document[kind])


def run_worker(coral_dir: str) -> dict[str, str]:
    """Fixed grader-owned one-shot entrypoint; never accepts Agent paths."""
    result = consolidate_memory(if_due=True, context=MemoryContext(Path(coral_dir)))
    return {"status": result.status}
