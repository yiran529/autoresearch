"""Analysis JSON is authoritative; Markdown is its deterministic reading view."""

import json
import re

import pytest

from coral.grader.analysis_artifacts import (
    convert_legacy_analysis,
    load_analysis,
    publish_analysis,
    render_analysis,
)


@pytest.fixture
def document():
    return {'node_id': 'N001', 'mode': 'research', 'research_sha': 'a' * 40,
            'summary': 'Measured result.', 'findings': ['[SUPPORTED] Evidence measured.'],
            'limitations': ['One batch only.'],
            'evaluation_snapshot': 'VERBOSE SNAPSHOT', 'evidence_manifest': {'hash': 'metadata'}}


def test_pair_preserves_complete_record_and_renders_only_reading_fields(tmp_path, document):
    path = tmp_path / 'N001/research_analysis_v1.md'
    publish_analysis(path, document)
    assert json.loads(path.with_suffix('.json').read_text()) == document
    text = path.read_text()
    assert text.startswith('# N001 Research Analysis · v1\n')
    assert '## Summary\n\nMeasured result.' in text
    assert '- [SUPPORTED] Evidence measured.' in text
    assert 'One batch only.' in text
    assert '[research_analysis_v1.json](research_analysis_v1.json)' in text
    assert 'VERBOSE SNAPSHOT' not in text
    assert 'evidence_manifest' not in text
    assert load_analysis(path) == document
    assert path.stat().st_mode & 0o444 == 0o444
    assert path.with_suffix('.json').stat().st_mode & 0o444 == 0o444
    assert text == render_analysis(document, path.name)


def test_pair_retry_repairs_missing_view_without_overwriting_json(tmp_path, document):
    path = tmp_path / 'N001/research_analysis_v1.md'
    publish_analysis(path, document)
    original = path.with_suffix('.json').read_bytes()
    path.unlink()
    publish_analysis(path, document)
    assert load_analysis(path) == document
    assert path.with_suffix('.json').read_bytes() == original
    with pytest.raises(ValueError, match='immutable'):
        publish_analysis(path, {**document, 'summary': 'Changed'})
    path.write_text('tampered view')
    with pytest.raises(ValueError, match='view'):
        load_analysis(path)


def test_legacy_json_markdown_remains_readable_and_unchanged(tmp_path, document):
    path = tmp_path / 'N001/research_analysis_v1.md'
    path.parent.mkdir()
    original = json.dumps(document)
    path.write_text(original)
    assert load_analysis(path) == document
    publish_analysis(path, document)
    assert path.read_text() == original
    assert load_analysis(path.with_suffix('.json')) == document


def test_pair_rejects_wrong_identity_and_symlink_sidecar(tmp_path, document):
    path = tmp_path / 'N002/research_analysis_v1.md'
    with pytest.raises(ValueError, match='identity'):
        publish_analysis(path, document)
    path = tmp_path / 'N001/research_analysis_v1.md'
    path.parent.mkdir()
    external = tmp_path / 'external'
    external.write_text('keep')
    path.with_suffix('.json').symlink_to(external)
    with pytest.raises(ValueError, match='symlink'):
        publish_analysis(path, document)
    assert external.read_text() == 'keep'


def test_explicit_conversion_preserves_original_and_is_idempotent(tmp_path, document):
    path = tmp_path / 'N001/research_analysis_v1.md'
    path.parent.mkdir()
    original = json.dumps(document, indent=4)
    path.write_text(original)
    backup = tmp_path / 'backup/original.md'
    convert_legacy_analysis(path, backup)
    assert backup.read_text() == original
    assert path.read_text() == render_analysis(document, path.name)
    assert load_analysis(path) == document
    convert_legacy_analysis(path, backup)
    assert backup.read_text() == original


def test_reading_view_does_not_link_disabled_diagnostic_directories(tmp_path, document):
    path = tmp_path / 'N001/research_analysis_v1.md'
    publish_analysis(path, document)
    links = re.findall(r'\]\(([^)]+)\)', path.read_text())
    assert all((path.parent / link).exists() for link in links)


def test_prior_markdown_navigation_remains_readable_without_rewriting(tmp_path, document):
    path = tmp_path / 'N001/research_analysis_v1.md'
    path.parent.mkdir()
    path.with_suffix('.json').write_text(json.dumps(document))
    original = (
        '# N001 Research Analysis · v1\n\nResearch SHA: `' + 'a' * 40 + '`\n\n'
        'Full record: [research_analysis_v1.json](research_analysis_v1.json)\n\n'
        '## Summary\n\nMeasured result.\n\n## Findings\n\n- [SUPPORTED] Evidence measured.\n\n'
        '## Limitations\n\n- One batch only.\n\n## Evidence navigation\n\n'
        '- [Diagnostic records (when enabled)](../../../diagnostics/N001-research-v1/records/)\n'
        '- The full JSON record preserves the frozen evaluation and evidence manifest.\n'
        '- Paths beginning `code/`, `evidence/` or `results/` in the text refer to '
        'the [Analyzer workspace](../../../diagnostics/N001-research-v1/workspace/). For diagnostic results, '
        'use the cited request ID under records/ for the immutable copy.\n'
    )
    path.write_text(original)
    assert load_analysis(path) == document
    publish_analysis(path, document)
    assert path.read_text() == original


def test_socket_publication_uses_only_fixed_sibling_and_rejects_direct_json(tmp_path, document):
    from types import SimpleNamespace

    from coral.grader.trusted_writes import AgentBinding, TrustedWriteError, TrustedWriteService

    service = TrustedWriteService(
        tmp_path / '.coral', SimpleNamespace(grader=SimpleNamespace(args={})),
        artifact_validator=lambda *_: {
            'relative_path': 'nodes/N001/research_analysis_v1.md',
            'format': 'analysis-json-markdown',
        })
    binding = AgentBinding('research-N001', 1, 1, str(tmp_path), None)
    payload = {'kind': 'analysis', 'node_id': 'N001', 'mode': 'research',
               'version': 1, 'content': json.dumps(document)}
    result = service.handle_authenticated('publish_artifact', payload, binding)
    path = tmp_path / '.coral/public/notes' / result['path']
    assert path.read_text() == render_analysis(document, path.name)
    assert load_analysis(path) == document
    service.handle_authenticated('publish_artifact', payload, binding)
    service.artifact_validator = lambda *_: {'relative_path': 'nodes/N001/research_analysis_v1.json'}
    with pytest.raises(TrustedWriteError, match='canonical'):
        service.handle_authenticated('publish_artifact', payload, binding)


