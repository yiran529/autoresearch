"""Read-only, bounded projections of staged research progress."""

from coral.hub.attempts import read_attempt


def _small(value):
    if isinstance(value, str):
        return value[:256]
    return value if value is None or isinstance(value, (bool, int, float)) else None


def progress(host, state, document):
    rows = document.get('iterations', [])
    selected = document.get('selected_iteration')
    row = next((item for item in rows if item.get('id') == selected), None)
    development = []
    for item in rows[-64:]:
        observations = item.get('observations') or []
        latest = observations[-1] if observations else {}
        development.append({'iteration': item.get('id'), 'sha': _small(item.get('code_sha')),
                            'status': _small(item.get('status')),
                            'quality_pass': _small(latest.get('quality_pass', latest.get('eligible')))})
    final = state.get('final_evaluation') or {}
    components = final.get('components') or {}
    legacy = not final and bool(state.get('transfer_terminal')) and isinstance(state.get('transfer'), dict)
    if legacy:
        final = {'status': state['transfer'].get('status', 'unknown')}
    names = list(dict.fromkeys(['timing', *(document.get('evaluation_stages') or {}).get('final_quality', []), *components]))
    if legacy:
        names.append('transfer')
    stages = {}
    for name in names[:64]:
        attempts = components.get(name) or []
        latest = attempts[-1] if attempts else {}
        result = latest.get('result') or {}
        if legacy and name == 'transfer':
            result = state['transfer']
            latest = {'status': result.get('status', 'unknown')}
        elif final.get('status') == 'skipped' and not attempts:
            latest = {'status': 'skipped'}
        elif legacy and name == 'timing':
            timings = (row or {}).get('timings') or []
            if timings:
                result = timings[-1].get('result') or {}
                latest = {'status': result.get('status', 'unknown')}
            elif any(ob.get('timing') for ob in (row or {}).get('observations', [])):
                latest = {'status': 'completed'}
                result = {'status': 'completed'}
            else:
                latest = {'status': 'unknown'}
        stages[str(name)[:128]] = {'status': _small(latest.get('status', 'pending')),
                        'result_status': _small(result.get('status')),
                        'quality_pass': _small(result.get('quality_pass')),
                        'evidence_path': _small(result.get('evidence_path'))}
    publication = 'completed' if state.get('publication_complete') else ('pending' if state.get('transfer_terminal') else 'not_started')
    eligibility = 'unknown' if state.get('publication_complete') else 'pending'
    sha = row.get('code_sha') if row else None
    if sha and state.get('publication_complete'):
        attempt = read_attempt(host.store.coral_dir, sha)
        if attempt and attempt.agent_id == host.agent_id and attempt.metadata.get('staged_research'):
            value = attempt.metadata.get('performance_eligible')
            eligibility = ('eligible' if value else 'ineligible') if isinstance(value, bool) else 'unknown'
        else:
            eligibility = 'unknown'
    return {'development': development, 'selected_iteration': selected,
            'final': {'status': _small(final.get('status', 'pending')), 'components': stages},
            'publication': publication, 'formal_eligibility': eligibility}


def summary(state):
    keys = ('request_id', 'kind', 'status', 'sha', 'iteration', 'selected_iteration', 'record_published')
    jobs = state.get('jobs') or {}
    projected = {}
    for request_id, job in list(jobs.items())[-256:]:
        item = {key: _small(job[key]) for key in keys if key in job}
        result = job.get('result') or {}
        if isinstance(result, dict):
            item['result'] = {key: _small(result[key]) for key in ('status', 'evidence_path') if key in result}
        projected[str(request_id)[:128]] = item
    return {'version': state.get('version'), 'node_id': state.get('node_id'),
            'agent_id': state.get('agent_id'), 'jobs': projected, 'job_count': len(jobs),
            'transfer_terminal': state.get('transfer_terminal', False),
            'diagnostic_limits': state['diagnostic_limits'], 'progress': state['progress']}
