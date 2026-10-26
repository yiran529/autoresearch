"""One frozen candidate, fresh formal timing, and durable final quality tasks."""
from __future__ import annotations

import hashlib

from .runtime_artifacts import settings
from .staged_timing import timing_context, validate_result


def _component(host, parent, component, row):
    from greedylore_autoresearch_grader.gpu import InfraFailure
    state = host._read()
    attempts = state['final_evaluation']['components'].setdefault(component, [])
    latest = attempts[-1] if attempts else None
    if latest and latest['status'] == 'completed':
        return latest['result']
    if latest and latest['status'] == 'failed':
        if latest.get('result', {}).get('status') != 'infra_failed':
            return latest['result']
        if len(attempts) >= 1 + settings(host.store.coral_dir).get('research_infra_retry_limit', 2):
            return latest['result']
        latest = None
    if latest is None:
        canonical = host.validate_candidate(row['code_sha'])
        request = 'final-' + hashlib.sha256(f"{host.node_id}:{row['code_sha']}:{component}:{len(attempts)}".encode()).hexdigest()[:32]
        latest = {'kind': 'final_timing' if component == 'timing' else 'final_quality',
            'component': component, 'workload': None if component == 'timing' else component,
            'iteration': row['id'], 'sha': row['code_sha'], 'request_id': request,
            'context': parent['context'], 'canonical': canonical, 'status': 'queued',
            'timing_context': timing_context(host, canonical)}
        attempts.append(latest)
        host._save(state)
    if 'result' not in latest:
        recover = getattr(host.backend, 'recover_final', None)
        try:
            recovered = recover(latest) if recover else None
            if recovered is not None:
                latest['result'] = recovered
            elif latest['status'] == 'running':
                latest['result'] = {'status': 'infra_failed', 'reason': 'final component interrupted'}
            else:
                latest['status'] = 'running'
                host._save(state)
                latest['result'] = host.backend.run(host, latest)
        except Exception as failure_exc:
            failure = failure_exc
            try:
                recovered = recover(latest) if recover else None
            except Exception as recovery_error:
                recovered, failure = None, recovery_error
            latest['result'] = recovered or {'status': 'infra_failed' if isinstance(failure, (InfraFailure, OSError)) else 'failed',
                'reason': f'{type(failure).__name__}: {failure}', **host._failure_evidence(latest)}
        host._save(state)  # Persist completed work before any public projection.
    result = latest['result']
    if component == 'timing' and result['status'] == 'completed':
        try:
            validate_result(result, latest['timing_context'])
        except (KeyError, TypeError, ValueError) as exc:
            result = latest['result'] = {'status': 'failed', 'reason': f'invalid final timing: {exc}',
                'invalid_result': result, **host._failure_evidence(latest)}
            host._save(state)
    publisher = getattr(host.backend, 'publish_final', None)
    if publisher is None and component == 'timing':
        publisher = getattr(host.backend, 'publish_timing', None)
    if publisher:
        publisher(latest, result)
    if not latest.get('record_published'):
        host.record.final_observe(component, {'request_id': latest['request_id'], 'sha': latest['sha'],
            'result': result}, latest['request_id'] + '.final')
        latest['record_published'] = True
    latest['status'] = 'completed' if result['status'] == 'completed' else 'failed'
    host._save(state)
    return result


def finalize(host, job):
    document = host.record.read()
    selected = job['selected_iteration']
    if not document['frozen']:
        if not any(row['code_sha'] for row in document['iterations']):
            document = host.record.bind_unrun(job['sha'], job['request_id'] + '.unrun')
        document = host.record.freeze(selected, job['reason'], job['request_id'] + '.freeze')
    else:
        document = host.record.publish()
    state = host._read()
    state.setdefault('final_evaluation', {'components': {}, 'status': 'running'})
    host._save(state)
    if selected is None:
        aggregate = {'status': 'skipped', 'reason': 'no eligible development candidate'}
    else:
        row = document['iterations'][selected - 1]
        components = ['timing', *document['evaluation_stages']['final_quality']]
        results = {component: _component(host, job, component, row) for component in components}
        state = host._read()
        cap = 1 + settings(host.store.coral_dir).get('research_infra_retry_limit', 2)
        pending = [key for key, result in results.items() if result['status'] == 'infra_failed'
                   and len(state['final_evaluation']['components'][key]) < cap]
        if pending:
            state['jobs'][job['request_id']].update(status='failed', result={
                'status': 'infra_failed', 'reason': 'final components need bounded infrastructure recovery', 'components': pending})
            host._save(state)
            return
        status = ('completed' if all(r['status'] == 'completed' for r in results.values()) else
                  'infra_failed' if any(r['status'] == 'infra_failed' for r in results.values()) else 'failed')
        aggregate = {'status': status,
                     'components': results}
    state = host._read()
    state['final_evaluation']['status'] = aggregate['status']
    state['transfer'] = aggregate  # Compatibility lifecycle marker; no task-size semantics.
    state['transfer_terminal'] = True
    host._save(state)
    host.backend.release()
    host.backend.finish(host, selected)
    state['publication_complete'] = True
    state['jobs'][job['request_id']].update(status='completed', result=aggregate)
    host._save(state)
