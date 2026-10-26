"""Timing identity, reuse and accounting, independent of training progress."""
from __future__ import annotations

import math

import yaml

from .runtime_artifacts import settings


def timing_context(host, canonical):
    from greedylore_autoresearch_grader.resources import ResourceOptions
    from greedylore_autoresearch_grader.timing_policy import plan_timing
    options = settings(host.store.coral_dir)
    contract = yaml.safe_load((host.store.coral_dir / 'private/eval_contract.yaml').read_text())
    efficiency = contract['efficiency']
    plan = plan_timing(canonical.get('timing_declaration', {'period_steps': contract['compressor']['update_gap']}),
        minimum=efficiency['measured_iterations'], maximum=efficiency.get('max_measured_iterations', 800),
        override=options.get('timing_measured_iterations'))
    return {'contract_hash': canonical.get('contract_hash'), 'plan': plan,
            'monitor_interval_seconds': ResourceOptions.from_args(options).monitor_interval}


def latest_timing(row, context=None):
    """Use the latest completed matching measurement, never the fastest one."""
    for entry in reversed(row.get('timings', [])):
        if (entry['sha'] == row['code_sha'] and entry['result']['status'] == 'completed'
                and (context is None or entry['context'] == context)):
            return entry
    # Already-completed staged records from before the independent timing action.
    for observation in reversed(row['observations']):
        bundle = observation.get('timing')
        if bundle and context is None:
            return {'request_id': observation.get('run_id'), 'sha': row['code_sha'],
                    'context': context, 'result': {'status': 'completed', 'bundle': bundle,
                    'performance_eligible': bundle['metadata'].get('performance_eligible', False)}}
    return None


def validate_result(result, context):
    """Validate cached measurements before they can enable freeze/transfer."""
    from coral.types import ScoreBundle
    if not isinstance(result, dict) or result.get('status') != 'completed':
        raise ValueError('timing result must be completed')
    bundle = ScoreBundle.from_dict(result['bundle'])
    metadata = bundle.metadata
    if metadata.get('contract_hash') != context['contract_hash']:
        raise ValueError('timing result contract mismatch')
    if type(result.get('performance_eligible')) is not bool or result['performance_eligible'] != metadata.get('performance_eligible'):
        raise ValueError('inconsistent timing eligibility')
    protocol = metadata.get('timing_protocol') or {}
    if any(protocol.get(key) != value for key, value in context['plan'].items()):
        raise ValueError('timing result protocol mismatch')
    for precision in ('bf16', 'fp32'):
        for metric in ('mean', 'spread'):
            value = bundle.scores[f'{precision}_timing_{metric}'].value
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (metric == 'mean' and value == 0):
                raise ValueError('invalid timing measurement')
    if result['performance_eligible'] and (
            metadata.get('failure_class') or metadata.get('comparability') not in {'comparable', 'cross_topology'}
            or not context['plan']['window_covers_period']):
        raise ValueError('timing is not performance eligible')
    return bundle


def eligible(row):
    if row['status'] != 'completed' or not row['observations']:
        return False
    observation = row['observations'][-1]
    if 'quality_pass' not in observation:  # Previously frozen legacy evidence.
        return bool(observation.get('eligible'))
    timing = latest_timing(row)
    return bool(observation['quality_pass'] and timing and timing['result'].get('performance_eligible'))


def check_budget(host, state, sha):
    options = settings(host.store.coral_dir)
    limits = [options.get('research_timing_success_limit', 5), options.get('research_timing_total_limit', 15)]
    if any(type(value) is not int or value <= 0 for value in limits):
        raise ValueError('research timing budget limits must be positive integers')
    jobs = [job for job in state['jobs'].values() if job['kind'] == 'timing' and not job.get('reused_from')]
    legacy = sum(bool(observation.get('timing')) for row in host.record.read()['iterations'] for observation in row['observations'])
    successful = legacy + sum(job.get('result', {}).get('status') == 'completed' for job in jobs)
    if successful >= limits[0] or legacy + len(jobs) >= limits[1]:
        raise ValueError('research timing budget exhausted')
    if any(job.get('sha') == sha and job.get('result', {}).get('status') == 'failed' for job in jobs):
        raise ValueError('candidate timing failed; fix it through a new design/SHA')
    failed = sum(job.get('sha') == sha and job.get('result', {}).get('status') == 'infra_failed' for job in jobs)
    if failed > options.get('research_infra_retry_limit', 2):
        raise ValueError('research timing infrastructure retry budget exhausted')
