"""Task-owned dynamic roster decoding without runtime-option injection."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from coral.agent.dynamic import DynamicAssignmentClient, prompt_for_assignment, spec_for_assignment
from coral.agent.manager import AgentManager
from coral.config import CoralConfig
from coral.template.coral_md import generate_coral_md
from coral.workspace.worktree import create_agent_worktree, retire_agent_worktree


def _config() -> CoralConfig:
    return CoralConfig.from_dict({
        "task": {"name": "dynamic", "description": "test"},
        "agents": {
            "runtime": "codex",
            "dynamic": {"enabled": True, "max_researchers": 2, "max_engineers": 1},
            "assignments": [
                {"role": "research", "runtime_options": {"role_file": "researcher.md"}},
                {"role": "engineering", "runtime_options": {"role_file": "engineer.md"}},
            ],
        },
    })


def test_dynamic_client_requests_empty_payload_and_builds_only_template_options(tmp_path):
    calls = []

    def request(coral_dir, operation, payload):
        calls.append((coral_dir, operation, payload))
        return {"assignments": [
            {"agent_id": "research-N001", "role": "research", "node_id": "N001", "status": "reserved"},
            {"agent_id": "engineer-E0001", "role": "engineering", "node_id": None, "status": "reserved"},
        ]}

    roster = DynamicAssignmentClient(tmp_path / ".coral", request=request).reconcile()
    assert calls == [(tmp_path / ".coral", "reconcile_workflow", {})]
    specs = [spec_for_assignment(_config(), item) for item in roster]
    assert [spec.agent_id for spec in specs] == ["research-N001", "engineer-E0001"]
    assert [spec.runtime_options["role_file"] for spec in specs] == [
        "researcher.md", "engineer.md",
    ]
    assert all("mounts" not in spec.runtime_options for spec in specs)


def test_dynamic_instruction_is_node_bound_and_omits_private_protocol():
    content = generate_coral_md(
        _config(), "research-N001", shared_dir=".codex",
        workflow_role="research", node_id="N001",
    )
    assert "N001" in content
    assert "autoresearch node initialize" in content
    assert ".codex/roles/research-N001.md" in content
    assert "trusted_writes.lock" not in content
    assert "SO_PEERCRED" not in content


@pytest.mark.parametrize("resuming", [False, True])
def test_research_instructions_keep_refinement_bounded_across_resume(tmp_path, resuming):
    assignment = DynamicAssignmentClient(
        tmp_path / ".coral", request=lambda *_: {"assignments": [{
            "agent_id": "research-N001", "role": "research",
            "node_id": "N001", "status": "active",
        }]},
    ).reconcile()[0]
    spec = spec_for_assignment(_config(), assignment)
    prompt = prompt_for_assignment(
        spec, instruction_file="AGENTS.md", shared_dir=".codex", resuming=resuming,
    )
    instructions = generate_coral_md(
        _config(), spec.agent_id, workflow_role="research", node_id=spec.node_id,
    )
    for content in (prompt, instructions):
        assert "3 successful scored formal Research runs" in content
        assert "10 total formal Research runs" in content
        assert "archived Attempts" in content
        assert "first terminal result" in content
        assert "autoresearch node complete --mode research" in content
        assert "all Attempts" in content
        assert "quality-gated Engineering" in content
        assert "one formal Research evaluation" not in content
        assert "one distinct formal Research candidate" not in content


def test_dynamic_score_feedback_does_not_request_open_ended_iteration():
    manager = AgentManager(_config())
    prompt = manager._build_score_prompt({
        "agent_id": "research-N001", "commit_hash": "a" * 40, "score": 0.5,
        "feedback": "Recorded quality result.", "title": "Research candidate",
    }, 1)
    assert "Recorded quality result." in prompt
    assert "does not reset" in prompt
    assert "Do NOT exit" not in prompt
    assert "sweep variants" not in prompt


@pytest.mark.parametrize("assignment", [
    {"agent_id": "research-N001", "role": "research", "node_id": "N002", "status": "active"},
    {"agent_id": "engineer-E0001", "role": "research", "node_id": "N001", "status": "active"},
    {"agent_id": "research-N001", "role": "research", "node_id": "N001", "status": "done"},
    {"agent_id": "research-N001", "role": "research", "node_id": "N001", "status": "active", "runtime_options": {"sandbox": "off"}},
])
def test_dynamic_client_rejects_malformed_roster(tmp_path, assignment):
    client = DynamicAssignmentClient(
        tmp_path / ".coral",
        request=lambda *_: {"assignments": [assignment]},
    )
    with pytest.raises(ValueError, match="assignment"):
        client.reconcile()


def test_manager_reconciles_dynamic_roster_without_restarting_survivors(tmp_path, monkeypatch):
    manager = AgentManager(_config())
    manager.paths = SimpleNamespace(
        coral_dir=tmp_path / ".coral", repo_dir=tmp_path / "repo",
        agents_dir=tmp_path / "agents",
    )
    desired = [
        {"agent_id": "research-N001", "role": "research", "node_id": "N001", "status": "active"},
        {"agent_id": "research-N002", "role": "research", "node_id": "N002", "status": "reserved"},
    ]
    manager._dynamic_client = DynamicAssignmentClient(
        manager.paths.coral_dir,
        request=lambda *_: {"assignments": desired},
    )
    starts = []
    registry = []

    class Handle:
        def __init__(self, agent_id):
            self.agent_id = agent_id
            self.alive = True
            self.stopped = False

        def stop(self):
            self.stopped = True

    monkeypatch.setattr(manager, "_setup_and_start_agent", lambda agent_id, **kwargs: (
        starts.append(agent_id) or Handle(agent_id)
    ))
    monkeypatch.setattr(manager, "_write_trusted_agent_registry", lambda: registry.append(
        tuple(handle.agent_id for handle in manager.handles)
    ))
    monkeypatch.setattr(manager, "_write_agent_pids", lambda: None)
    monkeypatch.setattr(manager, "_persist_agent_state", lambda: None)

    manager._reconcile_dynamic_agents()
    assert starts == ["research-N001", "research-N002"]
    assert registry == [("research-N001",), ("research-N001", "research-N002")]
    first = manager.handles[0]
    desired[:] = [
        {"agent_id": "research-N002", "role": "research", "node_id": "N002", "status": "active"},
        {"agent_id": "engineer-E0001", "role": "engineering", "node_id": None, "status": "reserved"},
    ]
    monkeypatch.setattr("coral.agent.manager.retire_agent_worktree", lambda *args: True)
    manager._reconcile_dynamic_agents()
    assert first.stopped
    assert starts == ["research-N001", "research-N002", "engineer-E0001"]
    assert [handle.agent_id for handle in manager.handles] == [
        "research-N002", "engineer-E0001",
    ]
    engineer_handle = manager.handles[1]
    desired[1] = {
        "agent_id": "engineer-E0001", "role": "engineering",
        "node_id": "N001", "status": "active",
    }
    manager._reconcile_dynamic_agents()
    assert manager.handles[1] is engineer_handle
    assert manager.specs_by_id["engineer-E0001"].node_id == "N001"
    # Finalizing keeps its slot/worktree but must not restart the producer session.
    retired = []
    monkeypatch.setattr("coral.agent.manager.retire_agent_worktree",
                        lambda *args: retired.append(args) or True)
    desired[1]["status"] = "finalizing"
    manager._reconcile_dynamic_agents()
    assert engineer_handle.stopped
    assert [h.agent_id for h in manager.handles] == ["research-N002"]
    assert retired == []
    manager._reconcile_dynamic_agents()
    assert starts.count("engineer-E0001") == 1
    desired.pop()
    manager._reconcile_dynamic_agents()
    assert len(retired) == 1
    assert "engineer-E0001" not in manager.specs_by_id


def test_retire_only_removes_clean_worktree_and_preserves_branch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args, cwd=repo):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                              text=True, check=True).stdout.strip()

    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (repo / "model.txt").write_text("baseline\n")
    git("add", "model.txt")
    git("commit", "-m", "baseline")
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    clean = create_agent_worktree(repo, "research-N001", agents_dir)
    (repo / ".git" / "info" / "exclude").write_text(".venv/\n")
    (clean / ".venv").mkdir()
    (clean / ".venv" / "generated.txt").write_text("cache")
    assert retire_agent_worktree(repo, clean)
    assert not clean.exists()
    assert git("show-ref", "--verify", "refs/heads/coral/research-N001")

    dirty = create_agent_worktree(repo, "research-N002", agents_dir)
    (dirty / "uncommitted.txt").write_text("keep me")
    assert not retire_agent_worktree(repo, dirty)
    assert dirty.exists()


def test_waiting_node_keeps_worktree_and_resumes_its_saved_session(tmp_path, monkeypatch):
    manager = AgentManager(_config())
    coral_dir = tmp_path / '.coral'
    logs = coral_dir / 'public/logs'
    logs.mkdir(parents=True)
    worktree = tmp_path / 'agents/research-N003'
    worktree.mkdir(parents=True)
    (worktree / 'candidate.txt').write_text('preserve candidate')
    session = '019fbea9-d7c0-7870-9606-cf1100d303e8'
    (logs / 'research-N003.0.log').write_text(
        '{"type":"thread.started","thread_id":"' + session + '"}\n')
    manager.paths = SimpleNamespace(coral_dir=coral_dir, repo_dir=tmp_path / 'repo',
                                    agents_dir=tmp_path / 'agents')
    assignment = {'agent_id': 'research-N003', 'role': 'research',
                  'node_id': 'N003', 'status': 'waiting'}
    manager._dynamic_client = DynamicAssignmentClient(
        coral_dir, request=lambda *_: {'assignments': [dict(assignment)]})
    launches = []
    monkeypatch.setattr(manager, '_setup_and_start_agent', lambda aid, **kw: (
        launches.append((aid, kw.get('resume_session_id'))) or
        SimpleNamespace(agent_id=aid)))
    monkeypatch.setattr(manager, '_write_trusted_agent_registry', lambda: None)
    monkeypatch.setattr(manager, '_write_agent_pids', lambda: None)
    monkeypatch.setattr(manager, '_persist_agent_state', lambda: None)
    manager._reconcile_dynamic_agents()
    manager._reconcile_dynamic_agents()
    assert manager.handles == []
    assert (worktree / 'candidate.txt').read_text() == 'preserve candidate'
    assignment['status'] = 'active'
    manager._reconcile_dynamic_agents()
    assert launches == [('research-N003', session)]


def test_crashed_dynamic_agent_restarts_same_identity_and_session(tmp_path, monkeypatch):
    manager = AgentManager(_config())
    assignment = DynamicAssignmentClient(
        tmp_path / ".coral",
        request=lambda *_: {"assignments": [{
            "agent_id": "research-N001", "role": "research",
            "node_id": "N001", "status": "active",
        }]},
    ).reconcile()[0]
    spec = spec_for_assignment(manager.config, assignment)
    manager.specs_by_id[spec.agent_id] = spec
    manager.runtimes[spec.agent_id] = manager.runtime
    log = tmp_path / "agent.log"
    session_id = "019fbea9-d7c0-7870-9606-cf1100d303e8"
    log.write_text('{"type":"start","session_id":"' + session_id + '"}\n')
    old = SimpleNamespace(agent_id=spec.agent_id, log_path=log, stop=lambda: None)
    manager.handles = [old]
    calls = []
    monkeypatch.setattr(manager, "_setup_and_start_agent", lambda agent_id, **kwargs: (
        calls.append((agent_id, kwargs["resume_session_id"])) or old
    ))
    manager._restart_agent(0)
    assert calls == [("research-N001", session_id)]


def test_dynamic_start_waits_for_trusted_endpoint_before_reconcile(tmp_path, monkeypatch):
    manager = AgentManager(_config())
    monkeypatch.setattr("coral.agent.manager.atexit.register", lambda *_: None)
    paths = SimpleNamespace(
        coral_dir=tmp_path / ".coral", repo_dir=tmp_path / "repo",
        run_dir=tmp_path, agents_dir=tmp_path / "agents",
    )
    paths.coral_dir.mkdir()
    events = []
    monkeypatch.setattr("coral.agent.manager.create_project", lambda *args, **kwargs: paths)
    monkeypatch.setattr(manager, "_start_gateway_if_enabled", lambda: None)
    monkeypatch.setattr(manager, "_start_sandbox_if_enabled", lambda: None)
    monkeypatch.setattr(manager, "_start_grader_daemon", lambda: events.append("grader"))
    monkeypatch.setattr(manager, "_reconcile_dynamic_agents", lambda: events.append("reconcile"))
    monkeypatch.setattr(manager, "_write_pid_file", lambda: None)
    monkeypatch.setattr(manager, "_persist_agent_state", lambda: None)
    assert manager.start_all() == []
    assert events == ["grader", "reconcile"]


def test_dynamic_resume_uses_saved_sessions_not_old_worktree_roster(tmp_path, monkeypatch):
    manager = AgentManager(_config())
    monkeypatch.setattr("coral.agent.manager.atexit.register", lambda *_: None)
    paths = SimpleNamespace(
        coral_dir=tmp_path / ".coral", repo_dir=tmp_path / "repo",
        run_dir=tmp_path, agents_dir=tmp_path / "agents",
    )
    paths.coral_dir.mkdir()
    events = []
    monkeypatch.setattr(manager, "_start_gateway_if_enabled", lambda: None)
    monkeypatch.setattr(manager, "_start_sandbox_if_enabled", lambda: None)
    monkeypatch.setattr(manager, "_start_grader_daemon", lambda: events.append("grader"))
    monkeypatch.setattr(manager, "_kill_old_agent_processes", lambda: None)
    monkeypatch.setattr(manager, "_load_saved_sessions", lambda: {"research-N001": "session-1"})
    monkeypatch.setattr("coral.agent.manager._validate_sessions", lambda *args, **kwargs: {
        "research-N001": "session-1",
    })
    monkeypatch.setattr(manager, "_reconcile_dynamic_agents", lambda resume_sessions=None: (
        events.append(("reconcile", resume_sessions))
    ))
    monkeypatch.setattr(manager, "_write_pid_file", lambda: None)
    monkeypatch.setattr(manager, "_persist_agent_state", lambda: None)
    assert manager.resume_all(paths) == []
    assert events == ["grader", ("reconcile", {"research-N001": "session-1"})]


def test_trusted_research_mode_reaches_spec_and_generated_instructions(tmp_path):
    roster = DynamicAssignmentClient(tmp_path / '.coral', request=lambda *_: {'assignments': [{
        'agent_id': 'research-N001', 'role': 'research', 'node_id': 'N001', 'status': 'reserved',
        'research_mode': 'EXPLORE',
    }]}).reconcile()
    spec = spec_for_assignment(_config(), roster[0])
    assert spec.research_mode == 'EXPLORE'
    content = generate_coral_md(_config(), spec.agent_id, workflow_role=spec.workflow_role,
                                node_id=spec.node_id, research_mode=spec.research_mode)
    assert 'EXPLORE' in content
    assert 'pinned' in content


@pytest.mark.parametrize('mode', ['explore', 'UNBOUNDED', None, {'runtime_options': {}}])
def test_dynamic_roster_rejects_invalid_research_mode(tmp_path, mode):
    client = DynamicAssignmentClient(tmp_path / '.coral', request=lambda *_: {'assignments': [{
        'agent_id': 'research-N001', 'role': 'research', 'node_id': 'N001', 'status': 'reserved',
        'research_mode': mode,
    }]})
    with pytest.raises(ValueError, match='assignment'):
        client.reconcile()


@pytest.mark.parametrize('resuming', [False, True])
def test_staged_startup_delegates_protocol_to_reachable_role_and_workflow(resuming):
    config = _config()
    config.grader.args['staged_research'] = True
    assignment = DynamicAssignmentClient(
        Path('/tmp/unused-coral'), request=lambda *_: {'assignments': [{
            'agent_id': 'research-N001', 'role': 'research', 'node_id': 'N001', 'status': 'active',
        }]},
    ).reconcile()[0]
    spec = spec_for_assignment(config, assignment)
    startup = prompt_for_assignment(spec, instruction_file='AGENTS.md', shared_dir='.codex',
                                    resuming=resuming, staged_research=True)
    instructions = generate_coral_md(config, spec.agent_id, shared_dir='.codex',
                                     workflow_role='research', node_id=spec.node_id)
    assert 'autoresearch research status --node N001' in startup
    assert 'autoresearch research wait --node N001 --request-id <saved-id>' in startup
    for content in (startup, instructions):
        assert '.codex/roles/research-N001.md' in content
        assert re.search(r'[\u4e00-\u9fff]', content) is None
        assert 'at most 3 candidate training trajectories' not in content
        assert "role's numbered steps" not in content
        assert '3 successful scored formal Research runs' not in content
        assert '10 total formal Research runs' not in content
        assert 'autoresearch node complete --mode research' not in content
    assert '.codex/notes/toolkit/workflow_rules.md' in instructions
    assert '.codex/notes/toolkit/workflow_research.md' in instructions
    assert '.codex/notes/nodes/N001/README.md' in instructions
    assert 'REFINE' in instructions
    assert 'AGENTS.md' in startup


def test_staged_role_selection_uses_only_trusted_config_and_consumes_override():
    config = _config()
    config.agents.assignments[0].runtime_options.update(
        role_file='researcher_legacy.md', staged_role_file='researcher.md')
    from coral.agent.dynamic import DynamicAssignment
    assignment = DynamicAssignment('research-N001', 'research', 'N001', 'active')
    assert spec_for_assignment(config, assignment).runtime_options['role_file'] == 'researcher_legacy.md'
    config.grader.args['staged_research'] = True
    options = spec_for_assignment(config, assignment).runtime_options
    assert options['role_file'] == 'researcher.md'
    assert 'staged_role_file' not in options


@pytest.mark.parametrize('mode,expected', [(None, 'REFINE'), ('REFINE', 'REFINE'), ('EXPLORE', 'EXPLORE')])
def test_staged_mode_and_feedback_survive_resume(mode, expected):
    from coral.agent.dynamic import DynamicAssignment

    config = _config()
    config.grader.args['staged_research'] = True
    instructions = generate_coral_md(config, 'research-N001', shared_dir='.codex',
                                     workflow_role='research', node_id='N001', research_mode=mode)
    assert f'`{expected}`' in instructions
    spec = spec_for_assignment(config, DynamicAssignment('research-N001', 'research', 'N001', 'active', expected))
    startup = prompt_for_assignment(spec, instruction_file='AGENTS.md', shared_dir='.codex',
                                    resuming=True, staged_research=True, feedback='Preserve saved request probe-1.')
    assert 'Preserve saved request probe-1.' in startup
    assert 'N001' in startup and 'AGENTS.md' in startup
    assert 'autoresearch research wait --node N001 --request-id <saved-id>' in startup


@pytest.mark.parametrize('mode', ['explore', 'UNBOUNDED'])
def test_staged_generated_instructions_reject_invalid_modes(mode):
    config = _config()
    config.grader.args['staged_research'] = True
    with pytest.raises(ValueError, match='invalid trusted research mode'):
        generate_coral_md(config, 'research-N001', workflow_role='research', node_id='N001', research_mode=mode)


@pytest.mark.parametrize('resuming', [False, True])
def test_engineer_startup_and_generated_instructions_use_managed_timing(resuming):
    from coral.agent.dynamic import DynamicAssignment

    config = _config()
    config.grader.args['staged_research'] = True
    spec = spec_for_assignment(config, DynamicAssignment('engineer-E0001', 'engineering', 'N001', 'active'))
    startup = prompt_for_assignment(spec, instruction_file='AGENTS.md', shared_dir='.codex',
                                    resuming=resuming, staged_research=True)
    instructions = generate_coral_md(config, spec.agent_id, shared_dir='.codex',
                                     workflow_role='engineering', node_id=spec.node_id)
    for content in (startup, instructions):
        assert '.codex/roles/engineer-E0001.md' in content
        assert re.search(r'[\u4e00-\u9fff]', content) is None
        assert "role's numbered steps" not in content
    assert 'autoresearch engineering claim N001' in instructions
    assert 'autoresearch engineering timing' in instructions
    assert '.codex/notes/toolkit/workflow_rules.md' in instructions
    assert 'workflow_research.md' not in instructions


@pytest.mark.parametrize(('role', 'agent_id'), [
    ('research', 'research-N001'), ('engineering', 'engineer-E0001'),
])
def test_staged_instructions_include_current_task_shared_goal(tmp_path, role, agent_id):
    config = _config()
    config.grader.args['staged_research'] = True
    config.task_dir = tmp_path
    prompt = tmp_path / 'prompts/shared_research_goal.md'
    prompt.parent.mkdir()
    prompt.write_text('# Shared Research Goal\n\nTask-specific cumulative research objective.\n')

    first = generate_coral_md(config, agent_id, workflow_role=role, node_id='N001')
    assert first.startswith('# dynamic: ' + agent_id + '\n')
    assert first.count('Task-specific cumulative research objective.') == 1
    assert '## Shared Research Goal\n' in first
    assert '## Current assignment\n' in first
    assert ('workflow_research.md' in first) == (role == 'research')

    prompt.write_text('# Shared Research Goal\n\nRevised objective for the next session.\n')
    second = generate_coral_md(config, agent_id, workflow_role=role, node_id='N001')
    assert 'Revised objective for the next session.' in second
    assert 'Task-specific cumulative research objective.' not in second


def test_legacy_instructions_do_not_load_staged_shared_goal(tmp_path):
    config = _config()
    config.task_dir = tmp_path
    prompt = tmp_path / 'prompts/shared_research_goal.md'
    prompt.parent.mkdir()
    prompt.write_text('# Shared Research Goal\n\nStaged-only objective.\n')
    content = generate_coral_md(config, 'research-N001', workflow_role='research', node_id='N001')
    assert 'Staged-only objective.' not in content


@pytest.mark.parametrize('shared_content', [None, '', ' \n'])
def test_staged_instructions_preserve_body_without_shared_goal(tmp_path, shared_content):
    config = _config()
    config.grader.args['staged_research'] = True
    original = generate_coral_md(config, 'research-N001', workflow_role='research', node_id='N001')
    config.task_dir = tmp_path
    if shared_content is not None:
        prompt = tmp_path / 'prompts/shared_research_goal.md'
        prompt.parent.mkdir()
        prompt.write_text(shared_content)
    assert generate_coral_md(
        config, 'research-N001', workflow_role='research', node_id='N001',
    ) == original


@pytest.mark.parametrize(('role', 'agent_id'), [
    ('research', 'research-N001'), ('engineering', 'engineer-E0001'),
])
def test_manager_delivers_shared_goal_from_task_directory(tmp_path, monkeypatch, role, agent_id):
    from coral.agent.dynamic import DynamicAssignment

    config = _config()
    config.grader.args['staged_research'] = True
    config.agents.heartbeat_enabled = False
    assert config.task_dir is None
    task_dir = tmp_path / 'task'
    prompt = task_dir / 'prompts/shared_research_goal.md'
    prompt.parent.mkdir(parents=True)
    prompt.write_text('# Shared Research Goal\n\nManager-resolved objective.\n')
    (task_dir / 'researcher.md').write_text('Research role.\n')
    (task_dir / 'engineer.md').write_text('Engineering role.\n')
    manager = AgentManager(config, config_dir=task_dir)
    spec = spec_for_assignment(config, DynamicAssignment(agent_id, role, 'N001', 'active'))
    manager.specs_by_id[agent_id] = spec
    manager.runtimes[agent_id] = manager.runtime
    worktree = tmp_path / 'agents' / agent_id
    worktree.mkdir(parents=True)
    (worktree / '.coral_agent_id').write_text(agent_id)
    (worktree / 'candidate.txt').write_text('Keep saved progress.\n')
    manager.paths = SimpleNamespace(coral_dir=tmp_path / '.coral', agents_dir=worktree.parent,
                                    repo_dir=tmp_path / 'repo')
    delivered = []

    def start(**kwargs):
        delivered.append(kwargs['coral_md_path'].read_text())
        return SimpleNamespace(agent_id=agent_id)

    monkeypatch.setattr(manager.runtime, 'start', start)
    manager._setup_and_start_agent(agent_id)
    assert delivered[0].startswith('# dynamic: ' + agent_id + '\n')
    assert delivered[0].count('Manager-resolved objective.') == 1

    prompt.write_text('# Shared Research Goal\n\nUpdated objective.\n')
    manager._setup_and_start_agent(agent_id, resume_session_id='saved-session')
    assert delivered[1].count('Updated objective.') == 1
    assert 'Manager-resolved objective.' not in delivered[1]
    assert (worktree / 'candidate.txt').read_text() == 'Keep saved progress.\n'
