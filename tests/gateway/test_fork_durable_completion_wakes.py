"""Fork completion wakes on v0.22.5's durable admission: the request-chain budget and the
same-turn completion drop survive a FIFO that persists only its own metadata."""
from types import SimpleNamespace

import pytest

from agent.iteration_budget import REQUEST_CHAIN_BUDGET_EVENT_KEY, IterationBudget
from hermes_state_runtime import list_session_admissions
from tests.gateway.test_automation_notification_category import _native_owner

_DONE = {'final_response': 'answer', 'completed': True}


def _completion(metadata, process_id, **extra):
    return {'type': 'completion', 'session_id': process_id, 'session_key': metadata['gateway_session_key'],
            'platform': 'discord', 'chat_id': '42', 'chat_type': 'dm', 'exit_code': 0, 'output': 'done',
            'started_at': 1.0, 'parent_session_id': metadata['gateway_session_id'], **extra}


def _automation_rows(authority, metadata, *, pending_only=True):
    owner = authority.logical_owner(metadata['gateway_session_id'])
    return [row for row in list_session_admissions(authority.db, session_id=owner, pending_only=pending_only)
            if row['principal_id'].startswith('automation:')]


@pytest.mark.asyncio
async def test_completion_with_request_budget_is_admitted_and_its_turn_finds_the_budget(tmp_path, monkeypatch):
    from gateway.run import _request_chain_budget_from_event
    from gateway.session_envelope import restore_native
    runner, authority, _adapter, _source, metadata = await _native_owner(tmp_path, monkeypatch, 'human')
    try:
        budget = IterationBudget(10)
        evt = _completion(metadata, 'proc_budget', **{REQUEST_CHAIN_BUDGET_EVENT_KEY: budget})
        assert await runner._inject_watch_notification('[done]', evt, raise_not_accepted=True) is True
        [row] = _automation_rows(authority, metadata)
        restored = restore_native(row['payload'], runner)
        assert _request_chain_budget_from_event(restored) is budget
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_drain_withdraws_a_queued_completion_the_turn_took_via_wait(tmp_path, monkeypatch):
    from gateway.session_ingress import executing_admission
    from tools.process_registry import process_registry
    runner, authority, adapter, source, metadata = await _native_owner(tmp_path, monkeypatch, 'human')
    try:
        for process_id in ('proc_taken', 'proc_other'):
            evt = _completion(metadata, process_id)
            assert await runner._inject_watch_notification('[done]', evt, raise_not_accepted=True) is True
        monkeypatch.setattr(process_registry, 'is_completion_consumed', lambda pid: pid == 'proc_taken')
        token = executing_admission.set(True)
        try:
            await runner._run_agent_drain_pending(dict(_DONE), adapter, source, metadata['gateway_session_key'])
        finally:
            executing_admission.reset(token)
        assert [row['request_id'] for row in _automation_rows(authority, metadata)] == ['["completion","proc_other",1.0]']
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('result, current_message_id, withdrawn', [
    (_DONE, 'human', True),
    (_DONE, 'later-message', False),
    ({'final_response': '', 'completed': True}, 'human', False),
    ({**_DONE, 'failed': True}, 'human', False),
])
async def test_polled_completion_is_withdrawn_only_for_its_own_successful_turn(
        tmp_path, monkeypatch, result, current_message_id, withdrawn):
    from gateway.session_automation import withdraw_covered_completions
    from tools.process_registry import process_registry
    runner, authority, _adapter, _source, metadata = await _native_owner(tmp_path, monkeypatch, 'human')
    try:
        assert await runner._inject_watch_notification(
            '[done]', _completion(metadata, 'proc_polled'), raise_not_accepted=True) is True
        monkeypatch.setattr(process_registry, 'is_completion_consumed', lambda pid: False)
        monkeypatch.setattr(process_registry, 'was_completion_polled', lambda pid: pid == 'proc_polled')
        monkeypatch.setattr(process_registry, 'get', lambda pid: SimpleNamespace(watcher_message_id='human'))
        await withdraw_covered_completions(runner, metadata['gateway_session_key'], dict(result), current_message_id)
        assert (not _automation_rows(authority, metadata)) is withdrawn
    finally:
        authority.db.close()
