"""One user request = one IterationBudget across every gateway-internal continuation turn.

Each internal path that re-enters the agent must hand the ORIGINATING request's budget to the
turn instead of silently starting a fresh per-turn budget: async-delegation completion,
background-process notify_on_complete / watch / heartbeat events, internal events queued behind
a running turn, and a leftover /steer. Real user messages start a fresh request.
"""

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.iteration_budget import REQUEST_CHAIN_BUDGET_EVENT_KEY, IterationBudget
from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import (
    GatewayRunner,
    _request_chain_budget_for_followup,
    _request_chain_budget_from_event,
)
from gateway.session import SessionSource
from tests.gateway.test_background_process_notifications import _build_runner
from tests.gateway.test_duplicate_user_message import _bootstrap
from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter, _make_runner


def _event(*, internal: bool, budget: IterationBudget | None = None, text="continuation", source=None):
    metadata = {REQUEST_CHAIN_BUDGET_EVENT_KEY: budget} if budget is not None else {}
    return MessageEvent(
        text=text, message_type=MessageType.TEXT, source=source, internal=internal, metadata=metadata,
    )


# ── selection rules ──────────────────────────────────────────────────────────


def test_internal_event_uses_its_originating_request_budget():
    current, originating = IterationBudget(90), IterationBudget(90)
    event = _event(internal=True, budget=originating)

    assert _request_chain_budget_from_event(event) is originating
    assert _request_chain_budget_for_followup(current, event) is originating


def test_internal_event_without_budget_continues_the_current_request():
    current = IterationBudget(90)
    assert _request_chain_budget_for_followup(current, _event(internal=True)) is current


def test_real_user_message_starts_a_fresh_request_even_if_it_claims_a_budget():
    current = IterationBudget(90)
    forged = _event(internal=False, budget=IterationBudget(90))

    assert _request_chain_budget_from_event(forged) is None
    assert _request_chain_budget_for_followup(current, forged) is None


def test_leftover_steer_continues_but_interrupt_text_does_not():
    current = IterationBudget(90)
    assert _request_chain_budget_for_followup(current, None, leftover_steer=True) is current
    assert _request_chain_budget_for_followup(current, None, leftover_steer=False) is None


# ── producers → synthetic internal event ─────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("evt_type", ["async_delegation", "completion", "watch_match", "watch_disabled", "heartbeat"])
async def test_injected_internal_event_carries_the_producer_budget(monkeypatch, tmp_path, evt_type):
    runner = _build_runner(monkeypatch, tmp_path, "all")
    adapter = runner.adapters[Platform.TELEGRAM]
    session_key = "agent:main:telegram:dm:123"
    runner.session_store._entries[session_key] = SimpleNamespace(
        origin=SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", user_id="1"),
    )
    budget = IterationBudget(90)
    evt = {"type": evt_type, "session_id": "proc_x", "session_key": session_key,
           REQUEST_CHAIN_BUDGET_EVENT_KEY: budget}

    assert await runner._inject_watch_notification("[SYSTEM: continuation]", evt) is True

    synth_event = adapter.handle_message.await_args.args[0]
    assert synth_event.internal is True
    assert _request_chain_budget_from_event(synth_event) is budget


def test_process_watcher_completion_event_carries_the_spawning_request_budget():
    budget = IterationBudget(90)
    session = SimpleNamespace(
        command="make", output_buffer="done\n", exit_code=0, started_at=1.0, request_chain_budget=budget,
    )
    evt = GatewayRunner._build_process_completion_event(
        {"platform": "telegram", "chat_id": "123"}, session, "proc_x",
    )
    assert evt[REQUEST_CHAIN_BUDGET_EVENT_KEY] is budget


@pytest.mark.asyncio
@pytest.mark.parametrize("internal", [True, False])
async def test_idle_session_first_turn_budget(monkeypatch, tmp_path, internal):
    """First-turn path (session idle): an internal wake runs on its originating budget; a user
    message carrying the key does not (``None`` → run_sync opens a fresh request)."""
    runner = _bootstrap(monkeypatch, tmp_path)
    runner._run_agent = AsyncMock(return_value={"final_response": "ok", "messages": [], "history_offset": 0})
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group", user_id="12345")
    budget = IterationBudget(90)

    await runner._handle_message_with_agent(
        _event(internal=internal, budget=budget, source=source), source,
        "agent:main:telegram:group:-1001:12345", 1,
    )

    assert runner._run_agent.await_args.kwargs["request_chain_budget"] is (budget if internal else None)


# ── turn runner → agent, and the queued follow-up chain ──────────────────────


class _BudgetRecordingAgent:
    budgets: list = []
    results: list = []

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, iteration_budget=None, **kwargs):
        type(self).budgets.append(iteration_budget)
        if type(self).results:
            return type(self).results.pop(0)
        return {"final_response": f"answer to {message[:20]}", "messages": [], "api_calls": 1, "completed": True}


async def _run_chain(monkeypatch, tmp_path, *, pending=None, request_chain_budget=None, results=()):
    _BudgetRecordingAgent.budgets = []
    _BudgetRecordingAgent.results = list(results)
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _BudgetRecordingAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    adapter = ProgressCaptureAdapter(platform=Platform.TELEGRAM)
    runner = _make_runner(adapter)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group", thread_id="17585")
    session_key = "agent:main:telegram:group:-1001:17585"
    if pending is not None:
        pending.source = source
        adapter._pending_messages[session_key] = pending
    kwargs = {"request_chain_budget": request_chain_budget} if request_chain_budget is not None else {}
    await runner._run_agent(
        message="hello", context_prompt="", history=[], source=source, session_id="sess-chain",
        session_key=session_key, **kwargs,
    )
    return _BudgetRecordingAgent.budgets


@pytest.mark.asyncio
async def test_user_turn_opens_a_fresh_request_budget(monkeypatch, tmp_path):
    budgets = await _run_chain(monkeypatch, tmp_path)
    assert len(budgets) == 1
    assert isinstance(budgets[0], IterationBudget) and budgets[0].used == 0


@pytest.mark.asyncio
async def test_supplied_request_budget_reaches_the_agent(monkeypatch, tmp_path):
    budget = IterationBudget(90)
    assert await _run_chain(monkeypatch, tmp_path, request_chain_budget=budget) == [budget]


@pytest.mark.asyncio
async def test_queued_internal_event_without_budget_shares_the_current_request(monkeypatch, tmp_path):
    budgets = await _run_chain(monkeypatch, tmp_path, pending=_event(internal=True))
    assert len(budgets) == 2 and budgets[1] is budgets[0]


@pytest.mark.asyncio
async def test_queued_internal_event_uses_its_own_originating_request(monkeypatch, tmp_path):
    originating = IterationBudget(90)
    budgets = await _run_chain(monkeypatch, tmp_path, pending=_event(internal=True, budget=originating))
    assert len(budgets) == 2 and budgets[1] is originating


@pytest.mark.asyncio
async def test_queued_user_message_starts_a_fresh_request(monkeypatch, tmp_path):
    budgets = await _run_chain(monkeypatch, tmp_path, pending=_event(internal=False, text="new question"))
    assert len(budgets) == 2
    assert isinstance(budgets[1], IterationBudget) and budgets[1] is not budgets[0]


@pytest.mark.asyncio
async def test_leftover_steer_continues_the_current_request(monkeypatch, tmp_path):
    first = {"final_response": "done", "messages": [], "api_calls": 1, "completed": True,
             "pending_steer": "also check the logs"}
    budgets = await _run_chain(monkeypatch, tmp_path, results=[first])
    assert len(budgets) == 2 and budgets[1] is budgets[0]
