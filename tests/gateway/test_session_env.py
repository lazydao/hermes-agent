import asyncio
import os
from contextlib import nullcontext

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from gateway.session_context import (
    get_session_env,
    set_session_vars,
    clear_session_vars,
    reset_session_vars,
    _VAR_MAP,
    _UNSET,
)


@pytest.fixture(autouse=True)
def _reset_contextvars():
    """Reset all session contextvars to _UNSET between tests.

    In production each asyncio.Task gets a fresh context copy where the
    defaults are _UNSET.  In tests all functions share the same thread
    context, so a clear_session_vars() from test A (which sets vars to "")
    would leak into test B.  This fixture ensures each test starts clean.
    """
    yield
    for var in _VAR_MAP.values():
        # Can't use var.reset() without a token; just set back to sentinel.
        var.set(_UNSET)


def test_set_session_env_sets_contextvars(monkeypatch):
    """_set_session_env should populate contextvars, not os.environ."""
    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_name="Group",
        chat_type="group",
        user_id="123456",
        user_name="alice",
        thread_id="17585",
    )
    context = SessionContext(source=source, connected_platforms=[], home_channels={})

    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_TYPE", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)

    tokens = runner._set_session_env(context)

    # Values should be readable via get_session_env (contextvar path)
    assert get_session_env("HERMES_SESSION_PLATFORM") == "telegram"
    assert get_session_env("HERMES_SESSION_SOURCE") == ""
    assert get_session_env("HERMES_SESSION_CHAT_ID") == "-1001"
    assert get_session_env("HERMES_SESSION_CHAT_NAME") == "Group"
    assert get_session_env("HERMES_SESSION_CHAT_TYPE") == "group"
    assert get_session_env("HERMES_SESSION_USER_ID") == "123456"
    assert get_session_env("HERMES_SESSION_USER_NAME") == "alice"
    assert get_session_env("HERMES_SESSION_THREAD_ID") == "17585"

    # os.environ should NOT be touched
    assert os.getenv("HERMES_SESSION_PLATFORM") is None
    assert os.getenv("HERMES_SESSION_SOURCE") is None
    assert os.getenv("HERMES_SESSION_CHAT_TYPE") is None
    assert os.getenv("HERMES_SESSION_THREAD_ID") is None

    # Clean up
    runner._clear_session_env(tokens)


def test_clear_session_env_restores_previous_state(monkeypatch):
    """_clear_session_env should restore contextvars to their pre-handler values."""
    runner = object.__new__(GatewayRunner)

    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_TYPE", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_name="Group",
        chat_type="group",
        user_id="123456",
        user_name="alice",
        thread_id="17585",
    )
    context = SessionContext(source=source, connected_platforms=[], home_channels={})

    tokens = runner._set_session_env(context)
    assert get_session_env("HERMES_SESSION_PLATFORM") == "telegram"
    assert get_session_env("HERMES_SESSION_USER_ID") == "123456"
    assert get_session_env("HERMES_SESSION_CHAT_TYPE") == "group"

    runner._clear_session_env(tokens)

    # After clear, contextvars should return to defaults (empty)
    assert get_session_env("HERMES_SESSION_PLATFORM") == ""
    assert get_session_env("HERMES_SESSION_CHAT_ID") == ""
    assert get_session_env("HERMES_SESSION_CHAT_NAME") == ""
    assert get_session_env("HERMES_SESSION_CHAT_TYPE") == ""
    assert get_session_env("HERMES_SESSION_USER_ID") == ""
    assert get_session_env("HERMES_SESSION_USER_NAME") == ""
    assert get_session_env("HERMES_SESSION_THREAD_ID") == ""


def _feishu_source(message_id=None):
    return SessionSource(
        platform=Platform.FEISHU, chat_id="oc_chat", chat_type="dm", user_id="ou_user", message_id=message_id,
    )


@pytest.mark.asyncio
async def test_run_agent_binds_each_turns_message_id_and_restores_the_outer_one(monkeypatch):
    """A queued follow-up recurses through ``_run_agent`` with its own source: it must run under
    ITS triggering message id (the reply anchor), not the first message's, and the outer value
    must be restored afterwards."""
    runner = object.__new__(GatewayRunner)
    observed = []

    async def fake_run_agent_inner(message, *args, **kwargs):
        observed.append((message, get_session_env("HERMES_SESSION_MESSAGE_ID")))
        if message == "first":
            await runner._run_agent("queued", "", [], _feishu_source("om_second"), "session-1")
            observed.append(("first-after", get_session_env("HERMES_SESSION_MESSAGE_ID")))
        return {"final_response": "ok"}

    monkeypatch.setattr(runner, "_run_agent_inner", fake_run_agent_inner)
    monkeypatch.setattr(runner, "_profile_scope_for_source", lambda _source: nullcontext())
    tokens = set_session_vars(message_id="outer-message")
    try:
        await runner._run_agent("first", "", [], _feishu_source("om_first"), "session-1")
        assert get_session_env("HERMES_SESSION_MESSAGE_ID") == "outer-message"
        # A source without a message id (steer/interrupt text re-uses the outer source; an
        # internal wake may carry none) binds exactly what _set_session_env would: "".
        await runner._run_agent("anchorless", "", [], _feishu_source(), "session-1")
    finally:
        clear_session_vars(tokens)

    assert observed == [
        ("first", "om_first"), ("queued", "om_second"), ("first-after", "om_first"), ("anchorless", ""),
    ]


@pytest.mark.asyncio
async def test_concurrent_run_agent_message_ids_are_isolated(monkeypatch):
    runner = object.__new__(GatewayRunner)
    both_entered = asyncio.Event()
    entered = 0
    observed = {}

    async def fake_run_agent_inner(message, *args, **kwargs):
        nonlocal entered
        observed[message] = [get_session_env("HERMES_SESSION_MESSAGE_ID")]
        entered += 1
        if entered == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), timeout=1)
        observed[message].append(get_session_env("HERMES_SESSION_MESSAGE_ID"))
        return {"final_response": "ok"}

    monkeypatch.setattr(runner, "_run_agent_inner", fake_run_agent_inner)
    monkeypatch.setattr(runner, "_profile_scope_for_source", lambda _source: nullcontext())
    await asyncio.gather(
        runner._run_agent("first", "", [], _feishu_source("om_first"), "session-1"),
        runner._run_agent("second", "", [], _feishu_source("om_second"), "session-2"),
    )

    assert observed == {"first": ["om_first", "om_first"], "second": ["om_second", "om_second"]}


def test_get_session_env_falls_back_to_os_environ(monkeypatch):
    """get_session_env should fall back to os.environ when contextvar is unset."""
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")

    # No contextvar set — should read from os.environ
    assert get_session_env("HERMES_SESSION_PLATFORM") == "discord"

    # Now set a contextvar — should prefer it
    tokens = set_session_vars(platform="telegram")
    assert get_session_env("HERMES_SESSION_PLATFORM") == "telegram"

    # After clear — should return "" (explicitly cleared), NOT fall back
    # to os.environ.  This is the fix for #10304: stale os.environ values
    # must not leak through after a gateway session is cleaned up.
    clear_session_vars(tokens)
    assert get_session_env("HERMES_SESSION_PLATFORM") == ""


# ---------------------------------------------------------------------------
# SESSION_KEY contextvars tests
# ---------------------------------------------------------------------------


def test_session_key_falls_back_to_os_environ(monkeypatch):
    """get_session_env for SESSION_KEY should fall back to os.environ."""
    monkeypatch.setenv("HERMES_SESSION_KEY", "env-session-123")

    # No contextvar set — should read from os.environ
    assert get_session_env("HERMES_SESSION_KEY") == "env-session-123"

    # Set contextvar — should prefer it
    tokens = set_session_vars(session_key="ctx-session-456")
    assert get_session_env("HERMES_SESSION_KEY") == "ctx-session-456"

    # After clear — should return "" (explicitly cleared), not os.environ (#10304)
    clear_session_vars(tokens)
    assert get_session_env("HERMES_SESSION_KEY") == ""


def test_session_key_no_race_condition_with_contextvars(monkeypatch):
    """Prove contextvars isolates SESSION_KEY across concurrent async tasks.

    Two tasks set different session keys. With contextvars each task
    reads back its own value. With os.environ the second task would
    overwrite the first (the old bug).
    """
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)

    results = {}

    async def handler(key: str, delay: float):
        tokens = set_session_vars(session_key=key)
        try:
            await asyncio.sleep(delay)
            read_back = get_session_env("HERMES_SESSION_KEY")
            results[key] = read_back
        finally:
            clear_session_vars(tokens)

    async def run():
        task_a = asyncio.create_task(handler("session-A", 0.15))
        await asyncio.sleep(0.05)
        task_b = asyncio.create_task(handler("session-B", 0.05))
        await asyncio.gather(task_a, task_b)

    asyncio.run(run())

    # Both tasks must read back their own session key
    assert results["session-A"] == "session-A", (
        f"Session A got '{results['session-A']}' instead of 'session-A' — race condition!"
    )
    assert results["session-B"] == "session-B", (
        f"Session B got '{results['session-B']}' instead of 'session-B' — race condition!"
    )


@pytest.mark.asyncio
async def test_run_in_executor_with_context_preserves_session_env(monkeypatch):
    """Gateway executor work should inherit session contextvars for tool routing."""
    runner = object.__new__(GatewayRunner)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_ID", raising=False)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="2144471399",
        chat_type="dm",
        user_id="123456",
        user_name="alice",
        thread_id=None,
    )
    context = SessionContext(
        source=source,
        connected_platforms=[],
        home_channels={},
        session_key="agent:main:telegram:dm:2144471399",
    )

    tokens = runner._set_session_env(context)
    try:
        result = await runner._run_in_executor_with_context(
            lambda: {
                "platform": get_session_env("HERMES_SESSION_PLATFORM"),
                "chat_id": get_session_env("HERMES_SESSION_CHAT_ID"),
                "user_id": get_session_env("HERMES_SESSION_USER_ID"),
                "session_key": get_session_env("HERMES_SESSION_KEY"),
            }
        )
    finally:
        runner._clear_session_env(tokens)
        runner._shutdown_executor()

    assert result == {
        "platform": "telegram",
        "chat_id": "2144471399",
        "user_id": "123456",
        "session_key": "agent:main:telegram:dm:2144471399",
    }




def test_cron_session_contextvar_preserves_legacy_env_fallback(monkeypatch):
    """Unset cron ContextVar keeps old env-only cron callers working."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")

    assert get_session_env("HERMES_CRON_SESSION") == "1"


def test_cron_session_explicit_blank_masks_leaked_env(monkeypatch):
    """Non-cron session bindings must override a stale process cron env flag."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")

    tokens = set_session_vars(platform="api_server", cron_session="")
    try:
        assert get_session_env("HERMES_CRON_SESSION") == ""
    finally:
        clear_session_vars(tokens)

    assert get_session_env("HERMES_CRON_SESSION") == ""


def test_cron_session_set_clear_and_reset_tristate(monkeypatch):
    """Cron marker supports _UNSET fallback, 1 cron, and  explicit clear."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")

    tokens = set_session_vars(cron_session="1")
    assert get_session_env("HERMES_CRON_SESSION") == "1"

    clear_session_vars(tokens)
    assert get_session_env("HERMES_CRON_SESSION") == ""

    reset_session_vars()
    assert get_session_env("HERMES_CRON_SESSION") == "1"


@pytest.mark.asyncio
async def test_plugin_slash_command_sees_session_env(monkeypatch):
    """A plugin-registered slash command handler must see the same HERMES_SESSION_*
    contextvars an agent turn would for that event (#108698): the agent-turn path binds
    them via _set_session_env before running, but plugin command dispatch is a separate,
    earlier path that previously called the handler with nothing bound."""
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.platforms.event import MessageEvent

    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    runner._draining = False

    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="c1", user_id="u1", user_name="tester", chat_type="dm",
    )
    event = MessageEvent(text="/gsd bind", source=source, message_id="m1")

    seen = {}

    def _handler(raw_args):
        seen["session_key"] = get_session_env("HERMES_SESSION_KEY")
        seen["chat_id"] = get_session_env("HERMES_SESSION_CHAT_ID")
        return f"Bound: {raw_args}"

    from hermes_cli import plugins as _plugins_mod
    monkeypatch.setattr(_plugins_mod, "get_plugin_command_handler",
                         lambda name: _handler if name == "gsd-bind" else None)

    handled, result, command = await runner._hm_dispatch_quick_and_plugin_commands(event, source, "gsd_bind")

    assert handled is True
    assert result == "Bound: bind"
    assert seen["session_key"] == runner._session_key_for_source(source)
    assert seen["session_key"] != ""
    assert seen["chat_id"] == "c1"
    # Bound only for the handler call, not leaked past dispatch
    assert get_session_env("HERMES_SESSION_KEY") == ""



@pytest.mark.asyncio
async def test_queued_followup_turn_sees_its_own_message_id_in_the_agent_thread(monkeypatch, tmp_path):
    """End to end through the turn worker: the agent (tool) thread of a queued follow-up reads
    the follow-up's triggering message id, not the first message's."""
    import importlib
    import sys
    import types

    from gateway.platforms.event import MessageEvent, MessageType
    from tests.gateway.test_run_progress_topics import ProgressCaptureAdapter, _make_runner

    seen = []

    class _Agent:
        def __init__(self, **kwargs):
            self.tools = []

        def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
            seen.append((message, get_session_env("HERMES_SESSION_MESSAGE_ID")))
            return {"final_response": "ok", "messages": [], "api_calls": 1, "completed": True}

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _Agent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    adapter = ProgressCaptureAdapter(platform=Platform.TELEGRAM)
    runner = _make_runner(adapter)

    def _source(message_id):
        return SessionSource(platform=Platform.TELEGRAM, chat_id="-1001", chat_type="group", message_id=message_id)

    session_key = "agent:main:telegram:group:-1001"
    adapter._pending_messages[session_key] = MessageEvent(
        text="second", message_type=MessageType.TEXT, source=_source("m-2"), message_id="m-2",
    )
    tokens = set_session_vars(message_id="m-1")
    try:
        await runner._run_agent(
            message="first", context_prompt="", history=[], source=_source("m-1"),
            session_id="sess-msg-id", session_key=session_key,
        )
    finally:
        clear_session_vars(tokens)

    assert seen == [("first", "m-1"), ("second", "m-2")]
