import base64
import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource


_ONE_BY_ONE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO6L2ioAAAAASUVORK5CYII="
)


class CaptureAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent = []
        self.typing = []
        self.send_results = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        if self.send_results:
            return self.send_results.pop(0)
        return SendResult(success=True, message_id=f"sent-{len(self.sent)}")

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class CaptureQueuedNativeImageAgent:
    calls = []

    def __init__(self, **kwargs):
        self.tools = []
        self.tool_progress_callback = kwargs.get("tool_progress_callback")

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls.append(message)
        return {
            "final_response": f"done-{len(type(self).calls)}",
            "messages": [],
            "api_calls": 1,
        }


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    runner._decide_image_input_mode = lambda **_kw: "native"
    return runner


@pytest.mark.asyncio
async def test_queued_followup_uses_pending_event_session_key_for_native_images(monkeypatch, tmp_path):
    CaptureQueuedNativeImageAgent.calls = []

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = CaptureQueuedNativeImageAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    adapter = CaptureAdapter()
    runner = _make_runner(adapter)

    image_path = tmp_path / "queued-image.png"
    image_path.write_bytes(_ONE_BY_ONE_PNG)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
    )
    pending_source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )

    adapter._pending_messages["agent:main:telegram:group:-1001"] = MessageEvent(
        text="describe this",
        message_type=MessageType.PHOTO,
        source=pending_source,
        media_urls=[str(image_path)],
        media_types=["image/png"],
        message_id="queued-1",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-native-image-followup",
        session_key="agent:main:telegram:group:-1001",
    )

    assert result["final_response"] == "done-2"
    assert len(CaptureQueuedNativeImageAgent.calls) == 2
    queued_message = CaptureQueuedNativeImageAgent.calls[1]
    assert isinstance(queued_message, list)
    assert queued_message[0]["type"] == "text"
    assert queued_message[0]["text"].startswith("describe this")
    assert any(part.get("type") == "image_url" for part in queued_message)


def _delivery_rows():
    from gateway import delivery_ledger as dl

    with dl._connect() as conn:
        return conn.execute(
            "SELECT state, content, reply_to_message_id FROM delivery_obligations "
            "ORDER BY created_at"
        ).fetchall()


async def _run_plain_queued_followup(monkeypatch, tmp_path, adapter):
    CaptureQueuedNativeImageAgent.calls = []

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = CaptureQueuedNativeImageAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {"api_key": "***"},
    )

    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
    )
    session_key = "agent:main:telegram:group:-1001"
    adapter._pending_messages[session_key] = MessageEvent(
        text="queued second turn",
        message_type=MessageType.TEXT,
        source=source,
        message_id="queued-2",
    )

    return await runner._run_agent(
        message="first turn",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-queued-delivery",
        session_key=session_key,
        event_message_id="incoming-1",
        platform_message_id="incoming-1",
    )


@pytest.mark.asyncio
async def test_queued_first_response_uses_retry_path_and_marks_ledger_delivered(
    monkeypatch,
    tmp_path,
):
    adapter = CaptureAdapter()
    retry_calls = []
    real_send_with_retry = adapter._send_with_retry

    async def tracked_send_with_retry(**kwargs):
        retry_calls.append(kwargs)
        return await real_send_with_retry(**kwargs)

    monkeypatch.setattr(adapter, "_send_with_retry", tracked_send_with_retry)

    result = await _run_plain_queued_followup(monkeypatch, tmp_path, adapter)

    assert result["final_response"] == "done-2"
    assert CaptureQueuedNativeImageAgent.calls == ["first turn", "queued second turn"]
    assert [item["content"] for item in adapter.sent] == ["done-1"]
    assert len(retry_calls) == 1
    rows = _delivery_rows()
    assert [(row[0], row[1], row[2]) for row in rows] == [
        ("delivered", "done-1", "incoming-1")
    ]


@pytest.mark.asyncio
async def test_queued_first_response_failure_is_observable_and_followup_runs_once(
    monkeypatch,
    tmp_path,
):
    adapter = CaptureAdapter()
    adapter.send_results = [
        SendResult(success=False, error="topic rejected"),
        SendResult(success=False, error="plain fallback rejected"),
    ]

    result = await _run_plain_queued_followup(monkeypatch, tmp_path, adapter)

    assert result["final_response"] == "done-2"
    assert CaptureQueuedNativeImageAgent.calls == ["first turn", "queued second turn"]
    assert [item["content"] for item in adapter.sent] == [
        "done-1",
        "(Response formatting failed, plain text:)\n\ndone-1",
    ]
    rows = _delivery_rows()
    assert [(row[0], row[1], row[2]) for row in rows] == [
        ("failed", "done-1", "incoming-1")
    ]
