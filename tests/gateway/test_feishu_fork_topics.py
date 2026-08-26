"""Regression coverage for the fork's Feishu topic routing behavior."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.session import Platform, SessionSource
from plugins.platforms.feishu.adapter import FeishuAdapter


def _topic_starter_ref(adapter: FeishuAdapter):
    return adapter._topic_starter_at_ref(
        message=SimpleNamespace(root_id="om_root"),
        sender_id=SimpleNamespace(open_id="ou_human", user_id="u_human"),
        sender_profile={
            "user_id": "u_human",
            "user_name": "Alice",
            "user_id_alt": "on_human",
        },
        is_bot=False,
        thread_id="om_root",
        reply_to_message_id="om_nested_reply",
    )


def test_topic_starter_mention_uses_open_id_for_nested_root_reply():
    adapter = FeishuAdapter(PlatformConfig())

    assert _topic_starter_ref(adapter) == ("ou_human", "Alice")


def test_topic_starter_without_open_id_is_not_mentioned():
    adapter = FeishuAdapter(PlatformConfig())

    result = adapter._topic_starter_at_ref(
        message=SimpleNamespace(root_id="om_root"),
        sender_id=SimpleNamespace(open_id="", user_id="u_human"),
        sender_profile={
            "user_id": "u_human",
            "user_name": "Alice",
            "user_id_alt": "on_human",
        },
        is_bot=False,
        thread_id="om_root",
        reply_to_message_id="om_root",
    )

    assert result is None


def test_existing_topic_does_not_replace_its_starter():
    adapter = FeishuAdapter(PlatformConfig())

    result = adapter._topic_starter_at_ref(
        message=SimpleNamespace(root_id="om_root"),
        sender_id=SimpleNamespace(open_id="ou_human", user_id="u_human"),
        sender_profile={"user_name": "Alice"},
        is_bot=False,
        thread_id="omt_existing_topic",
        reply_to_message_id="om_previous_topic_message",
    )

    assert result is None


def test_thread_metadata_preserves_topic_starter_mention():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_chat",
        chat_type="group",
        thread_id="om_root",
    )
    source.feishu_topic_starter_user_id = "ou_human"
    source.feishu_topic_starter_user_name = "Alice"

    assert runner._thread_metadata_for_source(source, "om_trigger") == {
        "thread_id": "om_root",
        "feishu_at_user_id": "ou_human",
        "feishu_at_user_name": "Alice",
    }


@pytest.mark.asyncio
async def test_only_first_chunk_mentions_topic_starter(monkeypatch):
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = object()
    monkeypatch.setattr(adapter, "truncate_message", lambda *_a, **_k: ["one", "two"])
    sent = []

    async def fake_send(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(message_id=f"om_{len(sent)}"),
        )

    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)

    result = await adapter.send(
        chat_id="oc_chat",
        content="ignored",
        metadata={
            "thread_id": "om_root",
            "notify": True,
            "feishu_at_user_id": "ou_human",
            "feishu_at_user_name": "Alice",
        },
    )

    assert result.success is True
    assert json.loads(sent[0]["payload"])["text"].startswith(
        '<at user_id="ou_human"></at> '
    )
    assert json.loads(sent[1]["payload"])["text"] == "two"
    assert adapter._outbound_at_by_message_id == {
        "om_1": ("ou_human", "Alice")
    }


@pytest.mark.asyncio
async def test_topic_reply_omits_new_topic_flag(monkeypatch):
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(reply=object())))
    )
    captured = {}

    def fake_body(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    async def fake_blocking(*_args):
        return SimpleNamespace(success=lambda: True)

    monkeypatch.setattr(adapter, "_build_reply_message_body", fake_body)
    monkeypatch.setattr(
        adapter,
        "_build_reply_message_request",
        lambda message_id, request_body: SimpleNamespace(
            message_id=message_id,
            request_body=request_body,
        ),
    )
    monkeypatch.setattr(adapter, "_run_blocking", fake_blocking)

    await adapter._send_raw_message(
        chat_id="oc_chat",
        msg_type="text",
        payload='{"text":"hello"}',
        reply_to=None,
        metadata={
            "thread_id": "om_root",
            "reply_to_message_id": "om_trigger",
        },
    )

    assert captured["reply_in_thread"] is None


def _feishu_response(
    success: bool,
    *,
    code: int = 0,
    message_id: str = "",
    msg: str | None = None,
):
    return SimpleNamespace(
        success=lambda: success,
        code=code,
        msg=("send failed" if not success else "") if msg is None else msg,
        data=SimpleNamespace(message_id=message_id) if success else None,
    )


@pytest.mark.asyncio
async def test_topic_send_success_does_not_fallback(monkeypatch):
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = object()
    calls = []

    async def fake_send(**kwargs):
        calls.append(kwargs)
        return _feishu_response(True, message_id="om_sent")

    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)
    fetch_last = AsyncMock()
    monkeypatch.setattr(adapter, "_fetch_last_message_in_thread", fetch_last)

    result = await adapter.send(
        chat_id="oc_chat",
        content="hello",
        metadata={"thread_id": "om_stale"},
    )

    assert result.success is True
    assert len(calls) == 1
    assert calls[0]["metadata"]["thread_id"] == "om_stale"
    fetch_last.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_topic_replies_to_last_thread_message(monkeypatch):
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = object()
    calls = []
    responses = iter(
        [
            _feishu_response(False, code=99992402),
            _feishu_response(True, message_id="om_reply"),
        ]
    )

    async def fake_send(**kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)
    monkeypatch.setattr(
        adapter,
        "_fetch_last_message_in_thread",
        AsyncMock(return_value="om_last"),
    )

    result = await adapter.send(
        chat_id="oc_chat",
        content="hello",
        metadata={"thread_id": "om_stale"},
    )

    assert result.success is True
    assert [call["reply_to"] for call in calls] == [None, "om_last"]
    assert all(call["metadata"]["thread_id"] == "om_stale" for call in calls)


@pytest.mark.asyncio
async def test_stale_topic_reply_failure_forwards_once_to_chat(monkeypatch):
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = object()
    calls = []
    responses = iter(
        [
            _feishu_response(False, code=99992402),
            _feishu_response(False, code=99992402),
            _feishu_response(True, message_id="om_chat"),
        ]
    )

    async def fake_send(**kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)

    result = await adapter.send(
        chat_id="oc_chat",
        content="hello",
        metadata={
            "thread_id": "om_stale",
            "reply_to_message_id": "om_anchor",
        },
    )

    assert result.success is True
    assert [call["reply_to"] for call in calls] == [None, "om_anchor", None]
    assert calls[2]["metadata"] is None
    assert json.loads(calls[2]["payload"])["text"] == (
        "原话题已失效，回复转发至群聊\n\nhello"
    )


@pytest.mark.asyncio
async def test_stale_topic_fallback_keeps_post_rejection_text_downgrade(monkeypatch):
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = object()
    calls = []
    responses = iter(
        [
            _feishu_response(
                False,
                code=230001,
                msg="content format of the post type is incorrect",
            ),
            _feishu_response(False, code=99992402),
            _feishu_response(False, code=99992402),
            _feishu_response(True, message_id="om_chat"),
        ]
    )

    async def fake_send(**kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)

    result = await adapter.send(
        chat_id="oc_chat",
        content="**hello**",
        metadata={
            "thread_id": "om_stale",
            "reply_to_message_id": "om_anchor",
        },
    )

    assert result.success is True
    assert [call["msg_type"] for call in calls] == ["post", "text", "text", "text"]
    assert json.loads(calls[-1]["payload"])["text"].startswith(
        "原话题已失效，回复转发至群聊"
    )


@pytest.mark.asyncio
async def test_stale_topic_chat_fallback_persists_for_later_chunks(monkeypatch):
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = object()
    calls = []
    responses = iter(
        [
            _feishu_response(False, code=99992402),
            _feishu_response(False, code=99992402),
            _feishu_response(True, message_id="om_chat_1"),
            _feishu_response(True, message_id="om_chat_2"),
        ]
    )

    async def fake_send(**kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)
    monkeypatch.setattr(
        adapter,
        "truncate_message",
        lambda _content, _limit: ["chunk one", "chunk two"],
    )

    result = await adapter.send(
        chat_id="oc_chat",
        content="long response",
        metadata={
            "thread_id": "om_stale",
            "reply_to_message_id": "om_anchor",
        },
    )

    assert result.success is True
    assert [call["reply_to"] for call in calls] == [
        None,
        "om_anchor",
        None,
        None,
    ]
    assert [call["metadata"] for call in calls[-2:]] == [None, None]
    assert json.loads(calls[2]["payload"])["text"].startswith(
        "原话题已失效，回复转发至群聊"
    )
    assert json.loads(calls[3]["payload"])["text"] == "chunk two"


@pytest.mark.asyncio
async def test_fetch_last_thread_message_requests_descending_order():
    adapter = FeishuAdapter(PlatformConfig())
    captured = {}

    def fake_list(request):
        captured["request"] = request
        return SimpleNamespace(
            success=lambda: True,
            data=SimpleNamespace(items=[SimpleNamespace(message_id="om_latest")]),
        )

    adapter._client = SimpleNamespace(
        im=SimpleNamespace(
            v1=SimpleNamespace(message=SimpleNamespace(list=fake_list)),
        ),
    )

    result = await adapter._fetch_last_message_in_thread("omt_thread")

    assert result == "om_latest"
    assert captured["request"].sort_type == "ByCreateTimeDesc"
    assert captured["request"].page_size == 1


@pytest.mark.asyncio
async def test_non_stale_topic_error_does_not_fallback(monkeypatch):
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = object()
    calls = []

    async def fake_send(**kwargs):
        calls.append(kwargs)
        return _feishu_response(False, code=230001)

    monkeypatch.setattr(adapter, "_feishu_send_with_retry", fake_send)
    fetch_last = AsyncMock()
    monkeypatch.setattr(adapter, "_fetch_last_message_in_thread", fetch_last)

    result = await adapter.send(
        chat_id="oc_chat",
        content="hello",
        metadata={"thread_id": "om_topic"},
    )

    assert result.success is False
    assert len(calls) == 1
    fetch_last.assert_not_awaited()
