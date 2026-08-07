"""Regression coverage for the fork's Feishu topic routing behavior."""

from __future__ import annotations

import json
from types import SimpleNamespace

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


@pytest.mark.asyncio
async def test_stale_topic_send_fails_closed(monkeypatch):
    adapter = FeishuAdapter(PlatformConfig())
    calls = []

    async def fake_send(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(success=lambda: False, code=99992402)

    monkeypatch.setattr(adapter, "_send_raw_message", fake_send)

    response = await adapter._feishu_send_with_retry(
        chat_id="oc_chat",
        msg_type="text",
        payload='{"text":"hello"}',
        reply_to=None,
        metadata={"thread_id": "om_stale"},
    )

    assert response.success() is False
    assert len(calls) == 1
    assert calls[0]["metadata"]["thread_id"] == "om_stale"
