"""Accepted clarify replies retain reactions until their owning turn ends."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter, MessageEvent, MessageType, ProcessingOutcome, SendResult,
)
from gateway.session import SessionSource


class Adapter(BasePlatformAdapter):
    link_clarify_reactions = True

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test", typing_indicator=False), Platform.FEISHU)
        self.on_processing_start = AsyncMock()
        self.on_processing_complete = AsyncMock()

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "group"}


def event(mid, thread="thread-a"):
    return MessageEvent(
        text="answer", message_type=MessageType.TEXT, message_id=mid,
        source=SessionSource(platform=Platform.FEISHU, chat_id="chat", chat_type="group", user_id="user", thread_id=thread),
    )


async def start_turn(adapter, key="session", mid="original", failure=False):
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(_event):
        entered.set()
        await release.wait()
        if failure:
            raise RuntimeError("failed task")
        return "done"

    adapter._message_handler = handler
    task = asyncio.create_task(adapter._process_message_background(event(mid), key))
    await asyncio.wait_for(entered.wait(), 3)
    return task, release


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", list(ProcessingOutcome))
async def test_owning_task_completes_linked_reaction_with_its_outcome(outcome):
    adapter = Adapter()
    task, release = await start_turn(adapter, failure=outcome is ProcessingOutcome.FAILURE)
    answer = event("answer")
    await adapter.note_clarify_answer_accepted("session", answer)
    adapter.on_processing_start.assert_any_await(answer)
    adapter.on_processing_complete.assert_not_awaited()
    if outcome is ProcessingOutcome.CANCELLED:
        adapter._expected_cancelled_tasks.add(task)
        task.cancel()
    else:
        release.set()
    await asyncio.gather(task, return_exceptions=True)
    adapter.on_processing_complete.assert_any_await(answer, outcome)
    assert not adapter._clarify_reaction_turns


@pytest.mark.asyncio
async def test_many_answers_and_duplicate_delivery_keep_every_cleanup_owner():
    adapter = Adapter()
    task, release = await start_turn(adapter)
    answers = [event(str(i)) for i in range(12)]
    for answer in answers + answers:
        await adapter.note_clarify_answer_accepted("session", answer)
    assert adapter.on_processing_start.await_count == 13
    release.set()
    await task
    assert adapter.on_processing_complete.await_count == 13
    for answer in answers:
        adapter.on_processing_complete.assert_any_await(answer, ProcessingOutcome.SUCCESS)


@pytest.mark.asyncio
async def test_completion_waits_for_inflight_reaction_creation():
    adapter = Adapter()
    task, release = await start_turn(adapter)
    started, allow_add = asyncio.Event(), asyncio.Event()
    answer = event("answer")

    async def slow_start(_event):
        started.set()
        await allow_add.wait()

    adapter.on_processing_start = AsyncMock(side_effect=slow_start)
    add = asyncio.create_task(adapter.note_clarify_answer_accepted("session", answer))
    await started.wait()
    release.set()
    await asyncio.sleep(0)
    assert not any(c.args[0] is answer for c in adapter.on_processing_complete.await_args_list)
    allow_add.set()
    await asyncio.gather(add, task)
    adapter.on_processing_complete.assert_any_await(answer, ProcessingOutcome.SUCCESS)


@pytest.mark.asyncio
async def test_old_turn_cannot_clear_new_turn_after_reset():
    adapter = Adapter()
    old, old_release = await start_turn(adapter)
    old_answer = event("old-answer")
    await adapter.note_clarify_answer_accepted("session", old_answer)
    # Reset replaces the active guard; old cleanup can finish after the new turn starts.
    adapter._active_sessions.pop("session", None)
    new, new_release = await start_turn(adapter, mid="new-original")
    new_answer = event("new-answer")
    await adapter.note_clarify_answer_accepted("session", new_answer)
    old_release.set()
    await old
    assert not any(c.args[0] is new_answer for c in adapter.on_processing_complete.await_args_list)
    assert "session" in adapter._clarify_reaction_turns
    new_release.set()
    await new
    adapter.on_processing_complete.assert_any_await(new_answer, ProcessingOutcome.SUCCESS)


@pytest.mark.asyncio
async def test_other_thread_has_independent_completion():
    adapter = Adapter()
    first, release_first = await start_turn(adapter, "first")
    second, release_second = await start_turn(adapter, "second", "second-original")
    answer = event("answer", "thread-b")
    await adapter.note_clarify_answer_accepted("second", answer)
    release_first.set()
    await first
    assert not any(c.args[0] is answer for c in adapter.on_processing_complete.await_args_list)
    release_second.set()
    await second
    adapter.on_processing_complete.assert_any_await(answer, ProcessingOutcome.SUCCESS)


@pytest.mark.asyncio
async def test_late_reply_and_original_message_do_not_create_extra_reactions():
    adapter = Adapter()
    task, release = await start_turn(adapter)
    await adapter.note_clarify_answer_accepted("session", event("original"))
    release.set()
    await task
    await adapter.note_clarify_answer_accepted("session", event("late"))
    assert adapter.on_processing_start.await_count == 1


@pytest.mark.asyncio
async def test_other_platforms_keep_typing_resume_without_reaction_change():
    adapter = Adapter()
    adapter.link_clarify_reactions = False
    adapter.pause_typing_for_chat("chat")
    await adapter.note_clarify_answer_accepted("session", event("answer"))
    assert "chat" not in adapter._typing_paused
    adapter.on_processing_start.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", list(ProcessingOutcome))
async def test_feishu_linked_reply_reaction_lifecycle(outcome, monkeypatch):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    monkeypatch.setenv("FEISHU_REACTIONS", "true")
    adapter = FeishuAdapter(PlatformConfig(typing_indicator=False))
    adapter._add_reaction = AsyncMock(side_effect=lambda mid, emoji: "reaction-" + mid)
    adapter._remove_reaction = AsyncMock(return_value=True)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="sent"))
    task, release = await start_turn(adapter, failure=outcome is ProcessingOutcome.FAILURE)
    answer = event("accepted")
    await adapter.note_clarify_answer_accepted("session", answer)
    await adapter.note_clarify_answer_accepted("session", answer)
    adapter._add_reaction.assert_any_await("accepted", "Typing")
    assert adapter._add_reaction.await_count == 2
    adapter._remove_reaction.assert_not_awaited()
    if outcome is ProcessingOutcome.CANCELLED:
        adapter._expected_cancelled_tasks.add(task)
        task.cancel()
    else:
        release.set()
    await asyncio.gather(task, return_exceptions=True)
    adapter._remove_reaction.assert_any_await("accepted", "reaction-accepted")
    assert not adapter._pending_processing_reactions
    if outcome is ProcessingOutcome.FAILURE:
        adapter._add_reaction.assert_any_await("accepted", "CrossMark")
    else:
        assert adapter._add_reaction.await_count == 2


@pytest.mark.asyncio
async def test_reset_during_linked_cleanup_does_not_abandon_remaining_reactions():
    adapter = Adapter()
    task, release = await start_turn(adapter)
    answers = [event("answer-a"), event("answer-b")]
    for answer in answers:
        await adapter.note_clarify_answer_accepted("session", answer)
    cleaning, allow_cleanup = asyncio.Event(), asyncio.Event()
    cleaned = []

    async def complete(message, outcome):
        if message is answers[0]:
            cleaning.set()
            await allow_cleanup.wait()
        cleaned.append(message.message_id)

    adapter.on_processing_complete = AsyncMock(side_effect=complete)
    release.set()
    await cleaning.wait()
    adapter._expected_cancelled_tasks.add(task)
    task.cancel()
    await asyncio.sleep(0)
    allow_cleanup.set()
    await asyncio.gather(task, return_exceptions=True)
    assert all(answer.message_id in cleaned for answer in answers)
