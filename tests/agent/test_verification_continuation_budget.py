"""End-to-end regression coverage for verification budget exhaustion (#61631, #65919 §7)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _response(content="composed report"):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        instance = AIAgent(
            session_id="verify-budget-test",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=1,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    instance._cached_system_prompt = "stable test prompt"
    instance._session_db = None

    instance.save_trajectories = False
    instance.compression_enabled = False
    instance._cleanup_task_resources = lambda *_a, **_kw: None
    instance._save_trajectory = lambda *_a, **_kw: None
    return instance


def _assert_pending_response_survives(agent, result):
    assert result["final_response"] == "composed report"
    assert result["turn_exit_reason"] == "max_iterations_reached(1/1)"
    assert result["completed"] is False
    assert agent._handle_max_iterations.call_count == 0
    # The nudge is stripped by _drop_verification_continuation_scaffolding,
    # so the role sequence is [user, assistant] — the candidate is the
    # tail and matches final_response so it is not duplicated. (#65919 §7)
    assert [message["role"] for message in result["messages"]] == [
        "user",
        "assistant",
    ]


def test_verify_on_stop_preserves_composed_report_at_budget_limit(agent, monkeypatch):
    def model_call(_api_kwargs):
        agent._turn_file_mutation_paths = {"changed.py"}
        return _response()

    agent._interruptible_api_call = model_call
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")

    with (
        patch("agent.verification_stop.build_verify_on_stop_nudge", return_value="verify it"),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("edit changed.py")

    _assert_pending_response_survives(agent, result)
    # The assistant response persists (it is real, unflagged content).
    assert not result["messages"][1].get("_verification_stop_synthetic")


def test_pre_verify_preserves_composed_report_at_budget_limit(agent, monkeypatch):
    def model_call(_api_kwargs):
        agent._turn_file_mutation_paths = {"changed.py"}
        return _response()

    agent._interruptible_api_call = model_call
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    with (
        patch("hermes_cli.plugins.has_hook", side_effect=lambda name: name == "pre_verify"),
        patch(
            "hermes_cli.plugins.get_pre_verify_continue_message",
            return_value="run project tests",
        ),
        patch("agent.verify_hooks.max_verify_nudges", return_value=2),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("edit changed.py")

    _assert_pending_response_survives(agent, result)
    # The assistant response persists (it is real, unflagged content).
    assert not result["messages"][1].get("_pre_verify_synthetic")


def test_intermediate_ack_uses_summary_instead_of_premature_text(agent, monkeypatch):
    agent.valid_tool_names = ["web_search"]
    agent._intent_ack_continuation = True
    agent._looks_like_codex_intermediate_ack = MagicMock(return_value=True)
    agent._interruptible_api_call = lambda _kwargs: _response("I'll inspect the files now")
    agent._handle_max_iterations = MagicMock(return_value="verified summary.")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    with (
        patch("hermes_cli.plugins.has_hook", return_value=False),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("inspect /tmp/project")

    assert result["final_response"] == "verified summary."
    assert result["turn_exit_reason"] == "max_iterations_reached(1/1)"
    agent._handle_max_iterations.assert_called_once()


def test_later_verified_response_supersedes_pending_report(agent, monkeypatch):
    agent.max_iterations = 2
    agent.iteration_budget.max_total = 2
    answers = iter([_response("premature report"), _response("verified final report")])
    agent._interruptible_api_call = lambda _kwargs: next(answers)
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")

    with (
        patch(
            "agent.verification_stop.build_verify_on_stop_nudge",
            side_effect=["verify it", None],
        ),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("edit changed.py")

    assert result["final_response"] == "verified final report"
    assert result["turn_exit_reason"] == "text_response(finish_reason=stop)"
    assert result["completed"] is True
    agent._handle_max_iterations.assert_not_called()


def test_multiple_verification_retries_publish_each_candidate_once(agent, monkeypatch):
    """Multiple verification retries should publish each candidate once, in order."""
    agent.max_iterations = 3
    agent.iteration_budget.max_total = 3
    answers = iter([
        _response("candidate one"),
        _response("candidate two"),
        _response("candidate three"),
    ])
    agent._interruptible_api_call = lambda _kwargs: next(answers)
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")

    # Three nudges, then None (so the third candidate is the final response).
    nudge_side_effects = ["verify it", "verify it", None]

    emitted = []
    agent.interim_assistant_callback = lambda text, **kw: emitted.append(text)

    with (
        patch(
            "agent.verification_stop.build_verify_on_stop_nudge",
            side_effect=nudge_side_effects,
        ),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("edit changed.py")

    # Each candidate was emitted as an interim message, in order.
    assert emitted == ["candidate one", "candidate two"]
    # The final response is the last candidate.
    assert result["final_response"] == "candidate three"
    assert result["turn_exit_reason"] == "text_response(finish_reason=stop)"
    assert result["completed"] is True
    agent._handle_max_iterations.assert_not_called()




def test_verify_on_stop_emits_interim_response_to_ui(agent, monkeypatch):
    """The verify-on-stop path must emit the full response to the UI callback.

    With no streaming set up in this test, _interim_content_was_streamed
    returns False, so already_streamed is False — the callback reports
    content the UI has not seen yet.
    """
    agent._interruptible_api_call = lambda _kwargs: _response("composed report")
    agent._handle_max_iterations = MagicMock(return_value="replacement summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "1")

    callback_calls = []

    def capture_callback(text, *, already_streamed=None):
        callback_calls.append({"text": text, "already_streamed": already_streamed})

    agent.interim_assistant_callback = capture_callback

    with (
        patch("agent.verification_stop.build_verify_on_stop_nudge", return_value="verify it"),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("edit changed.py")

    # The callback was called with the full response text and already_streamed=False
    assert len(callback_calls) == 1
    assert callback_calls[0]["text"] == "composed report"
    assert callback_calls[0]["already_streamed"] is False

    # The candidate persists as the final response.
    assert result["final_response"] == "composed report"


def test_streamed_interim_then_different_summary_not_marked_previewed(agent, monkeypatch):
    """Ordinary interim narration followed by a different non-streamed summary.

    The model streams "I'll inspect the files now" as an intermediate ack.
    _emit_interim_assistant_message is called for this ordinary narration,
    which must NOT set _response_was_previewed. Then _handle_max_iterations
    produces a different summary through the non-streaming Chat Completions
    path. The final result must NOT be marked as previewed — the interim was
    unrelated mid-turn commentary, not the final response — so the CLI renders
    the summary instead of suppressing it. (#65919 review: response-loss blocker)
    """
    agent.valid_tool_names = ["web_search"]
    agent._intent_ack_continuation = True
    agent._looks_like_codex_intermediate_ack = MagicMock(return_value=True)
    agent._interruptible_api_call = lambda _kwargs: _response("I'll inspect the files now")
    agent._handle_max_iterations = MagicMock(return_value="Here is the summary of what I found.")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    emitted = []
    agent.interim_assistant_callback = lambda text, **kw: emitted.append(text)

    with (
        patch("hermes_cli.plugins.has_hook", return_value=False),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("inspect /tmp/project")

    # The final response is the different summary from _handle_max_iterations.
    assert result["final_response"] == "Here is the summary of what I found."
    # CRITICAL: response_previewed must be False — the interim narration was
    # NOT the final response, so the CLI must render the summary.
    assert result["response_previewed"] is False


def _pre_response_patches(directive):
    return (
        patch("hermes_cli.plugins.has_hook", side_effect=lambda name: name == "pre_response"),
        patch(
            "hermes_cli.plugins.get_pre_response_directive",
            **({"side_effect": directive} if isinstance(directive, list) else {"return_value": directive}),
        ),
        patch("agent.response_hooks.max_pre_response_nudges", return_value=2),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    )


def test_pre_response_guard_uses_safe_fallback_at_budget_limit(agent, monkeypatch):
    agent._interruptible_api_call = lambda _kwargs: _response("premature confirmation")
    agent._handle_max_iterations = MagicMock(return_value="unsafe summary")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")
    emitted = []
    agent.interim_assistant_callback = lambda text, **kw: emitted.append(text)

    has_hook, directive, bound, invoke = _pre_response_patches({
        "action": "continue", "message": "persist the correction first",
        "fallback": "The correction was not persisted, so it is not confirmed.",
    })
    with has_hook, directive as get_directive, bound, invoke:
        result = agent.run_conversation("record this", persist_user_platform_id="om_123")

    assert result["final_response"] == "The correction was not persisted, so it is not confirmed."
    assert result["turn_exit_reason"] == "max_iterations_reached(1/1)"
    assert result["completed"] is False
    agent._handle_max_iterations.assert_not_called()
    # The withheld answer is never shown and its scaffolding pair never outlives the turn.
    assert emitted == []
    assert [(m["role"], m["content"]) for m in result["messages"]] == [
        ("user", "record this"),
        ("assistant", "The correction was not persisted, so it is not confirmed."),
    ]
    kwargs = get_directive.call_args.kwargs
    assert kwargs["attempt"] == 0
    assert kwargs["final_response"] == "premature confirmation"
    assert kwargs["platform_message_id"] == "om_123"
    assert kwargs["turn_id"] == agent._current_turn_id


def test_pre_response_guard_allows_later_verified_response(agent, monkeypatch):
    agent.max_iterations = 2
    agent.iteration_budget.max_total = 2
    answers = iter([_response("premature confirmation"), _response("verified confirmation")])
    sent = []

    def model_call(api_kwargs):
        sent.append([dict(m) for m in api_kwargs["messages"]])
        return next(answers)

    agent._interruptible_api_call = model_call
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    has_hook, directive, bound, invoke = _pre_response_patches([
        {"action": "continue", "message": "persist the correction first", "fallback": "not persisted"},
        None,
    ])
    with has_hook, directive, bound, invoke:
        result = agent.run_conversation("record this")

    assert result["final_response"] == "verified confirmation"
    assert result["completed"] is True
    # Cache safety: the second request extends the first byte-for-byte with exactly one
    # alternating assistant + synthetic-user pair.
    first, second = sent
    assert second[: len(first)] == first
    assert [(m["role"], m["content"]) for m in second[len(first):]] == [
        ("assistant", "premature confirmation"),
        ("user", "persist the correction first"),
    ]
    assert [(m["role"], m["content"]) for m in result["messages"]] == [
        ("user", "record this"), ("assistant", "verified confirmation"),
    ]


def test_pre_response_replace_delivers_hook_message(agent, monkeypatch):
    agent.max_iterations = 2
    agent.iteration_budget.max_total = 2
    agent._interruptible_api_call = lambda _kwargs: _response("I changed the protected file.")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    has_hook, directive, bound, invoke = _pre_response_patches(
        {"action": "replace", "message": "I cannot confirm that change."},
    )
    with has_hook, directive, bound, invoke:
        result = agent.run_conversation("change it")

    assert result["final_response"] == "I cannot confirm that change."
    assert result["completed"] is True
    assert result["messages"][-1]["content"] == "I cannot confirm that change."


def test_pre_response_continue_past_bound_delivers_fallback(agent, monkeypatch):
    agent.max_iterations = 5
    agent.iteration_budget.max_total = 5
    calls = []
    agent._interruptible_api_call = lambda _kwargs: calls.append(1) or _response("premature confirmation")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    has_hook, directive, bound, invoke = _pre_response_patches(
        {"action": "continue", "message": "persist first", "fallback": "Not persisted."},
    )
    with has_hook, directive, bound, invoke:
        result = agent.run_conversation("record this")

    # max_pre_response_nudges=2: two continuations, then the fallback replaces the third answer.
    assert len(calls) == 3
    assert result["final_response"] == "Not persisted."
    assert [(m["role"], m["content"]) for m in result["messages"]] == [
        ("user", "record this"), ("assistant", "Not persisted."),
    ]


def test_pre_response_hook_failure_lets_response_through(agent, monkeypatch):
    agent.max_iterations = 2
    agent.iteration_budget.max_total = 2
    agent._interruptible_api_call = lambda _kwargs: _response("answer")
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    with (
        patch("hermes_cli.plugins.has_hook", side_effect=lambda name: name == "pre_response"),
        patch("hermes_cli.plugins.get_pre_response_directive", side_effect=RuntimeError("boom")),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("q")

    assert result["final_response"] == "answer"
    assert result["completed"] is True
