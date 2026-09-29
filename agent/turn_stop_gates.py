"""Text-response stop gates for the conversation turn loop.

When the model stops with a text answer, three gates may instead append the answer as an
interim row plus a synthetic user-role nudge and continue the turn: verify-on-stop (#65919),
the ``pre_verify`` plugin hook after code edits, and the kanban worker terminal-tool guard.
Each keeps the candidate answer as a budget-exhaustion fallback
(``pending_verification_response``) and clears ``final_response`` so the finalizer can tell
this gate from error exits (#61631). The ``pre_response`` policy gate runs after them on any
accepted answer (see :func:`apply_pre_response_gate`). Nothing here imports
``agent.conversation_loop`` at module level (cycle).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.message_metadata import append_message

logger = logging.getLogger("agent.conversation_loop")


@dataclass
class StopGateVerdict:
    """``continue_turn`` True → a nudge was appended; re-enter the turn loop with
    ``final_response=None`` and the pending-verification fields updated."""

    continue_turn: bool
    final_response: Any
    pending_verification_response: Any
    pending_verification_response_previewed: Any


def _verify_on_stop_nudge(agent) -> Optional[str]:
    try:
        from agent.verification_stop import (
            build_verify_on_stop_nudge, verify_on_stop_enabled
        )

        if verify_on_stop_enabled():
            return build_verify_on_stop_nudge(
                session_id=getattr(agent, "session_id", None),
                changed_paths=getattr(agent, "_turn_file_mutation_paths", set()),
                attempts=getattr(agent, "_verification_stop_nudges", 0),
            )
    except Exception:
        logger.debug("verification stop-loop check failed", exc_info=True)
    return None


def _pre_verify_nudge(agent, final_response, attempt: int) -> Optional[str]:
    """After code edits a registered ``pre_verify`` hook may keep the agent going one
    more turn; no default continuation cost."""
    _edited = sorted(getattr(agent, "_turn_file_mutation_paths", set()) or [])
    try:
        from agent.verify_hooks import max_verify_nudges
        from hermes_cli.lifecycle import has_hook
        from hermes_cli.plugins import get_pre_verify_continue_message

        if _edited and has_hook("pre_verify") and attempt < max_verify_nudges():
            # Posture is fixed for the session — resolve once + cache.
            coding = getattr(agent, "_resolved_is_coding", None)
            if coding is None:
                from agent.coding_context import is_coding_context
                coding = bool(is_coding_context(platform=getattr(agent, "platform", "") or ""))
                agent._resolved_is_coding = coding
            return get_pre_verify_continue_message(
                session_id=getattr(agent, "session_id", None) or "",
                platform=getattr(agent, "platform", "") or "",
                model=getattr(agent, "model", "") or "", coding=coding, attempt=attempt,
                final_response=final_response, changed_paths=_edited,
            )
    except Exception:
        logger.debug("pre_verify hook check failed", exc_info=True)
    return None


def _kanban_stop_nudge(agent, messages) -> Optional[str]:
    """Workers must end with a terminal board tool (kanban_complete / kanban_block /
    kanban_request_review / kanban_request_changes); a narrated stop is recorded
    as protocol_violation, so nudge once or twice first."""
    try:
        from agent.kanban_stop import build_kanban_stop_nudge

        return build_kanban_stop_nudge(
            messages=messages, attempts=getattr(agent, "_kanban_stop_nudges", 0)
        )
    except Exception:
        logger.debug("kanban stop-loop check failed", exc_info=True)
        return None


def _append_interim_answer(agent, final_msg, messages, conversation_history, flush_fail_msg: str) -> None:
    """Real content: persist and emit as interim so the user sees the attempted answer;
    only the nudge is flagged synthetic (#65919)."""
    agent._emit_interim_assistant_message(final_msg)
    append_message(messages, final_msg)
    try:
        agent._flush_messages_to_session_db(messages, conversation_history)
    except Exception:
        logger.debug(flush_fail_msg, exc_info=True)


def apply_stop_gates(
    agent: Any, final_msg: Dict[str, Any], *, final_response: Any, messages: List[Dict[str, Any]],
    conversation_history: Any, pending_verification_response: Any,
    pending_verification_response_previewed: Any,
) -> StopGateVerdict:
    """Run verify-on-stop → pre_verify hook → kanban stop guard, in that order. Nudges
    are user-role rows appended only after the assistant answer row, so role alternation
    holds. Hook lookups are imported lazily from their origin modules (tests patch them
    there)."""

    def _continue(nudge: str, flag: str) -> StopGateVerdict:
        """Append the synthetic nudge row and hand the turn back to the loop."""
        append_message(messages, {"role": "user", "content": nudge, flag: True})
        agent._session_messages = messages
        # Keep the answer only as a budget-exhaustion fallback; clear ``final_response`` so
        # the finalizer can tell this gate from error exits. Mark previewed only if the
        # candidate is reused (#61631).
        return StopGateVerdict(
            continue_turn=True, final_response=None,
            pending_verification_response=final_response,
            pending_verification_response_previewed=agent._interim_content_was_streamed(
                final_response or ""
            ),
        )

    _verify_nudge = _verify_on_stop_nudge(agent)
    if _verify_nudge:
        agent._verification_stop_nudges = getattr(agent, "_verification_stop_nudges", 0) + 1
        final_msg["finish_reason"] = "verification_required"
        _append_interim_answer(
            agent, final_msg, messages, conversation_history, "verify-on-stop interim flush failed"
        )
        verdict = _continue(_verify_nudge, "_verification_stop_synthetic")
        # Internal nudge: stay silent on the terminal, debug-log only.
        logger.debug("verification stop-loop nudge issued (attempt %d)", agent._verification_stop_nudges)
        return verdict

    _attempt = getattr(agent, "_pre_verify_nudges", 0)
    _verify_nudge2 = _pre_verify_nudge(agent, final_response, _attempt)
    if _verify_nudge2:
        agent._pre_verify_nudges = _attempt + 1
        final_msg["finish_reason"] = "verify_hook_continue"
        _append_interim_answer(
            agent, final_msg, messages, conversation_history, "pre_verify interim flush failed"
        )
        verdict = _continue(_verify_nudge2, "_pre_verify_synthetic")
        logger.debug("pre_verify nudge issued (attempt %d)", agent._pre_verify_nudges)
        return verdict

    _kanban_nudge = _kanban_stop_nudge(agent, messages)
    if _kanban_nudge:
        agent._kanban_stop_nudges = getattr(agent, "_kanban_stop_nudges", 0) + 1
        final_msg["finish_reason"] = "kanban_terminal_required"
        final_msg["_kanban_stop_synthetic"] = True
        append_message(messages, final_msg)
        verdict = _continue(_kanban_nudge, "_kanban_stop_synthetic")
        logger.info(
            "kanban stop-loop nudge issued (attempt %d) task=%s",
            agent._kanban_stop_nudges,
            os.environ.get("HERMES_KANBAN_TASK", ""),
        )
        agent._emit_diagnostic_status(
            "⚠️ Kanban worker tried to exit without a terminal board call "
            "(kanban_complete/kanban_request_review/kanban_block) — nudging to finish"
        )
        return verdict
    return StopGateVerdict(
        continue_turn=False, final_response=final_response,
        pending_verification_response=pending_verification_response,
        pending_verification_response_previewed=pending_verification_response_previewed,
    )


@dataclass
class PreResponseVerdict:
    """``continue_turn`` True → the withheld answer and a nudge were appended; re-enter the loop
    with ``final_response=None``. Otherwise ``final_response`` is the (possibly replaced) answer."""

    continue_turn: bool
    final_response: Any
    pending_pre_response_fallback: Any


_DEFAULT_PRE_RESPONSE_FALLBACK = (
    "I could not complete the required response checks, so I have not confirmed completion."
)


def apply_pre_response_gate(
    agent: Any, final_msg: Dict[str, Any], *, final_response: Any, messages: List[Dict[str, Any]],
    user_message: Any, pending_pre_response_fallback: Any,
) -> PreResponseVerdict:
    """Final-response policy gate (``pre_response`` hook), run on any answer the stop gates
    accepted. ``continue`` withholds the answer for at most ``agent.max_pre_response_nudges``
    more iterations: the answer and the nudge are appended as ONE alternating assistant+user
    pair, both flagged ``_pre_response_synthetic`` (never persisted, never emitted, stripped at
    finalize). ``replace`` — or a ``continue`` past the bound, which delivers the hook's safe
    ``fallback`` — rewrites the answer in place. A failing hook check lets the answer through."""
    attempt = getattr(agent, "_pre_response_nudges", 0)
    directive = None
    max_nudges = 0
    try:
        from agent.response_hooks import max_pre_response_nudges
        from hermes_cli.lifecycle import has_hook
        from hermes_cli.plugins import get_pre_response_directive

        if has_hook("pre_response"):
            directive = get_pre_response_directive(
                session_id=getattr(agent, "session_id", None) or "",
                task_id=getattr(agent, "_current_task_id", "") or "",
                turn_id=getattr(agent, "_current_turn_id", "") or "",
                platform=getattr(agent, "platform", "") or "",
                model=getattr(agent, "model", "") or "",
                attempt=attempt, user_message=user_message,
                platform_message_id=getattr(agent, "_persist_user_message_platform_id", None) or "",
                final_response=final_response,
            )
            max_nudges = max_pre_response_nudges()
    except Exception:
        logger.debug("pre_response hook check failed", exc_info=True)
        directive = None
    if not directive:
        return PreResponseVerdict(False, final_response, pending_pre_response_fallback)

    action, message = directive["action"], directive["message"]
    fallback = directive.get("fallback") or _DEFAULT_PRE_RESPONSE_FALLBACK
    if action == "continue" and attempt < max_nudges:
        agent._pre_response_nudges = attempt + 1
        final_msg["finish_reason"] = "response_hook_continue"
        final_msg["_pre_response_synthetic"] = True
        append_message(messages, final_msg)
        append_message(messages, {"role": "user", "content": message, "_pre_response_synthetic": True})
        agent._session_messages = messages
        logger.debug("pre_response nudge issued (attempt %d)", agent._pre_response_nudges)
        return PreResponseVerdict(True, None, fallback)
    if action == "continue":
        # Continuation bound reached: deliver the hook's safe fallback, never the answer that
        # failed the gate.
        message = fallback
    final_msg["content"] = message
    # A promoted-reasoning sidecar would replay the withheld text on the wire.
    final_msg.pop("api_content", None)
    final_msg["finish_reason"] = "response_hook_replace"
    return PreResponseVerdict(False, message, pending_pre_response_fallback)
