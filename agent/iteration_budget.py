"""Per-agent iteration budget — thread-safe consume/refund counter.

Each ``AIAgent`` (parent or subagent) holds its own :class:`IterationBudget`: the parent's
cap is ``max_iterations`` (default 500), each subagent's ``delegation.max_iterations``
(default 50), so total iterations across parent + subagents can exceed the parent's cap.
The gateway passes ONE parent budget through every internal continuation turn of a user request
(async-delegation completion, background-process notify/watch events, queued internal events,
a leftover /steer), so a synthetic follow-up cannot silently reset the request's cap.
"""

from __future__ import annotations

import math
import threading

# Runtime-only key carrying a request-chain ``IterationBudget`` on completion events and on
# internal ``MessageEvent.metadata``. Never serialized (the value holds a lock).
REQUEST_CHAIN_BUDGET_EVENT_KEY = "_request_chain_iteration_budget"


def normalize_budget_warning_ratio(value) -> float | None:
    """A finite ratio strictly between zero and one, or None (feature off)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return None
    return ratio if math.isfinite(ratio) and 0 < ratio < 1 else None


class IterationBudget:
    """Thread-safe iteration counter; ``execute_code`` (programmatic tool calling)
    iterations are refunded via :meth:`refund` so they don't eat into the budget."""

    def __init__(self, max_total: int):
        self.max_total = max_total
        self._used = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """Try to consume one iteration.  Returns True if allowed."""
        with self._lock:
            if self._used >= self.max_total:
                return False
            self._used += 1
            return True

    def refund(self) -> None:
        """Give back one iteration (e.g. for execute_code turns)."""
        with self._lock:
            if self._used > 0:
                self._used -= 1

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._used)


_FOREGROUND_CAP_ORIGINAL_ATTR = "_foreground_iteration_cap_original"


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def apply_foreground_iteration_cap(agent, cap: int) -> None:
    """Lower *agent*'s loop cap (``max_iterations``) for the CURRENT turn only; the first call
    remembers the original. Always paired with :func:`restore_foreground_iteration_cap`, which
    ``conversation_loop.run_conversation`` runs on every exit and turn start runs again."""
    if not _positive_int(cap):
        return
    state = getattr(agent, "__dict__", None)
    if isinstance(state, dict) and _FOREGROUND_CAP_ORIGINAL_ATTR not in state:
        original = getattr(agent, "max_iterations", None)
        if not _positive_int(original):
            original = getattr(getattr(agent, "iteration_budget", None), "max_total", None)
        if _positive_int(original):
            state[_FOREGROUND_CAP_ORIGINAL_ATTR] = original
    agent.max_iterations = cap


def restore_foreground_iteration_cap(agent) -> None:
    """Undo :func:`apply_foreground_iteration_cap` (no-op when no cap is active)."""
    state = getattr(agent, "__dict__", None)
    original = state.pop(_FOREGROUND_CAP_ORIGINAL_ATTR, None) if isinstance(state, dict) else None
    if _positive_int(original):
        agent.max_iterations = original


def request_chain_budget_of(agent) -> "IterationBudget | None":
    """The request-chain budget that owns work started by *agent*: the ROOT agent's budget. A
    delegated child's own budget (``delegation.max_iterations``) must never become the cap of the
    parent's completion turn, so walk ``_delegate_parent_ref`` up to the top-level agent."""
    seen = 0
    while agent is not None and getattr(agent, "_delegate_depth", 0) and seen < 16:
        ref = getattr(agent, "_delegate_parent_ref", None)
        agent = ref() if callable(ref) else None
        seen += 1
    budget = getattr(agent, "iteration_budget", None) if agent is not None else None
    return budget if isinstance(budget, IterationBudget) else None


def current_request_chain_budget() -> "IterationBudget | None":
    """:func:`request_chain_budget_of` for the agent whose turn runs in this execution context."""
    try:
        from agent.subagent_lifecycle import get_active_subagent_parent
    except Exception:
        return None
    return request_chain_budget_of(get_active_subagent_parent())


__all__ = [
    "IterationBudget", "REQUEST_CHAIN_BUDGET_EVENT_KEY", "apply_foreground_iteration_cap",
    "current_request_chain_budget", "request_chain_budget_of", "restore_foreground_iteration_cap",
]
