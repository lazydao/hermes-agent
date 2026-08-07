"""Per-agent iteration budget — thread-safe consume/refund counter.

Extracted from ``run_agent.py``.  Each ``AIAgent`` instance (parent or
subagent) holds an :class:`IterationBudget`; the parent's cap comes from
``max_iterations`` (default 500), each subagent's cap comes from
``delegation.max_iterations`` (default 50). Gateway-internal continuation
turns for one user request may share the same parent budget object so
synthetic follow-ups cannot silently reset the request's cap.

``run_agent`` re-exports ``IterationBudget`` so existing
``from run_agent import IterationBudget`` imports keep working unchanged.
"""

from __future__ import annotations

import threading


REQUEST_CHAIN_BUDGET_EVENT_KEY = "_request_chain_iteration_budget"
_FOREGROUND_CAP_ORIGINAL_ATTR = "_foreground_iteration_cap_original"


def _apply_foreground_iteration_cap(agent, cap: int) -> None:
    """Temporarily lower an agent's loop cap for its current foreground turn."""
    if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
        return

    state = getattr(agent, "__dict__", None)
    if isinstance(state, dict) and _FOREGROUND_CAP_ORIGINAL_ATTR not in state:
        original = getattr(agent, "max_iterations", None)
        if not isinstance(original, int) or isinstance(original, bool) or original <= 0:
            budget = getattr(agent, "iteration_budget", None)
            original = getattr(budget, "max_total", None)
        if isinstance(original, int) and not isinstance(original, bool) and original > 0:
            setattr(agent, _FOREGROUND_CAP_ORIGINAL_ATTR, original)

    agent.max_iterations = cap


def _restore_foreground_iteration_cap(agent) -> None:
    """Restore a cap lowered by :func:`_apply_foreground_iteration_cap`."""
    state = getattr(agent, "__dict__", None)
    if not isinstance(state, dict):
        return
    original = state.pop(_FOREGROUND_CAP_ORIGINAL_ATTR, None)
    if isinstance(original, int) and not isinstance(original, bool) and original > 0:
        agent.max_iterations = original


class IterationBudget:
    """Thread-safe iteration counter for an agent.

    Each agent (parent or subagent) gets its own ``IterationBudget`` by default.
    The gateway may reuse a parent's instance across internal continuation
    turns that belong to one external user request. The parent's budget is
    capped at ``max_iterations`` (default 500).
    Each subagent gets an independent budget capped at
    ``delegation.max_iterations`` (default 50) — this means total
    iterations across parent + subagents can exceed the parent's cap.
    Users control the per-subagent limit via ``delegation.max_iterations``
    in config.yaml.

    ``execute_code`` (programmatic tool calling) iterations are refunded via
    :meth:`refund` so they don't eat into the budget.
    """

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


__all__ = ["IterationBudget", "REQUEST_CHAIN_BUDGET_EVENT_KEY"]
