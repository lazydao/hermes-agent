"""Configuration helper for the bounded ``pre_response`` final-response gate."""

from __future__ import annotations

from typing import Any, Optional

DEFAULT_MAX_PRE_RESPONSE_NUDGES = 2


def max_pre_response_nudges(config: Optional[dict[str, Any]] = None) -> int:
    """Bound on consecutive ``pre_response`` continue directives per turn (>= 0)."""
    from agent.verify_hooks import _agent_cfg

    try:
        return max(0, int(_agent_cfg(config).get("max_pre_response_nudges")))
    except (TypeError, ValueError):
        return DEFAULT_MAX_PRE_RESPONSE_NUDGES


__all__ = ["DEFAULT_MAX_PRE_RESPONSE_NUDGES", "max_pre_response_nudges"]
