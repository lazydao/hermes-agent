"""Configuration helpers for the bounded ``pre_response`` gate."""

from __future__ import annotations

from typing import Any, Optional


DEFAULT_MAX_PRE_RESPONSE_NUDGES = 2


def max_pre_response_nudges(config: Optional[dict[str, Any]] = None) -> int:
    """Return the maximum consecutive continuation directives per turn."""
    agent_cfg = _agent_cfg(config)
    raw = agent_cfg.get("max_pre_response_nudges")
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_MAX_PRE_RESPONSE_NUDGES


def _agent_cfg(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}
    agent_cfg = (config or {}).get("agent") if isinstance(config, dict) else None
    return agent_cfg if isinstance(agent_cfg, dict) else {}


__all__ = [
    "DEFAULT_MAX_PRE_RESPONSE_NUDGES",
    "max_pre_response_nudges",
]
