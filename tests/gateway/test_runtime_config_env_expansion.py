"""Regression tests for gateway runtime config env-var expansion."""

from __future__ import annotations

import json

import pytest

import gateway.run as gateway_run


def _write_config(home, body: str) -> None:
    (home / "config.yaml").write_text(body, encoding="utf-8")


@pytest.fixture
def gateway_home(monkeypatch, tmp_path):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("HERMES_PREFILL_MESSAGES_FILE", raising=False)
    monkeypatch.delenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_BUSY_INPUT_MODE", raising=False)
    monkeypatch.delenv("HERMES_RESTART_DRAIN_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_BACKGROUND_NOTIFICATIONS", raising=False)
    return tmp_path


def test_load_prefill_messages_prefers_top_level_over_legacy(monkeypatch, gateway_home):
    top_level = [{"role": "system", "content": "top-level"}]
    legacy = [{"role": "system", "content": "legacy"}]
    (gateway_home / "top.json").write_text(json.dumps(top_level), encoding="utf-8")
    (gateway_home / "legacy.json").write_text(json.dumps(legacy), encoding="utf-8")
    _write_config(
        gateway_home,
        "prefill_messages_file: top.json\n"
        "agent:\n"
        "  prefill_messages_file: legacy.json\n",
    )

    assert gateway_run.GatewayRunner._load_prefill_messages() == top_level


def test_load_system_prompt_prepends_configured_files(gateway_home):
    (gateway_home / "shared-one.md").write_text("shared one\n", encoding="utf-8")
    (gateway_home / "shared-two.md").write_text("shared two\n", encoding="utf-8")
    _write_config(
        gateway_home,
        "agent:\n"
        "  system_prompt_files:\n"
        "    - shared-one.md\n"
        f"    - {gateway_home / 'shared-two.md'}\n"
        "  system_prompt: profile-specific\n",
    )

    assert gateway_run.GatewayRunner._load_ephemeral_system_prompt() == (
        "shared one\n\nshared two\n\nprofile-specific"
    )

