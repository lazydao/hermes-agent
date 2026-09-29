"""Regression guard for Feishu bot-sender authorization bypass.

Mirrors tests/gateway/test_discord_bot_auth_bypass.py for Platform.FEISHU.
Without the bypass in gateway/run.py, Feishu bot senders admitted by the
adapter would be rejected at _is_user_authorized with "Unauthorized user"
— same class of bug as Discord #4466.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, PlatformConfig
from gateway.session import Platform, SessionSource


@pytest.fixture(autouse=True)
def _isolate_feishu_env(monkeypatch):
    for var in (
        "FEISHU_ALLOW_BOTS",
        "FEISHU_ALLOWED_USERS",
        "FEISHU_ALLOW_ALL_USERS",
        "FEISHU_GROUP_POLICY",
        "TELEGRAM_ALLOW_BOTS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_bare_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_kw: False)
    return runner


def _make_group_rule_runner(group_rules, **extra):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    platform_config = PlatformConfig(enabled=True, extra={"group_rules": group_rules, **extra})
    adapter = FeishuAdapter(platform_config)
    runner = _make_bare_runner()
    runner.adapters = {Platform.FEISHU: adapter}
    runner._profile_adapters = {}
    runner.config = GatewayConfig(platforms={Platform.FEISHU: platform_config})
    return runner, adapter


def _feishu_source(chat_id: str, user_id: str, chat_type: str = "group", **kw):
    return SessionSource(
        platform=Platform.FEISHU, chat_id=chat_id, chat_type=chat_type,
        user_id=user_id, user_name="Guest", **kw,
    )


def _make_feishu_bot_source(open_id: str = "ou_peer"):
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_1",
        chat_type="group",
        user_id=open_id,
        user_name="PeerBot",
        is_bot=True,
    )


def _make_feishu_human_source(open_id: str = "ou_human"):
    return SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_1",
        chat_type="group",
        user_id=open_id,
        user_name="Human",
        is_bot=False,
    )


def test_feishu_human_still_checked_against_allowlist_when_bot_policy_set(monkeypatch):
    """FEISHU_ALLOW_BOTS=all must NOT open the gate for humans."""
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", "all")
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_human")

    assert runner._is_user_authorized(_make_feishu_human_source("ou_stranger")) is False
    assert runner._is_user_authorized(_make_feishu_human_source("ou_human")) is True


@pytest.mark.parametrize("env_allowlist", ["ou_dm_owner", None])
def test_explicit_open_group_rule_authorizes_only_that_chat(monkeypatch, env_allowlist):
    if env_allowlist:
        monkeypatch.setenv("FEISHU_ALLOWED_USERS", env_allowlist)
    runner, _adapter = _make_group_rule_runner({"oc_1": {"policy": "open"}})

    assert runner._is_user_authorized(_make_feishu_human_source("ou_guest")) is True
    assert runner._is_user_authorized(_feishu_source("oc_1", None)) is True  # sender-less group event
    assert runner._is_user_authorized(_feishu_source("oc_2", "ou_guest")) is False
    assert runner._is_user_authorized(_feishu_source("oc_1", "ou_guest", chat_type="dm")) is False


def test_disabled_group_rule_does_not_authorize_chat(monkeypatch):
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_dm_owner")
    runner, _adapter = _make_group_rule_runner({"oc_1": {"policy": "disabled"}})

    assert runner._is_user_authorized(_make_feishu_human_source("ou_guest")) is False


def test_default_open_group_policy_is_not_a_chat_allowlist(monkeypatch):
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_dm_owner")
    monkeypatch.setenv("FEISHU_GROUP_POLICY", "open")
    runner, _adapter = _make_group_rule_runner({}, default_group_policy="open")

    assert runner._is_user_authorized(_make_feishu_human_source("ou_guest")) is False


def test_group_rule_sender_lists_are_rejudged_at_the_gateway(monkeypatch):
    """Reaction/card events skip adapter intake, so the gateway grant re-checks the rule's sender lists."""
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "ou_dm_owner")
    runner, _adapter = _make_group_rule_runner({
        "oc_1": {"policy": "allowlist", "allowlist": ["ou_member"]},
        "oc_2": {"policy": "blacklist", "blacklist": ["ou_banned"]},
        "oc_3": {"policy": "admin_only"},
    }, admins=["ou_admin"])

    assert runner._is_user_authorized(_feishu_source("oc_1", "ou_member")) is True
    assert runner._is_user_authorized(_feishu_source("oc_1", "ou_guest")) is False
    assert runner._is_user_authorized(_feishu_source("oc_2", "ou_guest")) is True
    assert runner._is_user_authorized(_feishu_source("oc_2", "ou_banned")) is False
    assert runner._is_user_authorized(_feishu_source("oc_3", "ou_admin")) is True
    assert runner._is_user_authorized(_feishu_source("oc_3", "ou_guest")) is False
    # The env allowlist still admits its own users everywhere, as upstream.
    assert runner._is_user_authorized(_feishu_source("oc_1", "ou_dm_owner")) is True


def test_group_rules_do_not_make_feishu_an_own_policy_adapter(monkeypatch):
    """Only the exact-rule grant is new: with no env allowlist, DMs and unlisted groups stay denied
    even under ``dm_policy``/``group_policy: allowlist`` (the own-policy trust path is not enabled)."""
    runner, adapter = _make_group_rule_runner(
        {"oc_1": {"policy": "open"}}, dm_policy="allowlist", group_policy="allowlist",
        default_group_policy="open",
    )

    assert adapter.enforces_own_access_policy is False
    assert adapter._group_policy == "allowlist"
    assert runner._is_user_authorized(_feishu_source("oc_2", "ou_guest")) is False
    assert runner._is_user_authorized(_feishu_source("oc_dm", "ou_guest", chat_type="dm")) is False
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    assert runner._is_user_authorized(_feishu_source("oc_2", "ou_guest")) is True


def test_group_rules_without_a_live_adapter_grant_nothing(monkeypatch):
    """No process-global config fallback: only the serving adapter's own rules count."""
    runner = _make_bare_runner()
    runner.adapters = {}
    runner._profile_adapters = {}
    runner.config = GatewayConfig(platforms={
        Platform.FEISHU: PlatformConfig(enabled=True, extra={"group_rules": {"oc_1": {"policy": "open"}}}),
    })

    assert runner._is_user_authorized(_make_feishu_human_source("ou_guest")) is False


def test_group_rules_are_scoped_to_the_serving_profile(monkeypatch):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    runner, _default_adapter = _make_group_rule_runner({})
    runner.config.multiplex_profiles = True
    client_adapter = FeishuAdapter(PlatformConfig(enabled=True, extra={"group_rules": {"oc_1": {"policy": "open"}}}))
    runner._profile_adapters = {"client": {Platform.FEISHU: client_adapter}}

    assert runner._is_user_authorized(_feishu_source("oc_1", "ou_guest", profile="client")) is True
    assert runner._is_user_authorized(_feishu_source("oc_1", "ou_guest", profile="default")) is False
    assert runner._is_user_authorized(_feishu_source("oc_1", "ou_guest", profile="serverrun")) is False


def test_feishu_bot_bypass_does_not_leak_to_other_platforms(monkeypatch):
    """FEISHU_ALLOW_BOTS=all must not authorize Telegram/Discord bot sources."""
    runner = _make_bare_runner()
    monkeypatch.setenv("FEISHU_ALLOW_BOTS", "all")

    telegram_bot = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="channel",
        user_id="999",
        is_bot=True,
    )
    assert runner._is_user_authorized(telegram_bot) is False
