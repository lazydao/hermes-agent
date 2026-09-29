"""TUI orphan cleanup must preserve live messaging leases in the same PID."""

import pytest

from hermes_cli import active_sessions
from tui_gateway import server


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_concurrent_sessions": 10})
    monkeypatch.setattr(server, "_sessions", {})
    # Leases written seconds ago are inside the self-orphan grace; these tests are about ownership.
    monkeypatch.setattr(active_sessions, "_SELF_ORPHAN_GRACE_SECONDS", 0.0)
    return tmp_path


def session_ids(home):
    return {
        entry["session_id"]
        for entry in active_sessions.active_session_registry_snapshot(home)
    }


def claim_gateway_lease(session_key):
    """The messaging gateway's own claim path (gateway/run_busy.py)."""
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(max_concurrent_sessions=10)
    runner._running_agents = {}
    source = SessionSource(platform=Platform.FEISHU, chat_id="chat", chat_type="group", user_id="user")
    lease, error = runner._claim_active_session_slot(session_key, source)
    assert error is None and lease is not None and lease.enabled
    return lease


def test_tui_sweep_preserves_live_feishu_and_reclaims_own_orphan(registry):
    gateway = claim_gateway_lease("feishu-turn")
    tui, error = server._claim_active_session_slot("tui-orphan", live_session_id="tile")
    assert error is None and tui.enabled

    server._reclaim_orphaned_leases()

    assert session_ids(registry) == {"feishu-turn"}
    assert not gateway.released
    gateway.release()
    assert session_ids(registry) == set()


def test_tui_transfer_keeps_owner_and_live_lease_survives_sweep(registry):
    lease, error = server._claim_active_session_slot("old-id", live_session_id="tile")
    assert error is None
    session = {"active_session_lease": lease, "source": "tui"}
    server._sessions["tile"] = session
    assert server._transfer_active_session_slot("tile", session, new_session_id="new-id")

    server._reclaim_orphaned_leases()
    assert session_ids(registry) == {"new-id"}

    server._sessions.clear()
    server._reclaim_orphaned_leases()
    assert session_ids(registry) == set()


def test_tui_sweep_across_profiles_spares_gateway_leases(registry):
    """Multiplex: one process leases in every served profile's registry; the sweep
    visits them all but reclaims only TUI-owned orphans in each."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    other_home = registry / "profiles" / "other"
    other_home.mkdir(parents=True)
    (other_home / "config.yaml").write_text("{}\n")  # identity marker: a live profile
    tui, error = server._claim_active_session_slot(
        "other-orphan", live_session_id="tile", profile_home=other_home,
    )
    assert error is None and tui.enabled
    token = set_hermes_home_override(other_home)
    try:
        gateway = claim_gateway_lease("other-feishu-turn")
    finally:
        reset_hermes_home_override(token)
    root_gateway = claim_gateway_lease("root-feishu-turn")

    server._reclaim_orphaned_leases()

    assert session_ids(other_home) == {"other-feishu-turn"}
    assert session_ids(registry) == {"root-feishu-turn"}
    gateway.release()
    root_gateway.release()
    assert session_ids(other_home) == set() and session_ids(registry) == set()
