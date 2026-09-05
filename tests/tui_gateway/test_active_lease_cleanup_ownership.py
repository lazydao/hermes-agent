"""TUI orphan cleanup must preserve live messaging leases in the same PID."""

import pytest

from hermes_cli import active_sessions
from tui_gateway import server


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_concurrent_sessions": 10})
    monkeypatch.setattr(server, "_sessions", {})
    return tmp_path


def session_ids(home):
    return {
        entry["session_id"]
        for entry in active_sessions.active_session_registry_snapshot(home)
    }


def test_tui_sweep_preserves_live_feishu_and_reclaims_own_orphan(registry):
    gateway, error = active_sessions.try_acquire_active_session(
        session_id="feishu-turn", surface="gateway:feishu",
        config={"max_concurrent_sessions": 10},
    )
    assert error is None
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


def test_tui_sweep_does_not_touch_other_profile_registry(registry):
    other_home = registry / "profiles" / "other"
    lease, error = server._claim_active_session_slot(
        "other-profile", live_session_id="tile", profile_home=other_home,
    )
    assert error is None and lease.enabled

    server._reclaim_orphaned_leases()
    assert session_ids(other_home) == {"other-profile"}
    lease.release()
    assert session_ids(other_home) == set()
