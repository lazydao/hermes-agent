"""Behavior tests for opt-in global credential pools across profiles."""

from __future__ import annotations

import base64
import json
import threading
import time
from pathlib import Path


def _jwt(exp: int, marker: str) -> str:
    def _part(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{_part({'alg': 'none'})}.{_part({'exp': exp, 'marker': marker})}.sig"


def _setup_shared_profile(tmp_path, monkeypatch, entries: list[dict]) -> tuple[Path, Path]:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "client"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_OAUTH_ACCESS_TOKEN", raising=False)
    (profile / "config.yaml").write_text(
        "credential_pool_strategies:\n"
        "  openai-codex: round_robin\n"
        "credential_pool_sharing:\n"
        "  openai-codex: global\n"
    )
    (root / "auth.json").write_text(json.dumps({
        "version": 1,
        "credential_pool": {"openai-codex": entries},
    }))
    (profile / "auth.json").write_text(json.dumps({
        "version": 1,
        "credential_pool": {
            "openai-codex": [{
                "id": "stale-local",
                "label": "stale-local",
                "auth_type": "api_key",
                "priority": 0,
                "source": "manual",
                "access_token": "stale-local-token",
            }],
        },
    }))
    return root, profile


def _api_key_entry(credential_id: str, priority: int) -> dict:
    return {
        "id": credential_id,
        "label": credential_id,
        "auth_type": "api_key",
        "priority": priority,
        "source": "manual",
        "access_token": f"token-{credential_id}",
    }


def test_concurrent_round_robin_selection_is_global_and_atomic(tmp_path, monkeypatch):
    root, _profile = _setup_shared_profile(
        tmp_path,
        monkeypatch,
        [_api_key_entry("account-a", 0), _api_key_entry("account-b", 1)],
    )

    from agent.credential_pool import load_pool

    first_pool = load_pool("openai-codex")
    second_pool = load_pool("openai-codex")
    barrier = threading.Barrier(3)
    selected: list[str] = []

    def _select(pool) -> None:
        barrier.wait()
        entry = pool.select()
        assert entry is not None
        selected.append(entry.id)

    threads = [
        threading.Thread(target=_select, args=(first_pool,)),
        threading.Thread(target=_select, args=(second_pool,)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert set(selected) == {"account-a", "account-b"}
    persisted = json.loads((root / "auth.json").read_text())
    assert {
        entry["id"]
        for entry in persisted["credential_pool"]["openai-codex"]
    } == {"account-a", "account-b"}


def test_shared_manual_codex_refresh_posts_once_and_propagates(tmp_path, monkeypatch):
    old_access = _jwt(int(time.time()) - 60, "old")
    new_access = _jwt(int(time.time()) + 3600, "new")
    root, _profile = _setup_shared_profile(tmp_path, monkeypatch, [{
        "id": "account-a",
        "label": "account-a",
        "auth_type": "oauth",
        "priority": 0,
        "source": "manual:device_code",
        "access_token": old_access,
        "refresh_token": "refresh-old",
    }])

    from agent.credential_pool import load_pool
    import hermes_cli.auth as auth_mod

    first_pool = load_pool("openai-codex")
    second_pool = load_pool("openai-codex")
    refresh_calls = 0
    refresh_calls_lock = threading.Lock()

    def _refresh(_access_token: str, refresh_token: str) -> dict:
        nonlocal refresh_calls
        assert refresh_token == "refresh-old"
        with refresh_calls_lock:
            refresh_calls += 1
        time.sleep(0.05)
        return {
            "access_token": new_access,
            "refresh_token": "refresh-new",
            "last_refresh": "2026-07-17T00:00:00Z",
        }

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", _refresh)
    barrier = threading.Barrier(3)
    refreshed_tokens: list[str] = []

    def _run_refresh(pool) -> None:
        barrier.wait()
        entry = pool.entries()[0]
        refreshed = pool._refresh_entry(entry, force=False)
        assert refreshed is not None
        refreshed_tokens.append(refreshed.refresh_token or "")

    threads = [
        threading.Thread(target=_run_refresh, args=(first_pool,)),
        threading.Thread(target=_run_refresh, args=(second_pool,)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert refresh_calls == 1
    assert refreshed_tokens == ["refresh-new", "refresh-new"]
    persisted = json.loads((root / "auth.json").read_text())
    shared_entry = persisted["credential_pool"]["openai-codex"][0]
    assert shared_entry["access_token"] == new_access
    assert shared_entry["refresh_token"] == "refresh-new"


def test_stale_shared_request_adopts_new_token_without_exhausting(tmp_path, monkeypatch):
    root, _profile = _setup_shared_profile(tmp_path, monkeypatch, [{
        "id": "account-a",
        "label": "account-a",
        "auth_type": "oauth",
        "priority": 0,
        "source": "manual:device_code",
        "access_token": "access-old",
        "refresh_token": "refresh-old",
    }])

    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    pool._current_id = "account-a"
    payload = json.loads((root / "auth.json").read_text())
    shared_entry = payload["credential_pool"]["openai-codex"][0]
    shared_entry["access_token"] = "access-new"
    shared_entry["refresh_token"] = "refresh-new"
    (root / "auth.json").write_text(json.dumps(payload))

    adopted = pool.mark_exhausted_and_rotate(
        status_code=401,
        api_key_hint="access-old",
        error_context={"reason": "token_invalidated"},
    )

    assert adopted is not None
    assert adopted.access_token == "access-new"
    persisted = json.loads((root / "auth.json").read_text())
    entry = persisted["credential_pool"]["openai-codex"][0]
    assert entry.get("last_status") in {None, "ok"}
    assert entry["refresh_token"] == "refresh-new"


def test_shared_rotation_persists_classified_failure_reason(tmp_path, monkeypatch):
    root, _profile = _setup_shared_profile(
        tmp_path,
        monkeypatch,
        [_api_key_entry("account-a", 0), _api_key_entry("account-b", 1)],
    )

    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    rotated = pool.mark_exhausted_and_rotate(
        status_code=403,
        credential_id="account-a",
        failure_reason="billing",
    )

    assert rotated is not None
    assert rotated.id == "account-b"
    persisted = json.loads((root / "auth.json").read_text())
    failed = next(
        entry
        for entry in persisted["credential_pool"]["openai-codex"]
        if entry["id"] == "account-a"
    )
    assert failed["last_status"] == "exhausted"
    assert failed["failure_reason"] == "billing"


def test_terminal_refresh_marks_shared_manual_credential_dead(tmp_path, monkeypatch):
    expired_access = _jwt(int(time.time()) - 60, "expired")
    root, _profile = _setup_shared_profile(tmp_path, monkeypatch, [{
        "id": "account-a",
        "label": "account-a",
        "auth_type": "oauth",
        "priority": 0,
        "source": "manual:device_code",
        "access_token": expired_access,
        "refresh_token": "refresh-revoked",
    }])

    from agent.credential_pool import load_pool
    import hermes_cli.auth as auth_mod
    from hermes_cli.auth import AuthError

    def _fail_refresh(_access_token: str, _refresh_token: str) -> dict:
        raise AuthError(
            "Refresh token was revoked",
            provider="openai-codex",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", _fail_refresh)
    pool = load_pool("openai-codex")
    assert pool._refresh_entry(pool.entries()[0], force=True) is None

    persisted = json.loads((root / "auth.json").read_text())
    entry = persisted["credential_pool"]["openai-codex"][0]
    assert entry["last_status"] == "dead"
    assert entry["last_error_reason"] == "invalid_grant"
