from __future__ import annotations

import asyncio
import fcntl
import os
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, GatewayRequestLockConfig, load_gateway_config
from gateway.request_lock import GatewayRequestLock, GatewayRequestLockTimeout


def _config(path: Path, timeout: float | None = 1.0) -> GatewayRequestLockConfig:
    return GatewayRequestLockConfig(
        path=path,
        wait_timeout_seconds=timeout,
        poll_interval_seconds=0.01,
    )


def test_gateway_config_parses_nested_request_lock(tmp_path):
    lock_path = tmp_path / "workspace.lock"

    config = GatewayConfig.from_dict(
        {
            "gateway": {
                "request_lock": {
                    "path": str(lock_path),
                    "wait_timeout_seconds": 30,
                    "poll_interval_seconds": 0.02,
                }
            }
        }
    )

    assert config.request_lock.path == lock_path
    assert config.request_lock.wait_timeout_seconds == 30
    assert config.request_lock.poll_interval_seconds == 0.02


def test_load_gateway_config_bridges_request_lock(tmp_path, monkeypatch):
    lock_path = tmp_path / "workspace.lock"
    (tmp_path / "config.yaml").write_text(
        "gateway:\n"
        "  request_lock:\n"
        f"    path: {lock_path}\n"
        "    wait_timeout_seconds: 45\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("gateway.config.get_hermes_home", lambda: tmp_path)

    config = load_gateway_config()

    assert config.request_lock.path == lock_path
    assert config.request_lock.wait_timeout_seconds == 45


@pytest.mark.asyncio
async def test_multiple_agent_requests_share_the_lock(tmp_path):
    lock_path = tmp_path / "workspace.lock"
    request_lock = GatewayRequestLock(_config(lock_path))

    first = await request_lock.acquire("first")
    second = await request_lock.acquire("second")
    exclusive_fd = os.open(lock_path, os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(exclusive_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(exclusive_fd)
        second.release()
        first.release()


@pytest.mark.asyncio
async def test_agent_request_waits_for_exclusive_maintenance(tmp_path):
    lock_path = tmp_path / "workspace.lock"
    exclusive_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(exclusive_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request_lock = GatewayRequestLock(_config(lock_path))

    acquire_task = asyncio.create_task(request_lock.acquire("waiting-request"))
    try:
        await asyncio.sleep(0.03)
        assert not acquire_task.done()
        fcntl.flock(exclusive_fd, fcntl.LOCK_UN)
        lease = await asyncio.wait_for(acquire_task, timeout=1)
        assert lease is not None
        lease.release()
    finally:
        os.close(exclusive_fd)


@pytest.mark.asyncio
async def test_agent_request_times_out_when_maintenance_never_releases(tmp_path):
    lock_path = tmp_path / "workspace.lock"
    exclusive_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(exclusive_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request_lock = GatewayRequestLock(_config(lock_path, timeout=0.03))

    try:
        with pytest.raises(GatewayRequestLockTimeout):
            await request_lock.acquire("timed-out-request")
    finally:
        os.close(exclusive_fd)
