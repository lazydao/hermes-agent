"""Cross-process request coordination for repository-wide maintenance.

When configured, every agent turn holds a POSIX advisory lock in shared mode.
An external maintenance process can take the same file in exclusive mode, so
multiple agent requests may coexist while a workspace-wide update cannot
overlap any of them.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


class GatewayRequestLockTimeout(TimeoutError):
    """Raised when an agent request cannot outwait external maintenance."""


class GatewayRequestLockLease:
    """One held shared lock; release is idempotent."""

    def __init__(self, fd: int, path: Path):
        self._fd = fd
        self.path = path

    def release(self) -> None:
        fd = self._fd
        if fd < 0:
            return
        self._fd = -1
        try:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            logger.warning("Failed to unlock request lock %s", self.path, exc_info=True)
        finally:
            try:
                os.close(fd)
            except OSError:
                logger.warning(
                    "Failed to close request lock %s", self.path, exc_info=True
                )


class GatewayRequestLock:
    """Acquire a configured advisory lock without blocking the event loop."""

    def __init__(self, config):
        self.path: Optional[Path] = getattr(config, "path", None)
        self.wait_timeout_seconds: Optional[float] = getattr(
            config, "wait_timeout_seconds", None
        )
        self.poll_interval_seconds = float(
            getattr(config, "poll_interval_seconds", 0.1)
        )

    @property
    def enabled(self) -> bool:
        return self.path is not None

    async def acquire(
        self,
        request_key: Optional[str] = None,
    ) -> Optional[GatewayRequestLockLease]:
        """Acquire a shared lease, waiting asynchronously for maintenance."""
        if self.path is None:
            return None

        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - POSIX-only feature
            raise RuntimeError(
                "gateway.request_lock requires POSIX fcntl support"
            ) from exc

        path = self.path
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        started = time.monotonic()
        deadline = (
            started + self.wait_timeout_seconds
            if self.wait_timeout_seconds is not None
            else None
        )
        waiting_logged = False

        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    waited = time.monotonic() - started
                    if waiting_logged:
                        logger.info(
                            "Agent request acquired shared lock after %.1fs: "
                            "request=%s lock=%s",
                            waited,
                            request_key or "?",
                            path,
                        )
                    return GatewayRequestLockLease(fd, path)
                except BlockingIOError:
                    if not waiting_logged:
                        logger.info(
                            "Agent request waiting for workspace maintenance: "
                            "request=%s lock=%s",
                            request_key or "?",
                            path,
                        )
                        waiting_logged = True

                    now = time.monotonic()
                    if deadline is not None and now >= deadline:
                        raise GatewayRequestLockTimeout(
                            "Timed out waiting for workspace maintenance to finish "
                            f"after {self.wait_timeout_seconds:g}s (lock: {path})"
                        )
                    sleep_for = self.poll_interval_seconds
                    if deadline is not None:
                        sleep_for = min(sleep_for, max(deadline - now, 0.0))
                    await asyncio.sleep(sleep_for)
        except BaseException:
            os.close(fd)
            raise
