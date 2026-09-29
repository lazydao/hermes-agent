"""Opt-in cross-profile credential pools (``credential_pool_sharing.<provider>: global``).

Profiles keep their own pool by default; ``read_credential_pool`` falls back to the global root
only while a profile has no rows of its own. An opted-in provider instead always uses the root
``auth.json`` as its single authority — reads, merged writes and the ``auth.lock`` serializing
selection and single-use refreshes — and the profile's own rows for it are ignored, never
deleted. Split out of ``hermes_cli/auth.py`` and re-exported there; facade helpers are imported
lazily so ``hermes_cli.auth.<name>`` patches still intercept.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import FrozenSet, Optional

from hermes_cli.auth_constants import AUTH_LOCK_TIMEOUT_SECONDS


def globally_shared_pool_providers() -> FrozenSet[str]:
    """Providers the ACTIVE profile's config.yaml opts into the global pool."""
    try:
        from hermes_cli.config import read_raw_config_readonly
        sharing = read_raw_config_readonly().get("credential_pool_sharing")
    except Exception:
        return frozenset()
    if not isinstance(sharing, dict):
        return frozenset()
    return frozenset(
        str(provider).strip().lower()
        for provider, mode in sharing.items()
        if str(provider).strip() and isinstance(mode, str) and mode.strip().lower() == "global")


def credential_pool_is_globally_shared(provider_id: Optional[str]) -> bool:
    """Whether the active profile opted *provider_id* into the global pool (default profile too:
    it must reload and lock like every other participant)."""
    normalized = str(provider_id or "").strip().lower()
    return bool(normalized) and normalized in globally_shared_pool_providers()


def shared_credential_pool_path(provider_id: Optional[str]) -> Optional[Path]:
    """Root auth.json owning *provider_id*'s pool when a named profile opted it in, else None.

    None means "use the active store as usual": the provider is not shared, or the active home IS
    the root (default profile / classic mode), where shared and local are the same file."""
    from hermes_cli.auth import _global_auth_file_path
    if not credential_pool_is_globally_shared(provider_id):
        return None
    return _global_auth_file_path()


@contextmanager
def credential_pool_store_lock(
        provider_id: Optional[str], timeout_seconds: float = AUTH_LOCK_TIMEOUT_SECONDS):
    """Lock the active auth store, then the shared root store when *provider_id* opted in.

    Same active -> root order as ``_provider_state_transaction`` and the forked-grant heal, so a
    shared-pool transaction can never invert against them."""
    from hermes_cli.auth import _auth_store_lock
    shared_path = shared_credential_pool_path(provider_id)
    with _auth_store_lock(timeout_seconds):
        if shared_path is None:
            yield
            return
        with _auth_store_lock(timeout_seconds, target_path=shared_path):
            yield
