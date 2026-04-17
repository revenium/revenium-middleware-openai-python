"""
Enforcement module for Revenium middleware circuit breaker.

Polls cost-limit rules from the Revenium API and caches them in-memory.
When a rule is tripped (cost limit exceeded), pre-call checks raise
ReveniumCostLimitExceeded to block the request before it reaches OpenAI.

Enabled via the REVENIUM_CIRCUIT_BREAKER_ENABLED environment variable.
"""

import logging
import os
import threading
import time
from typing import List, Optional
from urllib.parse import urlparse

import httpx

from .config import Config
from .exceptions import ReveniumCostLimitExceeded

logger = logging.getLogger("revenium_middleware.extension")

# How often to poll for updated rules (seconds)
_POLL_INTERVAL = 60
# How long a cached rule stays valid before a refresh is forced (seconds)
_CACHE_TTL = 120

# In-memory rule cache: list of rule dicts from the API
_cached_rules: List[dict] = []
_cache_lock = threading.Lock()
_cache_timestamp = 0.0

# Background polling thread
_poll_thread: Optional[threading.Thread] = None
_poll_lock = threading.Lock()
_stop_event = threading.Event()

# Serializes synchronous cache refreshes to prevent thundering herd
_refresh_lock = threading.Lock()

# Emit the missing-team-id warning only once
_team_id_warned = False


def is_circuit_breaker_enabled() -> bool:
    """Return True when the operator has opted in to enforcement."""
    return os.environ.get(Config.ENV_CIRCUIT_BREAKER_ENABLED, "").lower() in (
        "1", "true", "yes", "on",
    )


def _get_enforcement_base_url() -> str:
    """Derive the enforcement API base URL from the metering base URL.

    The metering SDK points at e.g. ``https://api.revenium.ai/meter/``.
    Enforcement rules live on the same origin at ``/v2/api/ai/enforcement-rules``.
    """
    metering_url = os.environ.get(
        "REVENIUM_METERING_BASE_URL", "https://api.revenium.ai/meter/"
    )
    parsed = urlparse(metering_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _fetch_rules() -> Optional[list]:
    """Fetch the current enforcement rules from the Revenium API.

    Returns a list of rule dicts on success (may be empty if the server
    has no rules configured), or None on failure so the caller can
    preserve the previous cache.
    """
    global _team_id_warned

    api_key = os.environ.get(Config.ENV_REVENIUM_API_KEY, "")
    if not api_key:
        logger.debug("No API key configured, skipping enforcement rule fetch")
        return None

    team_id = os.environ.get(Config.ENV_REVENIUM_TEAM_ID, "")
    if not team_id:
        if not _team_id_warned:
            logger.warning(
                "REVENIUM_TEAM_ID is not set — enforcement rule polling disabled. "
                "Set this to your hashed team ID to enable cost-limit enforcement."
            )
            _team_id_warned = True
        return None

    base_url = _get_enforcement_base_url()
    try:
        response = httpx.get(
            f"{base_url}/v2/api/ai/enforcement-rules/{team_id}",
            headers={"x-api-key": api_key},
            timeout=10,
        )
        # 204 No Content = no rules configured, cache empty list
        if response.status_code == 204:
            return []
        response.raise_for_status()
        data = response.json()
        return data.get("rules", [])
    except Exception:
        logger.debug("Failed to fetch enforcement rules, falling open", exc_info=True)
        return None


def _refresh_cache() -> None:
    """Refresh the in-memory rule cache (thread-safe)."""
    global _cached_rules, _cache_timestamp
    rules = _fetch_rules()
    with _cache_lock:
        if rules is not None:
            _cached_rules = rules
        _cache_timestamp = time.monotonic()


def _poll_loop() -> None:
    """Background loop that periodically refreshes the rule cache."""
    while not _stop_event.is_set():
        _refresh_cache()
        _stop_event.wait(_POLL_INTERVAL)


def _ensure_poller_running() -> None:
    """Start the background polling thread if it isn't already running."""
    global _poll_thread
    with _poll_lock:
        if _poll_thread is not None and _poll_thread.is_alive():
            return
        _stop_event.clear()
        _poll_thread = threading.Thread(
            target=_poll_loop,
            name="revenium-enforcement-poll",
            daemon=True,
        )
        _poll_thread.start()


def _get_rules() -> list:
    """Return the current cached rules, refreshing if stale."""
    now = time.monotonic()
    with _cache_lock:
        age = now - _cache_timestamp
        rules = list(_cached_rules)
    if age > _CACHE_TTL:
        # Serialize stale-cache refreshes to prevent thundering herd
        if _refresh_lock.acquire(blocking=False):
            try:
                _refresh_cache()
                with _cache_lock:
                    rules = list(_cached_rules)
            finally:
                _refresh_lock.release()
    return rules


def check_enforcement(usage_metadata: Optional[dict] = None) -> None:
    """Pre-call enforcement check.

    Call this before invoking the upstream OpenAI API.  If the circuit
    breaker is disabled or no rules are tripped, this is a no-op.

    Raises:
        ReveniumCostLimitExceeded: when a cost-limit rule blocks the call.
    """
    if not is_circuit_breaker_enabled():
        return

    _ensure_poller_running()
    rules = _get_rules()

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        # A rule is considered tripped when its ``blocked`` flag is set
        blocked = rule.get("blocked", False)
        if not blocked:
            continue

        credential = (usage_metadata or {}).get("subscriber_credential", "")
        rule_credential = rule.get("credential", "")

        # If the rule is scoped to a credential, only enforce for that credential
        if rule_credential and rule_credential != credential:
            continue

        limit_name = rule.get("name", "cost limit")
        raise ReveniumCostLimitExceeded(
            f"Request blocked by Revenium enforcement rule: {limit_name}"
        )


def stop_polling() -> None:
    """Gracefully stop the background polling thread."""
    _stop_event.set()
    if _poll_thread is not None:
        _poll_thread.join(timeout=5)
