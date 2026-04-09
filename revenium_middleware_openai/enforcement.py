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

from revenium_middleware import client

from .exceptions import ReveniumCostLimitExceeded

logger = logging.getLogger("revenium_middleware.extension")

# Environment variable that enables the circuit breaker
ENV_CIRCUIT_BREAKER_ENABLED = "REVENIUM_CIRCUIT_BREAKER_ENABLED"

# How often to poll for updated rules (seconds)
_POLL_INTERVAL = 60
# How long a cached rule stays valid before a refresh is forced (seconds)
_CACHE_TTL = 120

# In-memory rule cache: list of rule dicts from the API
_cached_rules = []
_cache_lock = threading.Lock()
_cache_timestamp = 0.0

# Background polling thread
_poll_thread = None
_stop_event = threading.Event()


def is_circuit_breaker_enabled() -> bool:
    """Return True when the operator has opted in to enforcement."""
    return os.environ.get(ENV_CIRCUIT_BREAKER_ENABLED, "").lower() in (
        "1", "true", "yes", "on",
    )


def _fetch_rules() -> list:
    """Fetch the current enforcement rules from the Revenium API.

    Returns a list of rule dicts.  On any failure the previous cache is
    preserved and an empty list is returned so callers can fall-open.
    """
    try:
        response = client.apis.meter_request(
            transaction_id="__enforcement_poll__",
            method="GET",
            resource="/v1/enforcement/rules",
            source_type="SDK_PYTHON",
        )
        # The API returns rules in the response body.
        # Adapt to however the metering client surfaces them.
        rules = getattr(response, "rules", None)
        if rules is not None:
            return list(rules)
        # If the response is dict-like, try key access
        if isinstance(response, dict):
            return response.get("rules", [])
        return []
    except Exception:
        logger.debug("Failed to fetch enforcement rules, falling open", exc_info=True)
        return []


def _refresh_cache() -> None:
    """Refresh the in-memory rule cache (thread-safe)."""
    global _cached_rules, _cache_timestamp
    rules = _fetch_rules()
    with _cache_lock:
        if rules:
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
        # Stale — do a synchronous refresh before returning
        _refresh_cache()
        with _cache_lock:
            rules = list(_cached_rules)
    return rules


def check_enforcement(usage_metadata: dict = None) -> None:
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
