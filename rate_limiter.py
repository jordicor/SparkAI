"""
Rate Limiter Module for Aurvek

In-memory rate limiter with multiple strategies:
- By IP (all attempts)
- By IP (failures only)
- By identifier (email/username)

For production with multiple workers, consider Redis-based implementation.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    In-memory rate limiter with multiple strategies.
    Thread-safe for single-process async applications.
    """

    def __init__(self):
        # {key: [timestamp, timestamp, ...]}
        self._attempts = defaultdict(list)
        self._last_cleanup = datetime.now()

    def _cleanup_old_entries(self, max_age_hours: int = 25):
        """Periodic cleanup of old entries to prevent memory bloat."""
        now = datetime.now()
        # Only cleanup every hour
        if (now - self._last_cleanup).total_seconds() < 3600:
            return

        cutoff = now - timedelta(hours=max_age_hours)
        keys_to_delete = []

        for key, timestamps in self._attempts.items():
            self._attempts[key] = [t for t in timestamps if t > cutoff]
            if not self._attempts[key]:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del self._attempts[key]

        self._last_cleanup = now
        logger.debug(f"Rate limiter cleanup: removed {len(keys_to_delete)} stale keys")

    def is_allowed(
        self,
        key: str,
        max_attempts: int,
        window_minutes: int
    ) -> Tuple[bool, int]:
        """
        Check if action is allowed and record attempt.

        Args:
            key: Unique identifier for this limit (e.g., "ip_all:login:1.2.3.4")
            max_attempts: Maximum attempts allowed in window
            window_minutes: Time window in minutes

        Returns:
            Tuple of (allowed: bool, remaining: int)
        """
        self._cleanup_old_entries()

        now = datetime.now()
        window_start = now - timedelta(minutes=window_minutes)

        # Filter to window
        self._attempts[key] = [t for t in self._attempts[key] if t > window_start]
        current_count = len(self._attempts[key])

        if current_count >= max_attempts:
            return False, 0

        self._attempts[key].append(now)
        return True, max_attempts - current_count - 1

    def record_failure(self, key: str):
        """Record a failure without checking limit (for failure-only tracking)."""
        self._attempts[key].append(datetime.now())

    def check_only(
        self,
        key: str,
        max_attempts: int,
        window_minutes: int
    ) -> Tuple[bool, int]:
        """Check limit without recording (for pre-check)."""
        now = datetime.now()
        window_start = now - timedelta(minutes=window_minutes)

        timestamps = [t for t in self._attempts.get(key, []) if t > window_start]
        current_count = len(timestamps)

        if current_count >= max_attempts:
            return False, 0
        return True, max_attempts - current_count

    def get_retry_after(self, key: str, window_minutes: int) -> int:
        """Get seconds until the oldest attempt in window expires."""
        now = datetime.now()
        window_start = now - timedelta(minutes=window_minutes)

        timestamps = [t for t in self._attempts.get(key, []) if t > window_start]
        if not timestamps:
            return 0

        oldest = min(timestamps)
        expires_at = oldest + timedelta(minutes=window_minutes)
        seconds_remaining = (expires_at - now).total_seconds()

        return max(0, int(seconds_remaining))


# Singleton instance
rate_limiter = RateLimiter()


# =============================================================================
# Configuration
# =============================================================================

class RateLimitConfig:
    """
    Centralized rate limit configuration.
    Format: (max_attempts, window_minutes)
    """

    # --- Login endpoints ---
    LOGIN_BY_IP_ALL = (20, 60)           # 20 attempts per hour per IP
    LOGIN_BY_IP_FAILURES = (5, 60)       # 5 failures per hour per IP
    LOGIN_BY_USER = (5, 1440)            # 5 per user per 24h

    # --- Registration endpoints ---
    REGISTER_BY_IP_ALL = (10, 60)        # 10 attempts per hour per IP
    REGISTER_BY_IP_FAILURES = (5, 60)    # 5 failures per hour per IP
    REGISTER_BY_EMAIL = (3, 1440)        # 3 per email per 24h

    # --- Magic link recovery ---
    RECOVERY_BY_IP = (10, 60)            # 10 per hour per IP
    RECOVERY_BY_EMAIL = (3, 1440)        # 3 per email per 24h

    # --- OAuth ---
    OAUTH_BY_IP = (15, 60)               # 15 per hour per IP
    OAUTH_CALLBACK_FAILURES = (5, 60)    # 5 failures per hour

    # --- Email verification ---
    VERIFY_BY_IP = (20, 60)              # 20 per hour per IP
    VERIFY_FAILURES = (10, 60)           # 10 failures per hour


# =============================================================================
# Helper Functions
# =============================================================================

def get_client_ip(request) -> str:
    """
    Extract client IP address, considering reverse proxies.

    Priority:
    1. CF-Connecting-IP (Cloudflare)
    2. X-Forwarded-For (first IP in chain)
    3. X-Real-IP (nginx)
    4. Direct client connection
    """
    # Cloudflare sends real client IP in this header
    cf_ip = request.headers.get("CF-Connecting-IP")
    if cf_ip:
        return cf_ip.strip()

    # X-Forwarded-For can have multiple IPs: client, proxy1, proxy2
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the first IP (original client)
        return forwarded.split(",")[0].strip()

    # X-Real-IP from nginx
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Direct connection
    if request.client and request.client.host:
        return request.client.host

    return "unknown"


def check_rate_limits(
    request,
    ip_limit: Tuple[int, int] = None,
    identifier: str = None,
    identifier_limit: Tuple[int, int] = None,
    action_name: str = "request"
) -> Optional[dict]:
    """
    Check multiple rate limits at once.

    Args:
        request: FastAPI/Starlette request object
        ip_limit: (max_attempts, window_minutes) for IP-based limit
        identifier: Email or username to track
        identifier_limit: (max_attempts, window_minutes) for identifier-based limit
        action_name: Name of action for logging and key generation

    Returns:
        None if all limits pass, or error dict if blocked.
    """
    ip = get_client_ip(request)

    # Check IP limit (all attempts)
    if ip_limit:
        key = f"ip_all:{action_name}:{ip}"
        allowed, remaining = rate_limiter.is_allowed(key, ip_limit[0], ip_limit[1])

        if not allowed:
            retry_after = rate_limiter.get_retry_after(key, ip_limit[1])
            logger.warning(
                f"Rate limit exceeded: {action_name} by IP {ip} "
                f"(limit: {ip_limit[0]}/{ip_limit[1]}min)"
            )
            return {
                "status": "error",
                "message": "Too many attempts. Please try again later.",
                "retry_after_seconds": retry_after
            }

    # Check identifier limit (email/username)
    if identifier and identifier_limit:
        key = f"id:{action_name}:{identifier.lower()}"
        allowed, remaining = rate_limiter.is_allowed(
            key, identifier_limit[0], identifier_limit[1]
        )

        if not allowed:
            retry_after = rate_limiter.get_retry_after(key, identifier_limit[1])
            logger.warning(
                f"Rate limit exceeded: {action_name} for identifier {identifier} "
                f"(limit: {identifier_limit[0]}/{identifier_limit[1]}min)"
            )
            return {
                "status": "error",
                "message": "Too many attempts for this account. Please try again later.",
                "retry_after_seconds": retry_after
            }

    return None  # All checks passed


def check_failure_limit(
    request,
    action_name: str,
    limit: Tuple[int, int]
) -> Optional[dict]:
    """
    Check failure-only limit (doesn't record, just checks).

    Args:
        request: FastAPI/Starlette request object
        action_name: Name of action for key generation
        limit: (max_failures, window_minutes)

    Returns:
        None if under limit, or error dict if blocked.
    """
    ip = get_client_ip(request)
    key = f"ip_fail:{action_name}:{ip}"

    allowed, _ = rate_limiter.check_only(key, limit[0], limit[1])

    if not allowed:
        retry_after = rate_limiter.get_retry_after(key, limit[1])
        logger.warning(
            f"Failure rate limit exceeded: {action_name} by IP {ip} "
            f"(limit: {limit[0]}/{limit[1]}min)"
        )
        return {
            "status": "error",
            "message": "Too many failed attempts. Please try again later.",
            "retry_after_seconds": retry_after
        }

    return None


def record_failure(request, action_name: str, identifier: str = None):
    """
    Record a failed attempt for failure-based limiting.

    Args:
        request: FastAPI/Starlette request object
        action_name: Name of action for key generation
        identifier: Optional email/username to also track by identifier
    """
    ip = get_client_ip(request)
    rate_limiter.record_failure(f"ip_fail:{action_name}:{ip}")

    if identifier:
        rate_limiter.record_failure(f"id_fail:{action_name}:{identifier.lower()}")

    logger.debug(f"Recorded failure: {action_name} from IP {ip}")
